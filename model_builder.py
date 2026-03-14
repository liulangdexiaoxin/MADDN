import torch
import torch.nn as nn
from resnet import resnet18_3d, resnet50_3d
from maddn_net_simple import MADDNShard
from maddn_net import MADDN
from lora import inject_lora, get_lora_parameters

def build_backbone(config):
    """构建ResNet backbone"""
    if config.backbone.model_type == "resnet18_3d":
        model = resnet18_3d(
            num_classes=config.backbone.num_classes,
            in_channels=config.backbone.in_channels
        )
    elif config.backbone.model_type == "resnet50_3d":
        model = resnet50_3d(
            num_classes=config.backbone.num_classes,
            in_channels=config.backbone.in_channels
        )
    else:
        raise ValueError(f"Unsupported model type: {config.backbone.model_type}")
    
    # 加载预训练权重
    if config.backbone.pretrained and config.backbone.pretrained_path:
        print(f"Loading pretrained weights from {config.backbone.pretrained_path}")
        checkpoint = torch.load(config.backbone.pretrained_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        # 移除可能存在的模块前缀
        for k in state_dict.keys():
            if k.startswith('module.'):
                state_dict[k[7:]] = state_dict.pop(k)
                
        # 加载权重
        model.load_state_dict(state_dict, strict=False)
        
    return model

def build_maddn_net(config):
    """构建完整的MADDN模型"""
    backbone = build_backbone(config)

    maddn_cfg = getattr(config, 'maddn', None)

    if maddn_cfg and maddn_cfg.use_shared_transformer:
        model = MADDNShard(
            backbone=backbone,
            num_classes=config.backbone.num_classes,
            embed_dim=maddn_cfg.embed_dim,
            depth=maddn_cfg.depth,
            num_heads=maddn_cfg.num_heads
        )
    else:
        model = MADDN(
            backbone=backbone,
            num_classes=config.backbone.num_classes,
            embed_dim=maddn_cfg.embed_dim if maddn_cfg else 256,
            depth=maddn_cfg.depth if maddn_cfg else 2,
            num_heads=maddn_cfg.num_heads if maddn_cfg else 8
        )

    # ===== LoRA 注入 =====
    if getattr(config, 'lora', None) and config.lora.enable_lora:
        linear_total = 0
        conv_total = 0
        print(f"[LoRA] 开始注入 LoRA 适配器...")
        print(f"[LoRA] 配置: rank={config.lora.rank}, alpha={config.lora.alpha}, freeze_base={config.lora.freeze_base}")
        print(f"[LoRA] 目标模块: {config.lora.target_modules}")
        
        if config.lora.apply_to_backbone:
            print("[LoRA] 在 backbone 中注入...")
            l_cnt, c_cnt = inject_lora(
                model.backbone,
                rank=config.lora.rank,
                alpha=config.lora.alpha,
                dropout=config.lora.dropout,
                target_substrings=config.lora.target_modules,
                include_conv3d=True,
                verbose=True
            )
            linear_total += l_cnt; conv_total += c_cnt
            print(f"[LoRA] Backbone 注入结果: Linear={l_cnt}, Conv3d={c_cnt}")

        if config.lora.apply_to_maddn:
            print("[LoRA] 在 MADDN fusion 网络中注入...")
            target_module = getattr(model, 'fusion_network', None)
            if target_module is not None:
                l_cnt, c_cnt = inject_lora(
                    target_module,
                    rank=config.lora.rank,
                    alpha=config.lora.alpha,
                    dropout=config.lora.dropout,
                    target_substrings=config.lora.target_modules,
                    include_conv3d=False,
                    verbose=True
                )
                linear_total += l_cnt; conv_total += c_cnt
                print(f"[LoRA] MADDN fusion 注入结果: Linear={l_cnt}, Conv3d={c_cnt}")
            else:
                print("[LoRA] Warning: 未找到 fusion_network 模块")
        print(f"[LoRA] 注入完成: Linear={linear_total}, Conv3d(1x1)={conv_total}, rank={config.lora.rank}, alpha={config.lora.alpha}")
        
        if config.lora.freeze_base:
            # 冻结除 LoRA 分支和分类头之外的参数
            lora_param_ids = {id(p) for p in get_lora_parameters(model)}
            frozen_count = 0
            trainable_count = 0
            for name, p in model.named_parameters():
                if id(p) in lora_param_ids:
                    p.requires_grad = True
                    trainable_count += 1
                elif any(h in name for h in ['classifier', 'fc']):
                    p.requires_grad = True
                    trainable_count += 1
                else:
                    p.requires_grad = False
                    frozen_count += 1
            print(f"[LoRA] 已冻结基础权重: frozen={frozen_count}, trainable={trainable_count} (LoRA分支+分类头)。")

    # ===== 加载 MADDN 预训练权重（与 backbone 独立） =====
    maddn_cfg = getattr(config, 'maddn', None)
    if maddn_cfg and getattr(maddn_cfg, 'pretrained', False) and getattr(maddn_cfg, 'pretrained_path', None):
        ckpt_path = maddn_cfg.pretrained_path
        if not torch.cuda.is_available():
            map_loc = 'cpu'
        else:
            map_loc = 'cuda'
        try:
            print(f"[MADDN Pretrained] 加载 MADDN 预训练权重: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=map_loc)
            if 'state_dict' in ckpt:
                state = ckpt['state_dict']
            elif 'model' in ckpt:
                state = ckpt['model']
            else:
                state = ckpt
            # 移除可能的 module. 前缀
            cleaned = {}
            for k,v in state.items():
                new_k = k[7:] if k.startswith('module.') else k
                cleaned[new_k] = v
            # 若只希望加载 MADDN 相关层, 过滤 backbone 和分类头(可选)
            load_dict = {}
            ignore_classifier = getattr(maddn_cfg, 'ignore_classifier', True)
            for k,v in cleaned.items():
                if ignore_classifier and ('classifier' in k or k.startswith('fc.') or k.startswith('fc')):
                    continue
                if k in model.state_dict():
                    load_dict[k] = v
            model.eval()
            load_result = model.load_state_dict(load_dict, strict=getattr(maddn_cfg, 'load_strict', False))
            model.train()
            missing = getattr(load_result, 'missing_keys', [])
            unexpected = getattr(load_result, 'unexpected_keys', [])
            print(f"[MADDN Pretrained] 已加载参数数: {len(load_dict)}; missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                print(f"[MADDN Pretrained] Missing keys (前10个): {missing[:10]}")
            if unexpected:
                print(f"[MADDN Pretrained] Unexpected keys (前10个): {unexpected[:10]}")
        except Exception as e:
            print(f"[MADDN Pretrained] 加载失败: {e}")
    
    return model

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)