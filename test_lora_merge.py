import torch
from config import Config
from model_builder import build_maddn_net
from lora import clone_and_merge_lora, get_lora_parameters

"""验证合并前后 Linear LoRA 的等价性（仅 Linear 已合并）"""

def test_merge_equivalence():
    cfg = Config()
    cfg.lora.enable_lora = True
    cfg.lora.apply_to_maddn = True
    cfg.lora.apply_to_backbone = False
    cfg.lora.rank = 4
    cfg.lora.alpha = 8
    model = build_maddn_net(cfg).eval()
    # 随机输入
    x = torch.randn(2, 1, 128, 128, 128)
    with torch.no_grad():
        out_before = model(x)
    merged = clone_and_merge_lora(model).eval()
    with torch.no_grad():
        out_after = merged(x)
    diff = (out_before - out_after).abs().max().item()
    print(f"Max abs difference (before vs merged clone): {diff:.6f}")
    # 理论上为 0（Linear 完全合并），允许数值误差阈值 1e-5
    assert diff < 1e-4, "Merged model output deviates from original beyond tolerance." 
    print("Merge equivalence test passed.")

if __name__ == '__main__':
    test_merge_equivalence()
