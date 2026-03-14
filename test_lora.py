import torch
from config import Config
from model_builder import build_maddn_net, count_parameters
from lora import get_lora_parameters


def test_lora_injection():
    cfg = Config()
    cfg.lora.enable_lora = True
    cfg.lora.apply_to_backbone = False
    cfg.lora.apply_to_maddn = True
    cfg.lora.rank = 4
    cfg.lora.alpha = 8
    model = build_maddn_net(cfg)
    model.eval()
    params_before = count_parameters(model)
    lora_params = get_lora_parameters(model)
    assert len(lora_params) > 0, "LoRA parameters should be injected"
    # Forward shape check
    x = torch.randn(2, 1, 128, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out.shape[-1] == cfg.backbone.num_classes, "Output class dim mismatch"
    print("LoRA injection test passed. Trainable params:", params_before)

if __name__ == '__main__':
    test_lora_injection()
