"""
tests/test_image_backbone.py
----------------------------
职责：测试 ResNet 图像编码器及其 Transformer token 导出接口。

测试内容：
- test_image_backbone_outputs_feature_map_and_tokens：验证输出 shape 与 token 展平逻辑
- test_image_backbone_freeze_only_backbone：验证冻结时仅主干参数被冻结

用法：
    pytest tests/test_image_backbone.py -v
"""

import torch

from src.models.backbones import ImageBackbone


def test_image_backbone_outputs_feature_map_and_tokens():
    """
    作用：验证图像编码器会输出融合所需的 token 接口。

    输入：
        无，内部构造随机图像 batch
    输出：
        无，通过断言验证结果
    """
    backbone = ImageBackbone({
        "type": "resnet50",
        "pretrained": False,
        "out_channels": 128,
    })
    images = torch.randint(0, 255, (2, 3, 224, 224), dtype=torch.uint8)

    outputs = backbone(images)

    assert outputs["tokens"].shape == (2, 49, 128)
    assert outputs["token_mask"].shape == (2, 49)
    assert outputs["token_pos"].shape == (2, 49, 2)
    assert torch.all(outputs["token_mask"])


def test_image_backbone_freeze_only_backbone():
    """
    作用：验证 freeze=True 时仅主干网络被冻结。

    输入：
        无，内部构造编码器实例
    输出：
        无，通过断言验证结果
    """
    backbone = ImageBackbone({
        "type": "resnet50",
        "pretrained": False,
        "out_channels": 64,
        "freeze": True,
    })

    assert all(not param.requires_grad for param in backbone.backbone.parameters())
    assert all(param.requires_grad for param in backbone.proj.parameters())
