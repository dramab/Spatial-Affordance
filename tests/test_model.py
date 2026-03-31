"""
tests/test_model.py
--------------------
职责：测试模型各子模块和整体前向传播的正确性。

测试内容：
- test_image_backbone：验证 ImageBackbone 输出 shape
- test_pc_backbone：验证 PCBackbone 输出 shape
- test_text_encoder：验证 TextEncoder 输出 shape 和 mask
- test_multimodal_fusion：验证 MultimodalFusion 输出 shape
- test_bbox3d_head：验证 BBox3DHead 输出 shape (B, 7)
- test_grounding_model_forward：验证整体模型端到端前向传播
- test_grounding_model_output_range：验证输出 bbox 尺寸为正值

用法：
    pytest tests/test_model.py -v
"""
