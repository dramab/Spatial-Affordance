"""
tests/test_dataset.py
----------------------
职责：测试数据集加载和预处理流程的正确性。

测试内容：
- test_dataset_init：验证 SpatialAffordanceDataset 能正常初始化
- test_getitem：验证 __getitem__ 返回正确的 key 和 tensor shape
- test_augmentation：验证增强后 bbox3d 坐标变换的一致性
- test_collate_fn：验证 collate_fn 正确处理变长点云（padding）
- test_dataloader：验证 DataLoader 能正常迭代，batch shape 正确

用法：
    pytest tests/test_dataset.py -v
"""
