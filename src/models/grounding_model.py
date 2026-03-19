"""
src/models/grounding_model.py
------------------------------
职责：顶层 3D Visual Grounding 模型，串联所有子模块。

数据流：
    RGB Image  (B,3,H,W)  -> ImageBackbone  -> img_feats  (B,C,H',W')
    PointCloud (B,N,6)    -> PCBackbone     -> pc_feats   (B,M,C), pc_xyz (B,M,3)
    Text tokens(B,L)      -> TextEncoder    -> text_feats (B,L,C)
    [img_feats, pc_feats, text_feats]
        -> MultimodalFusion (cross-attention, 文字作 query)
        -> fused_feats (B,M,C)
        -> BBox3DHead
        -> pred_boxes (B,7): cx,cy,cz,l,w,h,yaw

用法：
    model = GroundingModel(cfg.model)
    output = model(image, pointcloud, text)
    # output: {"pred_boxes": Tensor(B,7)}
"""
