# 模型构建说明

本文档说明 `src/models` 目录下模型构建部分的实现逻辑，重点描述模块职责、组装方式、前向数据流以及关键输入输出约定，便于后续维护、扩展和排查问题。

## 1. 模块总览

`src/models` 实现的是一个面向 3D BBox 预测的多模态模型。当前模型同时支持点云、图像、文本三种输入模态，并通过统一的 Transformer 编码器进行融合，最终输出 7 维 3D 包围框参数。

目录结构如下：

```text
src/models/
|-- __init__.py
|-- multimodal_model.py
|-- backbones/
|   |-- __init__.py
|   |-- image_backbone.py
|   |-- pc_backbone.py
|   |-- voxelnet_encoder.py
|   `-- voxel_token_utils.py
|-- encoders/
|   |-- __init__.py
|   `-- text_encoder.py
|-- fusion/
|   |-- __init__.py
|   `-- multimodal_transformer.py
`-- heads/
    |-- __init__.py
    |-- center3d_head.py
    `-- size_yaw_head.py
```

从功能上看，可以分为五个部分：

- `multimodal_model.py`：模型总入口，负责组装全部子模块并定义前向主流程。
- `backbones/`：点云和图像的底层特征提取模块。
- `encoders/`：文本编码模块。
- `fusion/`：多模态 token 对齐、拼接与 Transformer 编解码模块。
- `heads/`：检测头，负责输出最终 3D BBox 参数。

模型总入口为 `src/models/multimodal_model.py` 中的 `MultimodalModel`。

## 2. 整体构建流程

`MultimodalModel` 初始化时会从配置中读取各分支子配置，并完成如下组装：

1. 构建图像主干 `ImageBackbone`
2. 构建点云主干 `PCBackbone`
3. 构建文本编码器 `TextEncoder`
4. 根据三种模态输出维度，构建统一融合编码器 `UnifiedMultimodalEncoder`
5. 构建 query 解码器 `MultimodalDecoder`
6. 构建目标物体 center、放置位 center 与 size/yaw 输出头

对应关系如下：

```text
points_xyz / point_feats ----> PCBackbone ----> 点云 token
images ----------------------> ImageBackbone -> 图像 token
text_inputs -----------------> TextEncoder ---> 文本 token

三种 token ------------------> UnifiedMultimodalEncoder -> memory
memory ----------------------> MultimodalDecoder --------> object / placement queries
object query ----------------> Center3DHead -------------> pred_object_centers_norm
placement query -------------> Center3DHead -------------> pred_placement_centers_norm
object + placement queries --> SizeYawHead --------------> pred_size_yaw_norm
placement center + size/yaw -----------------------------> pred_boxes_norm
```

其中，`pred_boxes_norm` 是归一化坐标系下的预测框；世界坐标恢复由推理或评测脚本在模型外部完成。

## 3. 总入口 MultimodalModel

`MultimodalModel` 是整个模型的统一封装，主要职责有三项：

- 读取配置并构造所有子模块
- 将不同模态输入整理成统一 token 接口
- 串联融合、解码、回归和可选去归一化流程

其核心成员包括：

- `self.image_backbone`
- `self.pc_backbone`
- `self.text_encoder`
- `self.multimodal_encoder`
- `self.decoder`
- `self.object_center_head`
- `self.placement_center_head`
- `self.size_yaw_head`

这里有一个重要设计点：点云、图像、文本分支并不是直接输出最终结果，而是都被规整为统一格式的 token 字典，再送入融合模块。这样做的好处是：

- 降低不同模态之间的耦合
- 便于新增模态或替换现有编码器
- 让融合模块只依赖统一 token 接口，而不依赖具体 backbone 细节

当前各模态对融合模块提供的公共字段主要包括：

- `tokens`：形状通常为 `Tensor(B, L, C)` 的 batch-first token 序列
- `token_mask`：形状为 `BoolTensor(B, L)`，`True` 表示有效 token
- `token_pos`：位置编码信息，点云为 3D 坐标，图像为 2D 坐标，文本当前没有显式位置输入给融合层

## 4. 点云分支

### 4.1 PCBackbone

`PCBackbone` 是点云 backbone 的统一包装入口。当前实现只支持一种类型：

- `voxelnet`

这个包装层本身不承担复杂计算，主要作用是：

- 解析点云 backbone 配置
- 根据配置构造具体实现
- 对外暴露统一的调用入口

因此，如果后续需要接入其他点云编码器，例如 PointNet++、SparseConv 或其他 voxel-based 编码器，可以继续沿用 `PCBackbone` 作为统一调度入口。

### 4.2 VoxelNetEncoder

`VoxelNetEncoder` 是当前点云分支的核心实现，位于 `src/models/backbones/voxelnet_encoder.py`。它整体采用了 VoxelNet 前半段的思路，但目标不是直接完成检测，而是生成可供后续 Transformer 使用的体素级特征。

它的处理流程可以拆成四步：

1. 将整场景点云体素化
2. 在每个体素内用 VFE/SVFE 聚合点特征
3. 将稀疏体素特征散射回稠密体素网格
4. 用 3D 卷积进一步提取稠密体素特征

#### 4.2.1 体素网格规格

`VoxelGridSpec` 用来描述固定体素网格的几何规格，主要由两部分定义：

- `voxel_size`：体素大小，顺序为 `(x, y, z)`
- `point_cloud_range`：点云范围，顺序为 `(x_min, y_min, z_min, x_max, y_max, z_max)`

通过这两项可以计算出：

- `grid_shape_xyz`
- `grid_shape_dhw`

其中 `grid_shape_dhw` 的顺序是 `(depth_z, height_y, width_x)`，这是为了适配 3D 卷积输入格式。

#### 4.2.2 点云体素化

`voxelize_points` 负责把输入点云转换成固定范围内的稀疏体素表示，主要逻辑包括：

- 过滤掉超出 `point_cloud_range` 的点
- 根据体素大小计算每个点所在体素索引
- 统计唯一体素坐标
- 如果体素数超出 `max_voxels`，只保留前 `max_voxels` 个体素
- 对单个体素内的点做截断，最多保留 `max_points_per_voxel` 个点
- 构造体素内点特征

当前每个点的体素特征由以下部分拼接得到：

- 原始点坐标 `xyz`
- 相对体素内均值坐标 `rel_xyz`
- 可选附加点特征 `point_feats`

因此输入到 VFE 的点特征维度为：

```text
input_feature_dim = 6 + extra_feature_dim
```

其中前 6 维来自 `xyz + rel_xyz`。

#### 4.2.3 体素内特征聚合

体素内特征聚合由以下模块完成：

- `FCN`
- `VFE`
- `SVFE`

它们的作用分别是：

- `FCN`：对体素内每个点做逐点线性映射、归一化和激活
- `VFE`：将逐点特征和体素级聚合特征拼接，形成带局部上下文的点特征
- `SVFE`：堆叠两个 `VFE` 后，再经过一个 `FCN`，最终把每个体素编码成一个定长向量

经过 `SVFE` 后，每个有效体素得到一个稀疏体素特征向量，形状为：

```text
Tensor(K, C)
```

其中：

- `K` 表示当前 batch 中所有有效体素总数
- `C` 表示体素编码通道数

#### 4.2.4 稀疏转稠密与 3D 卷积

`voxel_indexing` 会将稀疏体素特征按照 `(batch, z, y, x)` 索引散射回稠密 5D 网格：

```text
Tensor(B, C, D, H, W)
```

之后通过 `ConvMiddleLayers` 执行多层 3D 卷积提取上下文信息，输出更强的体素级表征。

最终 `VoxelNetEncoder` 主要返回：

- `dense_voxel_feats`：稠密体素特征
- `valid_mask`：哪些体素位置有效
- `sparse_voxel_feats`：有效体素处的稀疏特征
- `sparse_coords`：有效体素的稀疏坐标
- `voxel_num_points`
- `points_per_batch`
- `dropped_points`
- `grid_meta`

### 4.3 体素特征转 Transformer Token

点云分支不会直接把 5D 稠密特征送入融合模块，而是先经过 `voxel_token_utils.py` 中的工具函数整理为 token 序列。

其中涉及两个主要函数：

- `flatten_voxel_grid_for_transformer`
- `build_padded_voxel_tokens`

#### 4.3.1 flatten_voxel_grid_for_transformer

这个函数会从稠密体素网格中取出有效体素位置的特征，并生成：

- `voxel_tokens`：有效体素特征
- `voxel_coords`：体素中心坐标，顺序为 `(x, y, z)`
- `token_mask`：有效 token 标记
- `sparse_coords`：稀疏坐标 `(batch, z, y, x)`

其中体素中心坐标由 `_compute_voxel_centers` 根据体素索引和网格规格反算得到。

#### 4.3.2 build_padded_voxel_tokens

这个函数会进一步把所有有效体素按 batch 重组为统一长度的 batch-first token 序列，并做 padding，输出：

- `tokens`：`Tensor(B, L, C)`
- `token_mask`：`BoolTensor(B, L)`
- `token_pos`：归一化到 `[-1, 1]` 的体素中心坐标
- `token_coords`：原始体素中心坐标
- `sparse_coords`：padding 后的稀疏索引
- `token_counts`：每个 batch 的有效 token 数

因此，点云模态最终提供给融合模块的是体素级 token，而不是原始点级 token。

## 5. 图像分支

图像分支由 `src/models/backbones/image_backbone.py` 中的 `ImageBackbone` 实现。当前支持的图像 backbone 类型包括：

- `resnet50`
- `resnet101`

### 5.1 图像特征提取

`ImageBackbone` 以 torchvision 中的 ResNet 为基础，保留从 `conv1` 到 `layer4` 的主干结构，然后通过一个 `1x1 Conv2d` 将输出通道统一映射到配置指定的 `out_channels`。

在输入侧，图像会先经过 `_normalize_images` 处理：

- 输入必须是 `Tensor(B, 3, H, W)`
- 支持 `uint8` 或 `float`
- 若像素最大值大于 1，则自动除以 255
- 随后按 ImageNet 均值方差做归一化

### 5.2 图像 token 构造

卷积特征图输出后，会被展平为 token 序列：

```text
feature_map: Tensor(B, C, H', W')
tokens:      Tensor(B, H' * W', C)
```

同时，图像分支还会构造：

- `token_mask`：全部为 `True`
- `token_pos`：归一化到 `[-1, 1]` 的二维网格坐标
- `feat_hw`：特征图大小 `(H', W')`

因此，图像模态对融合模块提供的是规则二维网格上的 patch-like token。

## 6. 文本分支

文本分支由 `src/models/encoders/text_encoder.py` 中的 `TextEncoder` 实现，底层依赖 HuggingFace 的 `AutoTokenizer` 和 `AutoModel`。

### 6.1 文本编码流程

初始化时，`TextEncoder` 会根据配置加载：

- tokenizer
- 预训练语言模型
- 输出投影层 `proj`

具体流程如下：

1. 若输入是字符串列表，则先调用 tokenizer 做批量分词
2. 若输入已经是 tokenizer 输出字典，则直接使用
3. 将编码结果移动到当前模块所在设备
4. 调用 HuggingFace 模型获取 `last_hidden_state`
5. 用线性层投影到统一输出维度

最终输出：

- `tokens`：`Tensor(B, L, C)`
- `token_mask`：由 `attention_mask` 转成 `BoolTensor(B, L)`

### 6.2 当前文本位置编码策略

当前文本分支在进入融合模块时，不额外传入显式 `token_pos`。也就是说，文本 token 仅经过：

- 文本 backbone 编码
- 线性投影到统一维度
- 模态 embedding 注入

其顺序信息主要依赖底层预训练语言模型本身产生的上下文表征，而不是在融合阶段再次显式加入位置编码。

## 7. 多模态融合模块

多模态融合模块位于 `src/models/fusion/multimodal_transformer.py`，包括两个核心类：

- `UnifiedMultimodalEncoder`
- `MultimodalDecoder`

### 7.1 UnifiedMultimodalEncoder

`UnifiedMultimodalEncoder` 的职责是把三种模态的 token 投影到同一隐藏维度，并编码成共享 memory。

#### 7.1.1 统一维度映射

三种模态分别使用独立的线性层投影到统一维度：

- `point_proj`
- `image_proj`
- `text_proj`

这样做的意义是，不要求各个 backbone 输出相同通道数，只要最后都能映射到统一 `hidden_dim` 即可。

#### 7.1.2 位置与模态编码

除了特征投影外，融合编码器还会给不同模态补充额外信息：

- 点云 token：通过 `point_pos_proj` 将 3D 坐标映射到隐藏空间
- 图像 token：通过 `image_pos_proj` 将 2D 坐标映射到隐藏空间
- 文本 token：当前不额外叠加显式位置投影
- 三种模态都会加上 `modality_embed`

因此，对点云和图像而言，进入 Transformer 编码器前的 token 形式可以理解为：

```text
fused_token = feature_proj(token) + position_proj(token_pos) + modality_embed
```

对文本而言则为：

```text
fused_token = feature_proj(token) + modality_embed
```

#### 7.1.3 拼接与共享编码

所有有效模态会按固定顺序拼接：

1. point
2. image
3. text

拼接后形成统一序列：

- `tokens`：`Tensor(B, L, H)`
- `memory_mask`：`BoolTensor(B, L)`

随后进入 `nn.TransformerEncoder` 得到编码结果 `memory`，并通过 `LayerNorm` 归一化。

输出字段包括：

- `memory`
- `memory_mask`
- `memory_pos`
- `modality_lengths`

其中 `modality_lengths` 记录各模态在拼接前的长度，便于后续分析或调试。

### 7.2 MultimodalDecoder

`MultimodalDecoder` 基于共享 memory 执行 query 解码，结构上类似 DETR 风格的 Transformer decoder。

它的主要特征有：

- 维护固定数量的可学习 query embedding
- 使用全零 target 加 query embedding 作为 decoder 输入
- 从共享 memory 中聚合与 query 相关的信息

主要配置项包括：

- `hidden_dim`
- `num_layers`
- `num_heads`
- `dropout`
- `num_queries`

当前默认 `num_queries=2`，固定语义为 `query 0 = object`、`query 1 = placement`。模型不做 query matching，而是直接按这两个固定位置读取目标物体信息和放置位信息。

输出字段包括：

- `decoder_tokens`

## 8. 检测头 Center3DHead 与 SizeYawHead

检测头位于 `src/models/heads/`，负责把两个 decoder query 映射为最终 3D 包围框参数。

当前头部拆成三部分：

- `object_center_head`：接收 object query，输出目标物体原始中心 `pred_object_centers_norm`
- `placement_center_head`：接收 placement query，输出放置位中心 `pred_placement_centers_norm`
- `size_yaw_head`：接收 object query 与 placement query 的拼接特征，输出 `pred_size_yaw_norm`

最终 `pred_boxes_norm` 由放置位中心和 size/yaw 拼接得到，格式固定为：

```text
(cx, cy, cz, log_l, log_w, log_h, yaw)
```

其中：

- `cx, cy, cz` 表示放置位框中心
- `log_l, log_w, log_h` 表示归一化尺寸的 log 回归值
- `yaw` 表示朝向角

尺寸项不在模型内部做正值约束，训练目标与推理恢复逻辑统一使用 `log(size_norm)`。

## 9. 前向数据流

`MultimodalModel.forward` 的执行顺序比较清晰，可以概括为如下步骤。

### 9.1 模态编码

首先调用三个内部辅助函数：

- `_encode_point_inputs`
- `_encode_image_inputs`
- `_encode_text_inputs`

它们分别完成：

- 点云输入编码和 token 化
- 图像输入编码和 token 化
- 文本输入编码

如果某个模态输入为 `None`，则该分支直接返回 `None`，融合阶段会自动跳过该模态。

### 9.2 融合编码

三种模态字典输入 `self.multimodal_encoder` 后，生成：

- `memory`
- `memory_mask`
- `memory_pos`
- `modality_lengths`

其中 `memory` 是后续 decoder 的共享上下文。

### 9.3 Query 解码

`self.decoder` 接收 `memory` 与 `memory_mask`，输出：

- `decoder_tokens`

### 9.4 3D 框回归

模型按固定语义拆分 `decoder_tokens`：

- object query 送入 `object_center_head`，得到 `pred_object_centers_norm`
- placement query 送入 `placement_center_head`，得到 `pred_placement_centers_norm`
- 两个 query 拼接后送入 `size_yaw_head`，得到 `pred_size_yaw_norm`
- `pred_placement_centers_norm` 与 `pred_size_yaw_norm` 拼接为 `pred_boxes_norm`

### 9.5 坐标恢复

`forward` 只输出 normalized 结果，世界坐标恢复由推理或评测脚本处理：

- 中心坐标使用 `scene_center` 与 `scene_scale` 反变换
- 框尺寸先对 log-size 做 `exp`，再使用 `scene_scale` 反变换
- `yaw` 使用数据集记录的 yaw 归一化比例反变换

因此模型输出中保留：

- `pred_boxes_norm`
- `pred_object_centers_norm`
- `pred_placement_centers_norm`
- `pred_size_yaw_norm`

## 10. 输入输出约定

### 10.1 forward 输入

`MultimodalModel.forward` 当前支持以下输入：

- `points_xyz`：`Tensor(B, N, 3)`
- `point_feats`：`Tensor(B, N, F)`，可选
- `images`：`Tensor(B, 3, H, W)`，可选
- `text_inputs`：`list[str]` 或 tokenizer 输出字典，可选

其中：

- 至少需要提供一种模态输入，否则融合模块会报错

### 10.2 forward 输出

当前 `forward` 返回一个字典，包含：

- `memory`
- `memory_mask`
- `modality_lengths`
- `decoder_tokens`
- `pred_boxes_norm`
- `pred_object_centers_norm`
- `pred_placement_centers_norm`
- `pred_size_yaw_norm`

这种设计保留了较多中间结果，优点是方便调试、可视化和损失扩展；代价是返回结构相对较重。

## 11. 配置项来源

`MultimodalModel` 初始化时，会从总配置中读取以下子配置：

- `image_backbone`
- `pc_backbone`
- `text_encoder`
- `fusion`
- `decoder`
- `object_center_head`
- `placement_center_head`
- `size_yaw_head`

也就是说，配置组织形式大致应为：

```python
model_cfg = {
    "image_backbone": {...},
    "pc_backbone": {...},
    "text_encoder": {...},
    "fusion": {...},
    "decoder": {...},
    "object_center_head": {...},
    "placement_center_head": {...},
    "size_yaw_head": {...},
}
```

其中比较关键的配置包括：

- 点云网格参数：`voxel_size`、`point_cloud_range`
- 点云体素化参数：`max_points_per_voxel`、`max_voxels`
- 图像 backbone 类型：`resnet50` 或 `resnet101`
- 文本 backbone 名称：例如 `roberta-base`
- 融合维度与层数：`hidden_dim`、`num_layers`、`num_heads`
- decoder query 数：`num_queries`

## 12. 当前实现特点与注意事项

结合当前代码，模型构建部分有以下几个特点和边界条件需要注意。

### 12.1 模态输入是可选的，但至少要有一个

三种模态输入都支持传 `None`，但融合模块要求至少有一个有效模态，否则会抛出异常。

### 12.2 点云 token 是体素级，不是点级

当前点云分支先做体素化，再做体素特征编码，因此进入融合模块的是体素 token。这样可以降低序列长度，但也意味着细粒度点级信息已经在体素编码阶段被聚合。

### 12.3 文本显式位置编码当前缺失

融合阶段对文本 token 没有额外加入显式位置投影，文本顺序信息主要依赖预训练语言模型自身表示。这一设计当前是成立的，但若后续需要更强的跨模态对齐能力，可以考虑加入单独的文本位置编码方案。

### 12.4 当前默认是双 query 单框预测

decoder 默认使用两个固定语义 query：object query 表达目标物体，placement query 表达放置位。当前实现仍然是“给定场景和描述，回归一个目标放置框”的任务设定，而不是完整的多目标检测框架。

### 12.5 返回中保留了大量中间字段

这对调试非常方便，但如果后续在推理阶段追求极简接口，可以再封装更轻量的推理输出。

### 12.6 点云体素截断和随机采样会影响稳定性

当单体素内点数超过 `max_points_per_voxel` 时，当前实现会随机采样点；当有效体素数超过 `max_voxels` 时，也会截断体素集合。对训练来说这是可接受的，但需要注意其对复现实验和边界场景的影响。

## 13. 小结

当前 `src/models` 的整体设计思路可以概括为：

- 各模态先各自编码
- 统一整理成 token 序列
- 在统一 Transformer 空间内做融合
- 用 query decoder 提取目标相关表示
- 用检测头回归最终 3D BBox

从工程结构上看，这一实现已经将“模态编码”、“融合建模”和“任务头预测”三部分较清晰地解耦，后续如果要扩展更多模态、替换 backbone 或调整输出头，整体改动成本相对可控。
