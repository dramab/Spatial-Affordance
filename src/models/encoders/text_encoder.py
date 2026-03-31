"""
src/models/encoders/text_encoder.py
-------------------------------------
职责：文本 prompt 编码，输出 token 级别特征。

实现：
- 使用 HuggingFace transformers 加载 BERT（bert-base-uncased）
- 内部完成 tokenize（支持 batch list of str 输入）
- 输出 last hidden state 作为 token 特征

输入：
    texts: List[str]  # batch 内的文字 prompt

输出：
    text_feats: Tensor(B, L, C)  # token 特征（L=max_length, C=out_channels）
    text_mask:  Tensor(B, L)     # padding mask（True=有效 token）

用法：
    encoder = TextEncoder(cfg.text_encoder)
    text_feats, text_mask = encoder(["the brown chair", "the red table"])
"""
