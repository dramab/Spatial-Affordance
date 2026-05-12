"""
tests/test_infer_multimodal.py
------------------------------
职责：测试多模态推理脚本的世界坐标恢复与可视化导出逻辑。

测试内容：
- test_infer_multimodal_exports_predictions_and_visualizations：
  验证脚本会导出 predictions.json、恢复世界坐标框并保存投影图片
- test_infer_multimodal_respects_sample_ids_and_limit：
  验证脚本会按 sample_ids 过滤，并在过滤后应用 limit

用法：
    pytest tests/test_infer_multimodal.py -v
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from src.datasets import PlacementMultimodalDataset, placement_multimodal_collate_fn
from src.models import MultimodalModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "infer_multimodal.py"
SPEC = importlib.util.spec_from_file_location("infer_multimodal", MODULE_PATH)
infer_multimodal = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(infer_multimodal)


class _FakeTokenizer:
    """
    作用：在测试中模拟 HuggingFace tokenizer。

    输入：
        texts: list[str] 原始文本
    输出：
        dict[str, Tensor]，包含 input_ids 与 attention_mask
    """

    def __call__(
            self,
            texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"):
        del padding, truncation, max_length, return_tensors
        batch_size = len(texts)
        seq_len = 6
        input_ids = torch.arange(batch_size * seq_len).view(batch_size, seq_len)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class _FakeModel(torch.nn.Module):
    """
    作用：在测试中模拟 HuggingFace Transformer 主干。

    输入：
        input_ids: Tensor(B, L)
        attention_mask: Tensor(B, L)
    输出：
        带有 last_hidden_state 属性的对象
    """

    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = torch.nn.Embedding(256, hidden_size)

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        del attention_mask, position_ids
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def _write_json(path: Path, payload: dict | list) -> None:
    """
    用法: _write_json(path, payload)
    作用: 为测试写入 JSON 文件
    输入: path: Path；payload: dict | list
    输出: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_ascii_ply(path: Path, points: np.ndarray) -> None:
    """
    用法: _write_ascii_ply(path, points)
    作用: 为测试写入最小可读的 ASCII PLY 点云
    输入: path: Path；points: ndarray(N, 3)
    输出: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        for x, y, z in points:
            f.write(f"{x:.4f} {y:.4f} {z:.4f} 255 0 0\n")


def _write_rgb_image(path: Path, width: int = 32, height: int = 24) -> None:
    """
    用法: _write_rgb_image(path)
    作用: 为测试写入一张纯色 RGB 图
    输入: path: Path；width: int；height: int
    输出: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = 32
    image[..., 1] = 64
    image[..., 2] = 96
    Image.fromarray(image, mode="RGB").save(path)


def _make_model_cfg() -> dict:
    """
    作用：构造轻量化多模态模型测试配置。

    输入：
        无
    输出：
        dict，模型配置
    """
    return {
        "image_backbone": {
            "type": "resnet50",
            "pretrained": False,
            "out_channels": 32,
        },
        "pc_backbone": {
            "type": "voxelnet",
            "voxel_size": [0.25, 0.25, 0.25],
            "point_cloud_range": [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            "max_points_per_voxel": 8,
            "max_voxels": 256,
            "input_feature_dim": 6,
            "svfe_hidden_channels": 16,
            "svfe_out_channels": 32,
            "cml_channels": [32],
        },
        "text_encoder": {
            "type": "roberta-base",
            "out_channels": 32,
            "max_length": 16,
        },
        "fusion": {
            "hidden_dim": 32,
            "num_layers": 2,
            "num_heads": 4,
            "dropout": 0.0,
        },
        "decoder": {
            "hidden_dim": 32,
            "num_layers": 2,
            "num_heads": 4,
            "dropout": 0.0,
            "num_queries": 2,
        },
        "placement_center_head": {
            "hidden_dim": 32,
            "num_layers": 2,
            "out_dim": 3,
        },
        "object_center_head": {
            "hidden_dim": 32,
            "num_layers": 2,
            "out_dim": 3,
        },
        "size_yaw_head": {
            "hidden_dim": 32,
            "num_layers": 2,
            "out_dim": 4,
        },
    }


def _build_annotation_root(tmp_path: Path) -> Path:
    """
    用法: annotation_dir = _build_annotation_root(tmp_path)
    作用: 构造供推理脚本使用的最小标注目录
    输入: tmp_path: Path，pytest 临时目录
    输出: Path，标注目录路径
    """
    annotation_dir = tmp_path / "data/annotations/placement_multimodal"
    rgb_dir = tmp_path / "outputs/placement_rgb_bbox_vis"
    point_dir = tmp_path / "outputs/demo/point_clouds"

    _write_rgb_image(rgb_dir / "demo__sample_a.png", width=32, height=24)
    _write_rgb_image(rgb_dir / "demo__sample_b.png", width=40, height=30)
    _write_ascii_ply(
        point_dir / "sample_a.ply",
        np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
        ], dtype=np.float64),
    )
    _write_ascii_ply(
        point_dir / "sample_b.ply",
        np.array([
            [1.0, 1.0, 0.0],
            [5.0, 1.0, 0.0],
            [1.0, 5.0, 0.0],
            [5.0, 5.0, 0.0],
        ], dtype=np.float64),
    )

    payload = {
        "schema_version": "placement_multimodal_dataset/v2",
        "split": "test",
        "sample_count": 2,
        "samples": [
            {
                "sample_id": "sample_a",
                "source_name": "demo",
                "rgb_path": "outputs/placement_rgb_bbox_vis/demo__sample_a.png",
                "point_cloud_path": "outputs/demo/point_clouds/sample_a.ply",
                "prompt": "raw prompt a",
                "polished_prompt": "polished prompt a",
                "placement": {
                    "target_box": [1.0, 2.0, 0.0, 2.0, 4.0, 6.0, 270.0],
                    "object_center": [2.0, 4.0, 0.0],
                },
                "camera": {
                    "fx": 100.0,
                    "fy": 100.0,
                    "cx": 16.0,
                    "cy": 12.0,
                    "img_w": 32,
                    "img_h": 24,
                    "E_c2w": np.eye(4, dtype=np.float64).tolist(),
                },
            },
            {
                "sample_id": "sample_b",
                "source_name": "demo",
                "rgb_path": "outputs/placement_rgb_bbox_vis/demo__sample_b.png",
                "point_cloud_path": "outputs/demo/point_clouds/sample_b.ply",
                "prompt": "raw prompt b",
                "polished_prompt": "polished prompt b",
                "placement": {
                    "target_box": [3.0, 3.0, 0.0, 2.0, 2.0, 4.0, 90.0],
                    "object_center": [1.0, 1.0, 0.0],
                },
                "camera": {
                    "fx": 120.0,
                    "fy": 121.0,
                    "cx": 20.0,
                    "cy": 15.0,
                    "img_w": 40,
                    "img_h": 30,
                    "E_c2w": np.eye(4, dtype=np.float64).tolist(),
                },
            },
        ],
    }
    _write_json(annotation_dir / "test.json", payload)
    return annotation_dir


def _build_checkpoint(tmp_path: Path, annotation_dir: Path, monkeypatch) -> Path:
    """
    用法: checkpoint_path = _build_checkpoint(tmp_path, annotation_dir, monkeypatch)
    作用: 构造供推理脚本读取的最小 checkpoint
    输入: tmp_path: Path；annotation_dir: Path；monkeypatch: pytest monkeypatch
    输出: Path，checkpoint 路径
    """
    monkeypatch.setattr(
        "src.models.encoders.text_encoder.AutoTokenizer.from_pretrained",
        lambda _: _FakeTokenizer(),
    )
    monkeypatch.setattr(
        "src.models.encoders.text_encoder.AutoModel.from_pretrained",
        lambda _: _FakeModel(),
    )

    torch.manual_seed(0)
    model = MultimodalModel(_make_model_cfg())
    checkpoint = {
        "epoch": 1,
        "best_metric": 0.1,
        "best_metric_name": "val_loss",
        "model_state_dict": model.state_dict(),
        "train_config": {
            "dataset": {
                "annotation_dir": annotation_dir.as_posix(),
                "prompt_key": "polished_prompt",
                "image_size": [64, 64],
                "scale_eps": 1.0e-6,
            },
            "dataloader": {
                "val_batch_size": 2,
                "num_workers": 0,
                "persistent_workers": False,
            },
        },
        "model_config": _make_model_cfg(),
    }
    checkpoint_path = tmp_path / "outputs/multimodal_train/best.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def _build_expected_prediction(
        annotation_dir: Path,
        checkpoint_path: Path,
        sample_index: int,
        monkeypatch,
        tmp_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    用法: pred_box_world, pred_center_world = _build_expected_prediction(annotation_dir, checkpoint_path, 0, monkeypatch, tmp_path)
    作用: 独立运行一遍模型前向，生成用于断言的期望世界坐标预测框与物体中心
    输入: annotation_dir: Path；checkpoint_path: Path；sample_index: int；monkeypatch；tmp_path: Path
    输出: tuple，分别为 ndarray(7,) 预测框与 ndarray(3,) 预测中心
    """
    monkeypatch.setattr("src.datasets.multimodal_dataset.PROJECT_ROOT", tmp_path)
    dataset = PlacementMultimodalDataset(
        annotation_dir=annotation_dir,
        split="test",
        prompt_key="polished_prompt",
        image_size=(64, 64),
        scale_eps=1.0e-6,
    )
    batch = placement_multimodal_collate_fn([dataset[sample_index]])

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = MultimodalModel(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.inference_mode():
        outputs = model(
            points_xyz=batch["points_xyz"],
            images=batch["images"],
            text_inputs=batch["text_inputs"],
        )
    pred_box_norm = infer_multimodal.flatten_pred_boxes(outputs["pred_boxes_norm"])[0].detach().cpu().numpy()
    pred_center_norm = infer_multimodal.flatten_pred_centers(
        outputs["pred_object_centers_norm"]
    )[0].detach().cpu().numpy()
    scene_center = batch["norm_meta"]["scene_center"][0].detach().cpu().numpy()
    scene_scale = float(batch["norm_meta"]["scene_scale"][0].item())
    yaw_scale = float(batch["norm_meta"]["yaw_scale"][0].item())
    pred_box_world = infer_multimodal.denormalize_world_box(
        pred_box_norm,
        scene_center,
        scene_scale,
        yaw_scale,
    )
    pred_center_world = infer_multimodal.denormalize_world_center(
        pred_center_norm,
        scene_center,
        scene_scale,
    )
    return pred_box_world, pred_center_world


def test_denormalize_world_box_exp_recovers_log_size_norm():
    """
    作用：验证推理反归一化会用 exp(log_size_norm) 恢复世界尺寸。

    输入：
        无，内部构造 normalized box 与归一化元信息
    输出：
        无，通过断言验证结果
    """
    box_norm = np.array(
        [0.25, -0.5, 0.0, np.log(0.5), np.log(1.0), np.log(1.5), -0.5],
        dtype=np.float64,
    )
    scene_center = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    box_world = infer_multimodal.denormalize_world_box(
        box_norm=box_norm,
        scene_center=scene_center,
        scene_scale=4.0,
        yaw_scale=180.0,
    )

    expected_box_world = np.array([2.0, 0.0, 3.0, 2.0, 4.0, 6.0, -90.0], dtype=np.float64)
    assert np.allclose(box_world, expected_box_world, atol=1e-6)


def test_save_prediction_visualization_draws_object_center_dot(tmp_path):
    """
    作用：验证推理可视化会用高对比圆点绘制预测移动前物体中心。

    输入：
        tmp_path: pytest 临时目录
    输出：
        无，通过中心像素与外圈颜色断言验证结果
    """
    rgb_path = tmp_path / "input.png"
    output_path = tmp_path / "vis.png"
    _write_rgb_image(rgb_path, width=32, height=24)
    camera = {
        "fx": 10.0,
        "fy": 10.0,
        "cx": 16.0,
        "cy": 12.0,
        "img_w": 32,
        "img_h": 24,
        "E_c2w": np.eye(4, dtype=np.float64).tolist(),
    }

    infer_multimodal.save_prediction_visualization(
        rgb_path=rgb_path,
        pred_box_world=np.array([0.0, 0.0, 10.0, 2.0, 2.0, 2.0, 0.0], dtype=np.float64),
        pred_object_center_world=np.array([0.0, 0.0, 10.0], dtype=np.float64),
        camera_dict=camera,
        output_path=output_path,
    )

    image = np.asarray(Image.open(output_path).convert("RGB"))
    assert tuple(image[12, 16].tolist()) == infer_multimodal.COLOR_OBJECT_CENTER
    assert tuple(image[12, 25].tolist()) == infer_multimodal.COLOR_OBJECT_CENTER_OUTLINE


def test_infer_multimodal_exports_predictions_and_visualizations(tmp_path, monkeypatch):
    """
    作用：验证推理脚本会导出 predictions.json、恢复世界坐标框并保存投影图。

    输入：
        tmp_path: pytest 临时目录
        monkeypatch: pytest monkeypatch 工具
    输出：
        无，通过断言验证结果
    """
    annotation_dir = _build_annotation_root(tmp_path)
    checkpoint_path = _build_checkpoint(tmp_path, annotation_dir, monkeypatch)

    monkeypatch.setattr("src.datasets.multimodal_dataset.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(infer_multimodal, "PROJECT_ROOT", tmp_path)

    args = SimpleNamespace(
        checkpoint=checkpoint_path,
        split="test",
        annotation_dir=annotation_dir,
        device="cpu",
        output_dir=Path("outputs/multimodal_infer"),
        sample_ids=None,
        limit=None,
        batch_size=2,
        num_workers=0,
    )

    payload = infer_multimodal.run_inference(args)
    predictions_path = tmp_path / "outputs/multimodal_infer/predictions.json"
    payload_from_disk = json.loads(predictions_path.read_text(encoding="utf-8"))
    expected_pred_box_world, expected_pred_object_center_world = _build_expected_prediction(
        annotation_dir=annotation_dir,
        checkpoint_path=checkpoint_path,
        sample_index=0,
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
    )

    assert payload["schema_version"] == infer_multimodal.SCHEMA_VERSION
    assert payload["sample_count"] == 2
    assert payload_from_disk["sample_count"] == 2

    first_prediction = payload["predictions"][0]
    vis_path = tmp_path / first_prediction["vis_path"]
    assert first_prediction["sample_id"] == "sample_a"
    assert first_prediction["source_name"] == "demo"
    assert first_prediction["gt_box_world"] == [1.0, 2.0, 0.0, 2.0, 4.0, 6.0, 270.0]
    assert first_prediction["gt_object_center_world"] == [2.0, 4.0, 0.0]
    assert np.allclose(first_prediction["pred_box_world"], expected_pred_box_world, atol=1e-6)
    assert np.allclose(
        first_prediction["pred_object_center_world"],
        expected_pred_object_center_world,
        atol=1e-6,
    )
    assert len(first_prediction["pred_object_center_norm"]) == 3
    assert predictions_path.exists()
    assert vis_path.exists()
    assert Image.open(vis_path).size == (32, 24)


def test_infer_multimodal_respects_sample_ids_and_limit(tmp_path, monkeypatch):
    """
    作用：验证推理脚本会按 sample_ids 过滤，并在过滤后应用 limit。

    输入：
        tmp_path: pytest 临时目录
        monkeypatch: pytest monkeypatch 工具
    输出：
        无，通过断言验证结果
    """
    annotation_dir = _build_annotation_root(tmp_path)
    checkpoint_path = _build_checkpoint(tmp_path, annotation_dir, monkeypatch)

    monkeypatch.setattr("src.datasets.multimodal_dataset.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(infer_multimodal, "PROJECT_ROOT", tmp_path)

    args = SimpleNamespace(
        checkpoint=checkpoint_path,
        split="test",
        annotation_dir=annotation_dir,
        device="cpu",
        output_dir=Path("outputs/multimodal_infer_filtered"),
        sample_ids=["sample_b", "sample_a"],
        limit=1,
        batch_size=2,
        num_workers=0,
    )

    payload = infer_multimodal.run_inference(args)

    assert payload["sample_count"] == 1
    assert [item["sample_id"] for item in payload["predictions"]] == ["sample_b"]
    assert payload["predictions"][0]["vis_path"] == "outputs/multimodal_infer_filtered/vis/demo__sample_b.png"
