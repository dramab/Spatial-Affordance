"""
tools/qwen3vl_single_prompt.py
------------------------------
调用本地 Qwen3-VL，为单张 placement 可视化图片生成一条放置指令。

用法示例:
    conda run -n qwen3vl python tools/qwen3vl_single_prompt.py \
        --image_path outputs/placement_rgb_bbox_vis/hope__scene_0000_0000_obj_3_p000.png

    conda run -n qwen3vl python tools/qwen3vl_single_prompt.py \
        --image_path outputs/placement_rgb_bbox_vis/hope__scene_0000_0000_obj_3_p000.png \
        --object_name bottle \
        --device cuda:0 \
        --max_new_tokens 128
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

DEFAULT_MODEL_PATH = Path(
    "/data/bingkun.yang/yolov13/Qwen3-VL/Qwen/qwen/Qwen3-VL-8B-Instruct"
)

PROMPT_TEMPLATE = (
    "Observe the position of {object_name} marked in orange box in the image "
    "and find the empty space on the table where it can be placed. "
    "Output one natural-language instruction in the format: "
    "'Move the bottle located in the bottom right corner of the green cup "
    "to the left of the can.' "
    "Do not mention the orange and green box in the instruction. "
    "Only output the instruction sentence."
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="使用本地 Qwen3-VL 为单张图片生成放置指令。")
    parser.add_argument("--image_path", type=Path, required=True, help="输入图片路径。")
    parser.add_argument(
        "--model_path", type=Path, default=DEFAULT_MODEL_PATH, help="本地 Qwen3-VL 模型目录。"
    )
    parser.add_argument(
        "--object_name", type=str, default="object", help="目标物体名称，用于构造更具体的提示词。"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=128, help="生成文本的最大 token 数。"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help='推理设备，如 "cuda:0" 或 "cpu"；为空时自动分卡。',
    )
    return parser.parse_args()


def load_model_and_processor(
    model_path: Path, device: str
) -> tuple[Qwen3VLForConditionalGeneration, AutoProcessor]:
    """加载模型和 processor。"""
    kwargs = {"pretrained_model_name_or_path": str(model_path), "dtype": "auto"}
    if device:
        model = Qwen3VLForConditionalGeneration.from_pretrained(**kwargs)
        model = model.to(device)
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(**kwargs, device_map="auto")
    model.eval()
    processor = AutoProcessor.from_pretrained(str(model_path))
    return model, processor


@torch.inference_mode()
def generate_instruction(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    image_path: Path,
    text_prompt: str,
    max_new_tokens: int,
    device: str,
) -> str:
    """调用模型生成放置指令。"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    target_device = device or str(model.device)
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, repetition_penalty=1.1)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    text = output_text[0] if output_text else ""
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.splitlines()[1:-1]).strip()
    return cleaned.strip('"').strip("'")


def main() -> None:
    args = parse_args()

    if not args.image_path.is_file():
        raise FileNotFoundError(f"找不到输入图片: {args.image_path}")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"找不到模型目录: {args.model_path}")

    prompt = PROMPT_TEMPLATE.format(object_name=(args.object_name.strip() or "object"))
    model, processor = load_model_and_processor(args.model_path, args.device)
    instruction = generate_instruction(
        model=model,
        processor=processor,
        image_path=args.image_path,
        text_prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )

    print(f"image_path: {args.image_path}")
    print(f"prompt: {instruction}")


if __name__ == "__main__":
    main()
