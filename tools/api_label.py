"""
======================================================================
异步多模态标签润色脚本 (颜色+形状专属版)
======================================================================
特点:
1. 异步并发：基于 AsyncOpenAI 库，利用 asyncio 实现高并发请求，速度提升十倍。
2. 精准润色：仅对物体添加「颜色+形状」描述，同时支持相对位置关系精准增强，零改动原句核心内容。
3. 容错回退：API 调用失败或图片丢失时，自动回退到原始标签，保证数据不丢失。
4. 增量保存：每处理50条自动保存，中断时触发紧急保存，防止数据丢失。

用法示例:
   conda run -n spatial python tools/api_label.py
   conda run -n spatial python tools/api_label.py --limit 100
   conda run -n spatial python tools/api_label.py --limit 100 --random-sample --seed 2026
   conda run -n spatial python tools/api_label.py \
       --json-path outputs/auto_labels_simple_merged/all_labels.json \
       --image-dirs outputs/placement_rgb_bbox_vis outputs/placement_rgb_bbox_vis_ycbv outputs/placement_rgb_bbox_vis_dopose \
       --output-json-path outputs/auto_labels_simple_merged/all_labels_polished.json
======================================================================
"""

import argparse
import asyncio
import base64
import json
import random
import threading  # 用于线程安全的计数器和保存锁
from pathlib import Path
from typing import List, Optional

from openai import AsyncOpenAI

# ================= 配置区域 =================
# 阿里云百炼 API Key
API_KEY = 'sk-c0897cbd1d1f4b0d91447b9b2b673cb6'  # 替换为你的真实 Key

# 文件路径配置
JSON_PATH = Path("outputs/auto_labels_simple_2/all_labels.json")           # 原始自动标注生成的 JSON
IMAGE_DIRS = [Path("outputs/placement_rgb_bbox_vis_ycbv")]             # 图片所在目录列表
OUTPUT_JSON_PATH = Path("outputs/auto_labels_simple_2/all_labels_polished.json") # 润色后输出的新 JSON

# 控制最大并发数 (推荐 5~10，防止触发阿里云 API 的 QPS 限制)
MAX_CONCURRENT_REQUESTS = 10

# 选择模型: 视觉增强任务推荐使用 qwen-vl-max 或 qwen-vl-plus
MODEL_NAME = 'qwen3.6-plus'

# 增量保存阈值 (每处理N条保存一次)
SAVE_THRESHOLD = 50
# ============================================

# 初始化异步客户端 (使用阿里云兼容 OpenAI 的 endpoint)
client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 全局计数器和保存锁 (线程安全)
process_counter = 0
counter_lock = threading.Lock()
processed_lock = threading.Lock()
save_lock = threading.Lock()

def encode_image(image_path: Path) -> str:
    """
    用法: base64_text = encode_image(Path("image.png"))
    作用: 将本地图片编码为可放入多模态请求的 Base64 字符串。
    输入: image_path 为图片文件路径。
    输出: 图片内容的 Base64 字符串。
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_polish_prompt(original_label: str) -> str:
    """
    用法: prompt = get_polish_prompt("Move the cup to the plate.")
    作用: 根据原始标签生成视觉属性润色提示词。
    输入: original_label 为原始英文操作指令。
    输出: 用于多模态大模型请求的 prompt 字符串。
    """
    prompt1 = f"""
You are a professional, strictly rule-abiding robotic operation annotation refinement specialist.
Your ONLY task is to enrich the provided original English operation instruction, with ZERO changes to the core content of the original sentence.

Original Operation Instruction: "{original_label}"

============= ABSOLUTE NON-NEGOTIABLE RULES =============
1.  ZERO CORE CONTENT ALTERATIONS: You MUST NOT change, delete, reorder, or rewrite any single core word from the original sentence, including all object nouns, action verbs, core prepositions of position (behind/front/left/right/top/bottom/in front of etc.), punctuation, capitalization, and sentence structure.
2.  PERMITTED ENRICHMENT 1 (OBJECT VISUAL MODIFICATION - ONLY 2 TYPES ALLOWED): You may add 1-2 descriptive adjectives **immediately before** concrete object nouns in the sentence. Adjectives are STRICTLY LIMITED to ONLY two categories:
    - Color attributes (e.g., red, blue, yellow, orange, white, black, grey, green, silver, red-and-white etc.)
    - Shape/geometry attributes (e.g., round, cylindrical, rectangular, square, flat, long, short, tall, slender, curved etc.)
    ❌ FORBIDDEN: Any adjectives describing material, packaging, container type, brand, texture, or other non-color/non-shape attributes are strictly prohibited.
3.  PERMITTED ENRICHMENT 2 (POSITION PRECISION ENHANCEMENT): You may add precise spatial adverbs **immediately before** the prepositional phrases of relative position in the original sentence, to make the position description more accurate and robot-readable. Allowed adverbs include: directly, immediately, slightly, far, closely, exactly, just. YOU MUST NOT MODIFY THE ORIGINAL POSITION PREPOSITION ITSELF, ONLY ADD ADVERBS TO ENHANCE PRECISION.
4.  FULL COVERAGE REQUIREMENT: You must add appropriate color/shape adjectives to every concrete object noun in the sentence, regardless of whether it is the action target or a position reference object.
5.  SAME NOUN TREATMENT: If the same object noun appears multiple times, you may use consistent color/shape adjectives for each occurrence, no forced differentiation is required.
6.  OUTPUT REQUIREMENT: You MUST output ONLY the final enriched complete English sentence. No explanations, notes, brackets, or extra content of any kind.

============= CORRECT EXAMPLE =============
Original Input: "Move the apple located behind the plate to the right of the fork."
Correct Output: "Move the red round apple located directly behind the white flat plate to slightly to the right of the silver slender fork."

============= FORBIDDEN EXAMPLES =============
Forbidden 1 (changed core word): "Move the fruit located behind the plate to the right of the fork." (changed "apple" to "fruit")
Forbidden 2 (added forbidden material description): "Move the red apple located behind the ceramic plate to the right of the metal fork." (added material words "ceramic" "metal")
Forbidden 3 (modified position logic): "Move the apple located in front of the plate to the left of the fork." (changed original position preposition)
Forbidden 4 (extra content): "Enriched instruction: Move the red apple..." (added extra explanation)

Now, output your refined instruction strictly following the rules above:
"""
    prompt2 = f"""
    You are a professional, strictly rule-abiding robotic operation annotation refinement specialist.
Your task is to enrich the provided original English operation instruction based STRICTLY on the visual evidence in the accompanying image.

Image Context Information:
* The specific object that needs to be moved is highlighted with an orange bounding box.
* The final empty placement destination is highlighted with a green bounding box.
* The reference object for the starting position is the physical object closest to the orange box. The destination reference object is closest to the green box.

Original Operation Instruction: "{original_label}"
Reference Image: [User Uploads Image]

============= ABSOLUTE NON-NEGOTIABLE RULES =============
1. STRICT IMAGE GROUNDING & ATTRIBUTE ISOLATION: Every single enrichment MUST accurately reflect the visual evidence.
    * CRITICAL: Do NOT confuse the attributes of the target object with the reference objects. (e.g., Do not apply the Tuna Can's "cylindrical" shape to the Clamp).
    * NEVER mention the boxes (orange/green/bounding box) in your output.
2. ZERO DELETION OF ORIGINAL WORDS (CRITICAL): You MUST NOT delete, replace, or alter ANY nouns or adjectives present in the original instruction. Core words are strictly "add-only".
3. VERB USAGE (FLEXIBLE & DIVERSE): You MAY replace the default action verb (e.g., "Move") with synonyms to increase diversity. Keeping the original verb (e.g., "Move") and maintaining the exact original sentence structure is also acceptable.
    * CRITICAL INSTRUCTION: Do NOT blindly copy the exact verbs used in the "CORRECT EXAMPLES" below. You must independently select appropriate and diverse manipulation verbs from your own vocabulary.
4. POSITIONAL WORD VARIATION: If the original instruction uses "located" (e.g., "located behind"), you MAY keep it, change it to synonyms, or remove it entirely (e.g., just "behind"). Do NOT change the core spatial preposition (behind, left, right, etc.).
5. STRICT PRE-MODIFIER GRAMMAR ONLY: You MUST ONLY use the `[Adjective(s)] + [Noun]` structure for visual attributes. NO relative clauses ("which is"), NO appositives, and NO prepositional phrases for attributes.
6. PERMITTED VISUAL ENRICHMENT: Add descriptive attributes derived entirely from the image immediately before EVERY concrete object noun:
    * Color (e.g., red, blue, green, black)
    * Shape/geometry (e.g., round, cylindrical, square, flat)
    * Size/scale (e.g., small, tall, short, large)
    ❌ FORBIDDEN: Material, packaging, brand, or texture.
7. FLUENCY & ADJECTIVE ORDER (CRITICAL): You MUST adhere strictly to the standard English adjective order: [Size] -> [Shape] -> [Color] -> [Noun] (e.g., "small round red apple"). Do not deviate from this order.

============= CORRECT EXAMPLES =============
Example 1 (Keep original verb and structure - ONLY add adjectives):
Original Input: "Move the red apple located behind the plate to the fork."
Correct Output: "Move the small round red apple located directly behind the white flat plate to the slender silver fork."
(Note: The original verb "Move" and the word "located" are kept exactly the same. Only adjectives are strictly added following the Size->Shape->Color order.)

Example 2 (Verb swap + Participle variation):
Original Input: "Move the cup near the block to the cup."
Correct Output: "Take the small cylindrical blue cup situated closely near the small square red block to the large cylindrical yellow cup."
(Note: Verb changed to "Take", "located" changed to "situated", fluent and strict Adj+Noun.)

Example 3 (Verb swap + Omitting positional word for natural flow):
Original Input: "Move TunaCan located at the front left of Clamp to the front right of Clamp."
Correct Output: "Set the small cylindrical blue TunaCan at the front left of the tall black Clamp to the front right of the tall black Clamp."
(Note: Verb changed to "Set". The word "located" is omitted entirely for a more natural phrasing.)

============= FORBIDDEN EXAMPLES =============
Forbidden 1 (Attribute Misplacement):
Output: "...to the right of the cylindrical blue clamp." (VIOLATION: Clamps are not cylindrical or blue; the model hallucinated the Tuna Can's attributes onto the Clamp.)
Forbidden 2 (Wrong Adjective Order):
Output: "Move the red round small apple..." (VIOLATION: Must strictly be Size->Shape->Color: "small round red apple".)
Forbidden 3 (Deleted original adjective):
Original: "Move the red apple..." -> Output: "Place the small round apple..." (VIOLATION: Deleted "red".)
Forbidden 4 (Lazy verb copying):
Output: Always relying on some verbs just because they appeared in the examples. (VIOLATION: You must independently generate appropriate and diverse verbs).

Now, enrich the instruction following the rules above:
"""
    prompt3 = f"""
You are a professional robotic operation annotation refinement specialist.
Your ONLY task is to enrich and naturally rephrase the provided original English operation instruction based STRICTLY on the visual evidence in the accompanying image. Your goal is to break away from rigid, templated language and provide a contextually accurate, natural-sounding instruction that maintains a simple imperative grammatical structure. The final sentence must be clear, concise, and not overly long.

Image Context Information:

The specific object that needs to be moved is highlighted with an orange bounding box.

The final empty placement destination (the spatial void where the object will go) is highlighted with a green bounding box.

The original instruction may mention spatial reference objects for both the starting position and the final placement position. To identify them visually: the reference object for the starting position is the physical object closest to the orange box; the reference object for the placement position is the physical object closest to the green box.
CRITICAL: These boxes are strictly for YOUR reference to locate the target, the destination, and their respective closest reference objects. You MUST NOT mention the boxes, the colors of the boxes, or the word "box" (e.g., "orange-boxed", "green box") in your final enriched sentence.

Original Operation Instruction: "{original_label}"
Reference Image: [User Uploads Image]

============= ABSOLUTE NON-NEGOTIABLE RULES =============

STRICT IMAGE GROUNDING & VISUAL CONTEXT: Every single enrichment MUST accurately reflect the actual visual evidence. Do not hallucinate. Use the orange box to identify the moving target, and use the green box to locate the empty destination. Identify the reference objects by finding the physical objects structurally closest to these respective boxes. Describe all objects using ONLY their intrinsic visual properties.

NATURAL REPHRASING & CORE MEANING PRESERVATION: You are encouraged to adapt the verb and prepositions to better fit the physical action and the scene shown in the image. For example, you can change a generic "Move" to "Put", "Place", "Slide", "Insert", or "Stack". You may also adjust awkward prepositions (e.g., smoothing "located to"). However, you MUST maintain a simple imperative sentence structure and preserve the core intent/target nouns of the original instruction. Do not make the sentence unnecessarily long.

PERMITTED VISUAL ENRICHMENT (ONLY 4 CATEGORIES ALLOWED): You may add brief descriptive adjectives derived entirely from the image. CRITICAL: If the original_label already contains descriptive adjectives, you MUST retain them. Allowed new additions are STRICTLY LIMITED to four categories:

Color attributes (e.g., red, blue, yellow, white, black, grey, green, silver etc.)

Shape/geometry attributes (e.g., round, cylindrical, rectangular, square, flat, curved etc.)

Size/scale attributes (e.g., small, large, smaller, bigger, tall, short, long, slender, thick etc.)

Posture/Orientation attributes (e.g., standing, laying flat, upright, upside down, tilted etc.)
❌ FORBIDDEN: Adjectives describing material, packaging, bounding boxes, annotations, highlights, brand, or texture are strictly prohibited.

MULTIPLE INSTANCE DISAMBIGUATION (CRITICAL): If the image contains multiple objects of the same category (e.g., several cups, multiple clamps), use the closest physical objects to the respective boxes to identify the correct reference objects. You MUST use ONLY their intrinsic distinguishing attributes (color, shape, size, posture) to uniquely identify them.

POSITION PRECISION ENHANCEMENT: You may add precise spatial adverbs to relative positions based on real spatial distances shown in the image (e.g., directly, immediately, slightly, far, closely, exactly, just) to make the operation clearer.

OUTPUT REQUIREMENT: You MUST output ONLY the final enriched complete English sentence. No explanations, notes, brackets, or extra content of any kind. Keep it concise, natural, and simple.

============= CORRECT EXAMPLES =============
Example 1 (Verb Adaptation and Posture):
Image Context: A red, round apple (orange box) is behind a white plate laying flat (closest to the orange box). A green box marks an empty spot slightly to the right of a standing, silver fork.
Original Input: "Move the apple located behind the plate to the right of the fork."
Correct Output: "Place the red round apple located directly behind the flat white plate to slightly to the right of the standing silver fork."

Example 2 (Retaining Original Modifiers & Disambiguation):
Image Context: Multiple cups. A blue, cylindrical cup is in the orange box, closest to a red block. The green box (destination) is closest to a yellow cup. The original input already calls it a "blue cup".
Original Input: "Move the blue cup near the block to the cup."
Correct Output: "Put the blue cylindrical cup closely near the red square block next to the yellow cup."

Example 3 (Action Specificity & Posture with Same-Category References):
Image Context: A blue TunaCan (orange box) is next to a standing, smaller black Clamp. The destination (green box) is on the right of a laying flat, bigger black Clamp.
Original Input: "Move TunaCan located at the left of Clamp to the right of Clamp."
Correct Output: "Slide the blue cylindrical TunaCan located closely at the left of the upright smaller black Clamp to just the right of the flat bigger black Clamp."

============= FORBIDDEN EXAMPLES =============
Forbidden 1 (changed core target noun): "Place the fruit located behind the plate..." (VIOLATION: changed the target object "apple" to "fruit". Verbs can change, but target nouns cannot).
Forbidden 2 (mentioned bounding boxes): "Put the blue cup into the green box..." (VIOLATION: mentioned the reference boxes).
Forbidden 3 (added forbidden material description): "Slide the red apple behind the ceramic plate..." (VIOLATION: added material word "ceramic").
Forbidden 4 (overly verbose/complex): "Please take the blue cylindrical TunaCan that is currently resting near the smaller clamp and carefully transport it over to the area on the right side of the larger black clamp." (VIOLATION: loss of simple imperative structure, too wordy).
Forbidden 5 (deleted original descriptors): Original: "Move the red cup...". Output: "Put the cylindrical cup..." (VIOLATION: removed the existing word "red").

Now, output your refined instruction strictly following the rules above:
"""
    prompt4 = f"""
    You are a professional robotic operation annotation refinement specialist. Your ONLY task is to enrich and naturally rephrase the provided original English operation instruction based STRICTLY on the visual evidence in the accompanying image. Your goal is to break away from rigid, templated language and provide a contextually accurate, natural-sounding instruction that maintains a simple imperative grammatical structure. The final sentence must be clear, concise, and not overly long.
Image Context Information:
* The specific object that needs to be moved is highlighted with an orange bounding box.
* The final empty placement destination (the spatial void where the object will go) is highlighted with a green bounding box.
* The original instruction may mention spatial reference objects for both the starting position and the final placement position. To identify them visually: the reference object for the starting position is the physical object closest to the orange box; the reference object for the placement position is the physical object closest to the green box. CRITICAL: These boxes are strictly for YOUR reference to locate the target, the destination, and their respective closest reference objects. You MUST NOT mention the boxes, the colors of the boxes, or the word "box" (e.g., "orange-boxed", "green box") in your final enriched sentence.
Original Operation Instruction: "{original_label}" Reference Image: [User Uploads Image]
============= ABSOLUTE NON-NEGOTIABLE RULES =============
1. STRICT IMAGE GROUNDING & VISUAL CONTEXT: Every single enrichment MUST accurately reflect the actual visual evidence. Do not hallucinate. Use the orange box to identify the moving target, and use the green box to locate the empty destination. Identify the reference objects by finding the physical objects structurally closest to these respective boxes. Describe all objects using ONLY their intrinsic visual properties.
2. NATURAL REPHRASING & CORE MEANING PRESERVATION: You are encouraged to adapt the verb, prepositions, and spatial status words to better fit the physical action and the scene shown in the image. For example:
    * Change a generic "Move" to "Put", "Place", "Slide", "Insert", or "Stack" and so on.
    * Change rigid spatial words like "located" to more natural scene-specific words like "sitting", "resting", "positioned", "placed", or remove "located" entirely if the sentence flows better without it.
    * Smooth out awkward prepositions. However, you MUST maintain a simple imperative sentence structure and preserve the core intent/target nouns of the original instruction. Do not make the sentence unnecessarily long.
3. PERMITTED VISUAL ENRICHMENT (ONLY 4 CATEGORIES ALLOWED): You may add brief descriptive adjectives derived entirely from the image. CRITICAL: If the original_label already contains descriptive adjectives, you MUST retain them. Allowed new additions are STRICTLY LIMITED to four categories:
    * Color attributes (e.g., red, blue, yellow, white, black, grey, green, silver etc.)
    * Shape/geometry attributes (e.g., round, cylindrical, rectangular, square, flat, curved etc.)
    * Size/scale attributes (e.g., small, large, smaller, bigger, tall, short, long, slender, thick etc.)
    * Posture/Orientation attributes (e.g., standing, laying flat, upright, upside down, tilted etc.) ❌ FORBIDDEN: Adjectives describing material, packaging, bounding boxes, annotations, highlights, brand, or texture are strictly prohibited.
4. MULTIPLE INSTANCE DISAMBIGUATION (CRITICAL): If the image contains multiple objects of the same category (e.g., several cups, multiple clamps), use the closest physical objects to the respective boxes to identify the correct reference objects. You MUST use ONLY their intrinsic distinguishing attributes (color, shape, size, posture) to uniquely identify them.
5. POSITION PRECISION ENHANCEMENT: You may add precise spatial adverbs to relative positions based on real spatial distances shown in the image (e.g., directly, immediately, slightly, far, closely, exactly, just) to make the operation clearer.
6. OUTPUT REQUIREMENT: You MUST output ONLY the final enriched complete English sentence. No explanations, notes, brackets, or extra content of any kind. Keep it concise, natural, and simple.
============= CORRECT EXAMPLES ============= Example 1 (Verb Adaptation, Posture, and Replacing "located"): Image Context: A red, round apple (orange box) is behind a white plate laying flat (closest to the orange box). A green box marks an empty spot slightly to the right of a standing, silver fork. Original Input: "Move the apple located behind the plate to the right of the fork." Correct Output: "Place the red round apple resting directly behind the flat white plate slightly to the right of the standing silver fork." (Changed "Move" to "Place", changed "located" to "resting", smoothed prepositions).
Example 2 (Retaining Original Modifiers & Disambiguation): Image Context: Multiple cups. A blue, cylindrical cup is in the orange box, closest to a red block. The green box (destination) is closest to a yellow cup. The original input already calls it a "blue cup". Original Input: "Move the blue cup near the block to the cup." Correct Output: "Put the blue cylindrical cup sitting closely near the red square block next to the yellow cup." (Added "sitting" for natural flow).
Example 3 (Action Specificity & Posture with Same-Category References): Image Context: A blue TunaCan (orange box) is next to a standing, smaller black Clamp. The destination (green box) is on the right of a laying flat, bigger black Clamp. Original Input: "Move TunaCan located at the left of Clamp to the right of Clamp." Correct Output: "Slide the blue cylindrical TunaCan positioned closely at the left of the upright smaller black Clamp to just the right of the flat bigger black Clamp." (Changed "located" to "positioned").
============= FORBIDDEN EXAMPLES ============= Forbidden 1 (changed core target noun): "Place the fruit located behind the plate..." (VIOLATION: changed the target object "apple" to "fruit". Verbs can change, but target nouns cannot). Forbidden 2 (mentioned bounding boxes): "Put the blue cup into the green box..." (VIOLATION: mentioned the reference boxes). Forbidden 3 (added forbidden material description): "Slide the red apple behind the ceramic plate..." (VIOLATION: added material word "ceramic"). Forbidden 4 (overly verbose/complex): "Please take the blue cylindrical TunaCan that is currently resting near the smaller clamp and carefully transport it over to the area on the right side of the larger black clamp." (VIOLATION: loss of simple imperative structure, too wordy). Forbidden 5 (deleted original descriptors): Original: "Move the red cup...". Output: "Put the cylindrical cup..." (VIOLATION: removed the existing word "red").
Now, output your refined instruction strictly following the rules above:
"""
    return prompt4


def build_parser() -> argparse.ArgumentParser:
    """
    用法: parser = build_parser()
    作用: 构建脚本命令行参数解析器。
    输入: 无。
    输出: 配置完成的 argparse.ArgumentParser。
    """
    parser = argparse.ArgumentParser(description="异步调用 VLM 润色自动生成的操作标签")
    parser.add_argument("--json-path", type=Path, default=JSON_PATH, help="输入标签 JSON 路径")
    parser.add_argument(
        "--image-dirs",
        type=Path,
        nargs="+",
        default=IMAGE_DIRS,
        help="图片搜索目录列表；按顺序查找 image_filename",
    )
    parser.add_argument("--output-json-path", type=Path, default=OUTPUT_JSON_PATH, help="润色结果输出 JSON 路径")
    parser.add_argument("--limit", type=int, default=None, help="最多处理 N 条标签；默认处理全部")
    parser.add_argument("--random-sample", action="store_true", help="从输入 JSON 中随机抽取待处理标签")
    parser.add_argument("--seed", type=int, default=42, help="随机采样种子，仅在 --random-sample 开启时生效")
    return parser


def resolve_image_path(image_filename: str, image_dirs: List[Path]) -> Optional[Path]:
    """
    用法: image_path = resolve_image_path("sample.png", [Path("outputs/a"), Path("outputs/b")])
    作用: 在多个图片目录中按顺序查找标签记录对应的图片文件。
    输入: image_filename 为 JSON 中的图片路径或文件名；image_dirs 为候选图片根目录列表。
    输出: 找到时返回图片 Path，未找到时返回 None。
    """
    image_path = Path(image_filename)
    if image_path.is_absolute():
        return image_path if image_path.exists() else None

    if image_path.exists():
        return image_path

    for image_dir in image_dirs:
        candidate = image_dir / image_path
        if candidate.exists():
            return candidate

    return None


def select_items(data: List[dict],
                 limit: Optional[int],
                 random_sample: bool,
                 seed: int) -> List[dict]:
    """
    用法: selected = select_items(data, limit=100, random_sample=True, seed=42)
    作用: 根据数量上限和随机采样开关选择本次需要处理的标签。
    输入: data 为完整标签列表；limit 为处理上限；random_sample 控制是否随机选择；seed 控制随机可复现。
    输出: 本次待处理标签的浅拷贝列表。
    """
    if limit is not None and limit < 1:
        raise ValueError("--limit 必须大于等于 1")

    item_indices = list(range(len(data)))
    if random_sample:
        rng = random.Random(seed)
        if limit is None:
            selected_indices = item_indices[:]
            rng.shuffle(selected_indices)
        else:
            sample_count = min(limit, len(item_indices))
            selected_indices = rng.sample(item_indices, sample_count)
    else:
        selected_indices = item_indices[:limit]

    return [dict(data[index]) for index in selected_indices]


def save_current_data(data: list, output_json_path: Path):
    """
    用法: save_current_data(processed_items, Path("outputs/all_labels_polished.json"))
    作用: 将已处理完成的 item 保存到输出 JSON 文件，跳过尚未完成的空槽位。
    输入: data 为已处理 item 列表，或包含 None 空槽位的处理结果列表；output_json_path 为输出 JSON 路径。
    输出: 无返回值，写入 output_json_path。
    """
    with save_lock:
        try:
            with processed_lock:
                completed_items = [item for item in data if item is not None]
            output_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(completed_items, f, indent=2, ensure_ascii=False)
            print(f"\n💾 保存成功：已写入 {len(completed_items)} 条，文件路径: {output_json_path}\n")
        except Exception as e:
            print(f"\n❌ 增量保存失败: {str(e)}\n")


async def process_single_label(item_index: int,
                               item: dict,
                               semaphore: asyncio.Semaphore,
                               processed_items: list,
                               image_dirs: List[Path],
                               output_json_path: Path) -> bool:
    """
    用法: success = await process_single_label(index, item, semaphore, processed_items, image_dirs, output_json_path)
    作用: 异步处理单条标签，成功时写入 polished_label，失败时回退为原始 label。
    输入: item_index 为输出槽位索引；item 为标签记录；semaphore 控制并发；processed_items 保存已完成结果；
         image_dirs 为图片搜索目录列表；output_json_path 为增量保存路径。
    输出: bool，True 表示 API 润色成功，False 表示异常或回退。
    """
    global process_counter
    async with semaphore:
        image_filename = item.get('image_filename')
        original_label = item.get('label')
        is_success = False

        # 异常数据过滤
        if not image_filename or not original_label:
            item['polished_label'] = original_label
        else:
            image_path = resolve_image_path(image_filename, image_dirs)

            # 1. 图片不存在，直接回退为原始标签
            if image_path is None:
                print(f"[-] 图片丢失，跳过润色: {image_filename}")
                item['polished_label'] = original_label
            else:
                # 2. 生成优化后的Prompt和Base64图片
                prompt = get_polish_prompt(original_label)

                try:
                    base64_image = encode_image(image_path)

                    # 构造多模态消息体
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                            ]
                        }
                    ]

                    # 3. 发送异步请求
                    response = await client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=0.1, # 低随机性，严格遵守规则
                    )

                    # 4. 提取结果并写入
                    result_text = response.choices[0].message.content.strip()
                    item['polished_label'] = result_text

                    print(f"✅ 润色成功 [{image_filename}]:")
                    print(f"   原: {original_label}")
                    print(f"   新: {result_text}\n")
                    is_success = True

                except Exception as e:
                    # 请求失败，错误回退
                    print(f"❌ 大模型请求失败 ({image_filename}): {str(e)}")
                    item['polished_label'] = original_label

        # 5. 记录当前完成的 item，更新计数器，检查是否需要增量保存
        with processed_lock:
            processed_items[item_index] = item
        with counter_lock:
            process_counter += 1
            if process_counter % SAVE_THRESHOLD == 0:
                # 异步执行保存，不阻塞事件循环
                asyncio.get_event_loop().run_in_executor(None, save_current_data, processed_items, output_json_path)

        return is_success


async def main():
    """
    用法: asyncio.run(main())
    作用: 读取标签 JSON，按参数选择样本，并发调用 VLM 润色后保存本次处理结果。
    输入: 命令行参数由 build_parser 解析。
    输出: 无返回值，处理结果写入 OUTPUT_JSON_PATH。
    """
    global process_counter
    parser = build_parser()
    args = parser.parse_args()
    json_path = args.json_path
    image_dirs = args.image_dirs
    output_json_path = args.output_json_path

    print("========================================")
    print("🚀 启动异步 VLM 标签润色流水线（颜色+形状专属版）")
    print(f"💾 每处理 {SAVE_THRESHOLD} 条自动保存，中断时触发紧急保存")
    print(f"📄 输入 JSON: {json_path}")
    print(f"🖼️ 图片目录: {', '.join(str(path) for path in image_dirs)}")
    print(f"📁 输出 JSON: {output_json_path}")
    print("========================================")

    # 1. 读取原始标签
    if not json_path.exists():
        print(f"❌ 找不到 JSON 文件: {json_path}")
        return

    missing_image_dirs = [path for path in image_dirs if not path.exists()]
    if missing_image_dirs:
        print(f"⚠️ 以下图片目录不存在，将跳过: {', '.join(str(path) for path in missing_image_dirs)}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    try:
        selected_items = select_items(data, args.limit, args.random_sample, args.seed)
    except ValueError as err:
        parser.error(str(err))

    if not selected_items:
        print("⚠️ 输入 JSON 中没有可处理的数据")
        return

    processed_items = [None] * len(selected_items)

    print(f"📦 成功加载 {len(data)} 条数据，本次选择处理 {len(selected_items)} 条")
    if args.random_sample:
        print(f"🎲 已开启随机采样，seed={args.seed}")
    print("⏳ 准备向千问发起并发请求...")

    # 2. 并发信号量
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # 3. 创建协程任务
    tasks = [
        process_single_label(item_index, item, semaphore, processed_items, image_dirs, output_json_path)
        for item_index, item in enumerate(selected_items)
    ]

    try:
        # 4. 并发执行所有任务
        results = await asyncio.gather(*tasks)

        # 5. 任务全部完成后最终保存
        save_current_data(processed_items, output_json_path)

        # 6. 统计汇总
        success_count = sum(1 for r in results if r is True)
        fallback_count = len(results) - success_count

        print("========================================")
        print(f"🎉 润色任务全部完成！")
        print(f"📊 统计数据:")
        print(f"   - 输入总数: {len(data)} 条")
        print(f"   - 本次处理: {len(results)} 条")
        print(f"   - 成功润色: {success_count} 条")
        print(f"   - 失败回退: {fallback_count} 条")
        print(f"📁 最终结果已保存至: {output_json_path}")
        print("========================================")

    except KeyboardInterrupt:
        # 捕获Ctrl+C中断，紧急保存
        print("\n⚠️  检测到手动中断，执行紧急保存...")
        save_current_data(processed_items, output_json_path)
        print(f"✅ 紧急保存完成：已处理 {process_counter} 条数据")
        print("💡 可重新运行脚本继续处理剩余数据")

if __name__ == "__main__":
    asyncio.run(main())
