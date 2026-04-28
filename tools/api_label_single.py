"""
======================================================================
单样本多模态标签润色脚本 (颜色+形状专属版)
======================================================================
功能：对单张图片进行标签润色，仅对物体添加「颜色+形状」描述。
特点:
1. 同步调用：直接调用 VLM API，无需异步并发
2. 精准润色：仅允许颜色+形状描述，零改动原句核心内容
3. 容错回退：API 调用失败时，自动回退到原始标签

用法示例:
   python tools/api_label_single.py
======================================================================
"""

import base64
from pathlib import Path
from openai import OpenAI

# ================= 配置区域 =================
# 阿里云百炼 API Key
API_KEY = 'sk-c0897cbd1d1f4b0d91447b9b2b673cb6'  # 替换为你的真实 Key

# 图片路径（命令行参数传入）
IMAGE_PATH = Path("/data/jiajun.xie/Spatial-Affordance/outputs/placement_rgb_bbox_vis_ycbv/ycbv_test__000048_000320_obj_000006_1_p000.png")  # 待润色的图片路径

# 原始标签（直接在此处设定）
ORIGINAL_LABEL = "Move TunaCan located at the left of Clamp to the right of Clamp."

# 选择模型: 视觉增强任务推荐使用 qwen-vl-max 或 qwen-vl-plus
MODEL_NAME = 'qwen3.6-plus'
# ============================================

# 初始化客户端 (使用阿里云兼容 OpenAI 的 endpoint)
client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def encode_image(image_path: Path) -> str:
    """将图片编码为 Base64 字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_polish_prompt(original_label: str) -> str:
    """专属优化Prompt：仅允许颜色+形状描述，彻底移除材质相关要求，保留位置精准增强"""
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
Your ONLY task is to enrich the provided original English operation instruction based STRICTLY on the visual evidence in the accompanying image, with ZERO changes to the core content of the original sentence.
Image Context Information: - The specific object that needs to be moved is highlighted with an orange bounding box.
* The final empty placement destination (the spatial void where the object will go) is highlighted with a green bounding box.
* The spatial reference object(s) mentioned in the original instruction are the physical objects located around or near this green box.CRITICAL: These boxes are strictly for YOUR reference to locate the target and the placement area. You MUST NOT mention the boxes, the colors of the boxes, or the word "box" (e.g., "orange-boxed", "green box") in your final enriched sentence.
Original Operation Instruction: "{original_label}"
Reference Image: [User Uploads Image]
============= ABSOLUTE NON-NEGOTIABLE RULES =============
1. STRICT IMAGE GROUNDING & VISUAL CONTEXT: Every single enrichment MUST accurately reflect the actual visual evidence. Do not hallucinate. Use the orange box to identify the moving target. Use the green box to locate the empty destination area so you can correctly visually identify the physical reference objects surrounding it. Describe all objects using ONLY their intrinsic visual properties.
2. ZERO CORE CONTENT ALTERATIONS: You MUST NOT change, delete, reorder, or rewrite any single core word from the original sentence, including all object nouns, action verbs, core prepositions of position (behind/front/left/right/top/bottom/in front of etc.), punctuation, capitalization, and sentence structure.
3. PERMITTED ENRICHMENT 1 (OBJECT VISUAL MODIFICATION - ONLY 2 TYPES ALLOWED): You may add 1-2 descriptive adjectives immediately before concrete object nouns in the sentence, derived entirely from the image. Adjectives are STRICTLY LIMITED to ONLY two categories:
    * Color attributes (e.g., red, blue, yellow, white, black, grey, green, silver etc.)
    * Shape/geometry attributes (e.g., round, cylindrical, rectangular, square, flat, long, short, tall, slender, curved,big,small etc.)❌ FORBIDDEN: Any adjectives describing material, packaging, bounding boxes, annotations, highlights, brand, texture, or other non-color/non-shape attributes are strictly prohibited.
4. MULTIPLE INSTANCE DISAMBIGUATION (CRITICAL): If the image contains multiple objects of the same category (e.g., several cups), use the orange box to locate the exact target, and look near the green box to locate the correct reference object. You MUST use ONLY their intrinsic distinguishing attributes (color, shape) in your text to uniquely identify them and distinguish them from unboxed or distant distractors.
5. PERMITTED ENRICHMENT 2 (POSITION PRECISION ENHANCEMENT): You may add precise spatial adverbs immediately before the prepositional phrases of relative position in the original sentence, based on the real spatial distances shown in the image. Allowed adverbs include: directly, immediately, slightly, far, closely, exactly, just. YOU MUST NOT MODIFY THE ORIGINAL POSITION PREPOSITION ITSELF.
6. FULL VISIBLE COVERAGE REQUIREMENT: You must add appropriate adjectives (color/shape) to every concrete object noun in the sentence, regardless of whether it is the action target or a position reference object, provided those attributes are clearly visible.
7. OUTPUT REQUIREMENT: You MUST output ONLY the final enriched complete English sentence. No explanations, notes, brackets, or extra content of any kind.
============= CORRECT EXAMPLES =============
Example 1 (Basic Context):
Image Context: A red, round apple (orange box) is on the table. A green box marks an empty spot. Directly behind this green empty spot is a white, flat plate. The empty spot is slightly to the right of a long, silver fork.
Original Input: "Move the apple located behind the plate to the right of the fork."
Correct Output: "Move the red round apple located directly behind the white flat plate to slightly to the right of the silver slender fork."
Example 2 (Disambiguation with Empty Placement Area):
Image Context: There are three identical-looking cups on the table, but they have different colors. A blue, cylindrical cup is in an orange box. A green box marks an empty spot on the table. Right next to this green box is a yellow, cylindrical cup.
Original Input: "Move the cup next to the cup."
Correct Output: "Move the blue cylindrical cup to immediately next to the yellow cylindrical cup."
============= FORBIDDEN EXAMPLES =============
Forbidden 1 (changed core word): "Move the fruit located behind the plate..." (changed "apple" to "fruit")
Forbidden 2 (mentioned bounding boxes): "Move the blue cup into the green box..." (VIOLATION: mentioned the reference boxes)
Forbidden 3 (added forbidden material description): "Move the red apple behind the ceramic plate..." (added material word "ceramic")
Forbidden 4 (modified position logic): "Move the apple located in front of the plate..." (changed original preposition)
Forbidden 5 (hallucinated visual attribute): "Move the green square apple..." (added attributes contradicting the image)
Forbidden 6 (extra content): "Enriched instruction: Move the red apple..." (added extra explanation)
Now, output your refined instruction strictly following the rules above:
"""

    prompt3 = f"""
You are a professional, strictly rule-abiding robotic operation annotation refinement specialist.
Your ONLY task is to enrich the provided original English operation instruction based STRICTLY on the visual evidence in the accompanying image, with ZERO changes to the core content of the original sentence.
Image Context Information: - The specific object that needs to be moved is highlighted with an orange bounding box.
* The final empty placement destination (the spatial void where the object will go) is highlighted with a green bounding box.
* The original instruction may mention spatial reference objects for both the starting position and the final placement position. To identify them visually: the reference object for the starting position is the physical object closest to the orange box; the reference object for the placement position is the physical object closest to the green box.CRITICAL: These boxes are strictly for YOUR reference to locate the target, the destination, and their respective closest reference objects. You MUST NOT mention the boxes, the colors of the boxes, or the word "box" (e.g., "orange-boxed", "green box") in your final enriched sentence.
Original Operation Instruction: "{original_label}"
Reference Image: [User Uploads Image]
============= ABSOLUTE NON-NEGOTIABLE RULES =============
1. STRICT IMAGE GROUNDING & VISUAL CONTEXT: Every single enrichment MUST accurately reflect the actual visual evidence. Do not hallucinate. Use the orange box to identify the moving target, and use the green box to locate the empty destination. Identify the reference objects in the original sentence by finding the physical objects that are structurally closest to these respective boxes. Describe all objects using ONLY their intrinsic visual properties.
2. ZERO CORE CONTENT ALTERATIONS: You MUST NOT change, delete, reorder, or rewrite any single core word from the original sentence, including all object nouns, action verbs, core prepositions of position (behind/front/left/right/top/bottom/in front of etc.), punctuation, capitalization, and sentence structure.
3. PERMITTED ENRICHMENT 1 (OBJECT VISUAL MODIFICATION - ONLY 3 TYPES ALLOWED): You may add 1-2 descriptive adjectives immediately before concrete object nouns in the sentence, derived entirely from the image. Adjectives are STRICTLY LIMITED to ONLY three categories:
    * Color attributes (e.g., red, blue, yellow, white, black, grey, green, silver etc.)
    * Shape/geometry attributes (e.g., round, cylindrical, rectangular, square, flat, curved etc.)
    * Size/scale attributes (e.g., small, large, smaller, bigger, tall, short, long, slender, thick etc.)❌ FORBIDDEN: Any adjectives describing material, packaging, bounding boxes, annotations, highlights, brand, texture, or other non-color/non-shape/non-size attributes are strictly prohibited.
4. MULTIPLE INSTANCE DISAMBIGUATION (CRITICAL): If the image contains multiple objects of the same category (e.g., several cups, multiple clamps), use the orange box to locate the exact target, and look for the closest physical objects to the respective boxes to identify the correct reference objects. You MUST use ONLY their intrinsic distinguishing attributes (color, shape, size) in your text to uniquely identify them and distinguish them from unboxed or distant distractors.
5. PERMITTED ENRICHMENT 2 (POSITION PRECISION ENHANCEMENT): You may add precise spatial adverbs immediately before the prepositional phrases of relative position in the original sentence, based on the real spatial distances shown in the image. Allowed adverbs include: directly, immediately, slightly, far, closely, exactly, just. YOU MUST NOT MODIFY THE ORIGINAL POSITION PREPOSITION ITSELF.
6. FULL VISIBLE COVERAGE REQUIREMENT: You must add appropriate adjectives (color/shape/size) to every concrete object noun in the sentence, regardless of whether it is the action target or a position reference object, provided those attributes are clearly visible.
7. OUTPUT REQUIREMENT: You MUST output ONLY the final enriched complete English sentence. No explanations, notes, brackets, or extra content of any kind.
============= CORRECT EXAMPLES =============
Example 1 (Start and End Reference Context):
Image Context: A red, round apple (orange box) is behind a white, flat plate (the closest object to the orange box). A green box marks an empty spot. The object closest to this green empty spot is a long, silver fork, and the spot is to the right of it.
Original Input: "Move the apple located behind the plate to the right of the fork."
Correct Output: "Move the red round apple located directly behind the white flat plate to slightly to the right of the silver slender fork."
Example 2 (Disambiguation with Closest Reference Logic):
Image Context: There are multiple cups. A blue, cylindrical cup is in the orange box, closest to a red, square block. The green box (destination) is closest to a yellow, cylindrical cup.
Original Input: "Move the cup near the block to the cup."
Correct Output: "Move the blue cylindrical cup closely near the red square block to the yellow cylindrical cup."
Example 3 (Disambiguation with Size and Same-Category References):
Image Context: A blue, cylindrical TunaCan (orange box) is on a surface. There are two clamps. The TunaCan is positioned next to the standing, smaller black Clamp (closest to the orange box). The green box (destination) is located on the right side of the flat, bigger black Clamp.
Original Input: "Move TunaCan located at the left of Clamp to the right of Clamp."
Correct Output: "Move blue cylindrical TunaCan located closely at the left of black smaller Clamp to slightly to the right of bigger black Clamp."
============= FORBIDDEN EXAMPLES =============
Forbidden 1 (changed core word): "Move the fruit located behind the plate..." (changed "apple" to "fruit")
Forbidden 2 (mentioned bounding boxes): "Move the blue cup into the green box..." (VIOLATION: mentioned the reference boxes)
Forbidden 3 (added forbidden material description): "Move the red apple behind the ceramic plate..." (added material word "ceramic")
Forbidden 4 (modified position logic): "Move the apple located in front of the plate..." (changed original preposition)
Forbidden 5 (hallucinated visual attribute): "Move the green square apple..." (added attributes contradicting the image)
Forbidden 6 (extra content): "Enriched instruction: Move the red apple..." (added extra explanation)
Now, output your refined instruction strictly following the rules above:
"""
    return prompt3

def polish_single_sample(image_path: Path, original_label: str) -> dict:
    """
    处理单条样本的标签润色

    Args:
        image_path: 图片路径
        original_label: 原始标签文本

    Returns:
        包含原始标签和润色后标签的字典
    """
    result = {
        'image_path': str(image_path),
        'original_label': original_label,
        'polished_label': original_label  # 默认回退原标签
    }

    # 检查图片是否存在
    if not image_path.exists():
        print(f"❌ 图片不存在: {image_path}")
        return result

    # 生成优化后的Prompt和Base64图片
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

        # 发送同步请求
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1,  # 低随机性，严格遵守规则
        )

        # 提取结果
        polished_label = response.choices[0].message.content.strip()
        result['polished_label'] = polished_label

    except Exception as e:
        print(f"❌ API 请求失败: {str(e)}")

    return result

def print_result(result: dict):
    """格式化打印结果"""
    print("\n" + "=" * 60)
    print("📋 润色结果")
    print("=" * 60)
    print(f"📷 图片路径: {result['image_path']}")
    print(f"📝 原始标签: {result['original_label']}")
    print(f"✨ 润色标签: {result['polished_label']}")
    print("=" * 60)

def main():
    print("========================================")
    print("🚀 启动单样本 VLM 标签润色（颜色+形状专属版）")
    print("========================================")

    # 处理样本
    result = polish_single_sample(IMAGE_PATH, ORIGINAL_LABEL)

    # 打印结果
    print_result(result)

if __name__ == "__main__":
    main()
