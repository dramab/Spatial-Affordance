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
   python api_label.py
======================================================================
"""

import json
import base64
import asyncio
from pathlib import Path
from openai import AsyncOpenAI
import threading  # 用于线程安全的计数器和保存锁

# ================= 配置区域 =================
# 阿里云百炼 API Key
API_KEY = 'sk-c0897cbd1d1f4b0d91447b9b2b673cb6'  # 替换为你的真实 Key

# 文件路径配置
JSON_PATH = Path("../outputs/auto_labels/all_labels.json")           # 原始自动标注生成的 JSON
IMAGE_DIR = Path("../outputs/placement_rgb_bbox_vis")                # 图片所在目录
OUTPUT_JSON_PATH = Path("../outputs/auto_labels/all_labels_polished.json") # 润色后输出的新 JSON

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
save_lock = threading.Lock()

def encode_image(image_path: Path) -> str:
    """将图片编码为 Base64 字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_polish_prompt(original_label: str) -> str:
    """专属优化Prompt：仅允许颜色+形状描述，彻底移除材质相关要求，保留位置精准增强"""
    return f"""
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

def save_current_data(data: list):
    """保存当前数据到JSON文件（加锁避免并发写入）"""
    with save_lock:
        try:
            with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"\n💾 增量保存成功：已处理 {process_counter} 条，文件路径: {OUTPUT_JSON_PATH}\n")
        except Exception as e:
            print(f"\n❌ 增量保存失败: {str(e)}\n")

async def process_single_label(item: dict, semaphore: asyncio.Semaphore, data: list) -> bool:
    """处理单条标签数据的异步任务"""
    global process_counter
    async with semaphore:
        image_filename = item.get('image_filename')
        original_label = item.get('label')
        
        # 异常数据过滤
        if not image_filename or not original_label:
            with counter_lock:
                process_counter += 1
            item['polished_label'] = original_label
            return False

        image_path = IMAGE_DIR / image_filename
        
        # 1. 图片不存在，直接回退为原始标签
        if not image_path.exists():
            print(f"[-] 图片丢失，跳过润色: {image_filename}")
            item['polished_label'] = original_label
            with counter_lock:
                process_counter += 1
            return False

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
            is_success = False
        
        # 5. 更新计数器，检查是否需要增量保存
        with counter_lock:
            process_counter += 1
            if process_counter % SAVE_THRESHOLD == 0:
                # 异步执行保存，不阻塞事件循环
                asyncio.get_event_loop().run_in_executor(None, save_current_data, data)
        
        return is_success

async def main():
    global process_counter
    print("========================================")
    print("🚀 启动异步 VLM 标签润色流水线（颜色+形状专属版）")
    print(f"💾 每处理 {SAVE_THRESHOLD} 条自动保存，中断时触发紧急保存")
    print("========================================")

    # 1. 读取原始标签
    if not JSON_PATH.exists():
        print(f"❌ 找不到 JSON 文件: {JSON_PATH}")
        return

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"📦 成功加载 {len(data)} 条数据，准备向千问发起并发请求...")

    # 2. 并发信号量
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # 3. 创建协程任务
    tasks = [process_single_label(item, semaphore, data) for item in data]
    
    try:
        # 4. 并发执行所有任务
        results = await asyncio.gather(*tasks)

        # 5. 任务全部完成后最终保存
        save_current_data(data)

        # 6. 统计汇总
        success_count = sum(1 for r in results if r is True)
        fallback_count = len(results) - success_count
        
        print("========================================")
        print(f"🎉 润色任务全部完成！")
        print(f"📊 统计数据:")
        print(f"   - 总计处理: {len(data)} 条")
        print(f"   - 成功润色: {success_count} 条")
        print(f"   - 失败回退: {fallback_count} 条")
        print(f"📁 最终结果已保存至: {OUTPUT_JSON_PATH}")
        print("========================================")

    except KeyboardInterrupt:
        # 捕获Ctrl+C中断，紧急保存
        print("\n⚠️  检测到手动中断，执行紧急保存...")
        save_current_data(data)
        print(f"✅ 紧急保存完成：已处理 {process_counter} 条数据")
        print("💡 可重新运行脚本继续处理剩余数据")

if __name__ == "__main__":
    asyncio.run(main())