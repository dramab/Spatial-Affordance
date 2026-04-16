"""
======================================================================
润色结果合法性校验 + 修饰检索二合一脚本
======================================================================
功能:
1. 逐条校验润色后的label：原句所有单词必须存在，且顺序完全一致
2. 智能修饰检索：自动找出所有新增修饰词，判断修饰关系并归类
3. 生成详细的校验统计报告 + 修饰词统计报告
4. 支持保存失败条目和修饰检索结果

用法示例:
   python validate_and_extract.py
======================================================================
"""

import json
from pathlib import Path
from collections import defaultdict

# ================= 配置区域 =================
# 原始标注JSON路径
ORIGINAL_JSON_PATH = Path("../outputs/auto_labels/all_labels.json")
# 润色后JSON路径
POLISHED_JSON_PATH = Path("../outputs/auto_labels/all_labels_polished.json")
# 校验失败条目保存路径
FAILED_JSON_PATH = Path("../outputs/auto_labels/all_labels_validation_failed.json")
# 修饰检索结果保存路径
MODIFIERS_JSON_PATH = Path("../outputs/auto_labels/all_labels_modifiers_extracted.json")
# ============================================

def validate_polished_label(original: str, polished: str) -> bool:
    """
    校验润色后的句子是否合法：
    1. 原句中出现的所有单词，新句中必须全部存在
    2. 原句单词的相对顺序，在新句中必须完全一致
    3. 允许新句在原句单词之间插入其他修饰词
    """
    original_words = original.strip().split()
    polished_words = polished.strip().split()
    
    original_ptr = 0
    original_total = len(original_words)
    
    for word in polished_words:
        if original_ptr >= original_total:
            break
        if word == original_words[original_ptr]:
            original_ptr += 1
    
    return original_ptr == original_total

def extract_modifiers(original: str, polished: str) -> tuple[dict, list]:
    """
    智能提取润色后句子中的修饰词及其修饰关系
    
    Args:
        original: 原始标注句子
        polished: 润色后的句子
    
    Returns:
        tuple: (modifier_dict, all_modifiers)
            - modifier_dict: 字典 {被修饰词: [修饰词列表]}
            - all_modifiers: 所有用到的修饰词列表（去重）
    """
    original_words = original.strip().split()
    polished_words = polished.strip().split()
    
    modifier_dict = defaultdict(list)
    all_modifiers = set()
    
    original_ptr = 0
    original_total = len(original_words)
    
    # 临时存储当前累积的修饰词，等待遇到下一个原句词时分配
    current_modifiers = []
    
    for word in polished_words:
        if original_ptr < original_total and word == original_words[original_ptr]:
            # 遇到了原句中的词，将之前累积的修饰词分配给它
            if current_modifiers:
                modifier_dict[word].extend(current_modifiers)
                all_modifiers.update(current_modifiers)
                current_modifiers = []  # 清空临时修饰词列表
            original_ptr += 1
        else:
            # 这是新增的修饰词，加入临时列表
            current_modifiers.append(word)
    
    # 处理句尾可能剩余的修饰词（虽然按规则应该不会有，但以防万一）
    if current_modifiers and original_ptr > 0:
        last_original_word = original_words[original_ptr - 1]
        modifier_dict[last_original_word].extend(current_modifiers)
        all_modifiers.update(current_modifiers)
    
    # 将defaultdict转为普通dict，方便JSON序列化
    modifier_dict = dict(modifier_dict)
    # 将set转为排序后的list
    all_modifiers = sorted(list(all_modifiers))
    
    return modifier_dict, all_modifiers

def main():
    print("========================================")
    print("🔍 启动润色结果合法性校验 + 修饰检索")
    print("========================================")

    # 1. 检查文件是否存在
    if not ORIGINAL_JSON_PATH.exists():
        print(f"❌ 找不到原始标注文件: {ORIGINAL_JSON_PATH}")
        return
    if not POLISHED_JSON_PATH.exists():
        print(f"❌ 找不到润色后文件: {POLISHED_JSON_PATH}")
        return

    # 2. 读取JSON数据
    print(f"📖 正在读取数据...")
    with open(ORIGINAL_JSON_PATH, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    with open(POLISHED_JSON_PATH, 'r', encoding='utf-8') as f:
        polished_data = json.load(f)

    # 3. 构建原始数据的索引
    original_dict = {item['image_filename']: item for item in original_data if 'image_filename' in item}
    print(f"📦 原始数据: {len(original_data)} 条")
    print(f"📦 润色后数据: {len(polished_data)} 条")

    # 4. 逐条校验 + 修饰检索
    total_count = 0
    success_count = 0
    failed_count = 0
    failed_items = []
    all_modifiers_extraction = []  # 存储所有条目的修饰检索结果
    global_modifier_dict = defaultdict(list)  # 全局统计：每个被修饰词的所有修饰词
    global_all_modifiers = set()  # 全局统计：所有用到的修饰词

    print(f"\n🔍 开始处理...")
    for polished_item in polished_data:
        total_count += 1
        image_filename = polished_item.get('image_filename', 'unknown_filename')
        original_label = polished_item.get('label', '')
        polished_label = polished_item.get('polished_label', '')

        # 检查必要字段是否存在
        if not original_label or not polished_label:
            print(f"⚠️  跳过 (字段缺失): {image_filename}")
            failed_count += 1
            failed_items.append({
                'image_filename': image_filename,
                'reason': 'Missing original_label or polished_label',
                'original_label': original_label,
                'polished_label': polished_label
            })
            continue

        # 数据对齐校验
        data_aligned = True
        if image_filename in original_dict:
            original_item = original_dict[image_filename]
            if original_item.get('label') != original_label:
                data_aligned = False

        if not data_aligned:
            print(f"❌ 校验失败 (数据错位): {image_filename}")
            failed_count += 1
            failed_items.append({
                'image_filename': image_filename,
                'reason': 'Original label mismatch (possible data misalignment)',
                'original_label': original_item.get('label'),
                'polished_original_label': original_label,
                'polished_label': polished_label
            })
            continue

        # 核心合法性校验
        is_valid = validate_polished_label(original_label, polished_label)

        if is_valid:
            success_count += 1
            # 执行修饰检索
            modifier_dict, all_modifiers = extract_modifiers(original_label, polished_label)
            
            # 保存当前条目的检索结果
            all_modifiers_extraction.append({
                'image_filename': image_filename,
                'original_label': original_label,
                'polished_label': polished_label,
                'modifier_dict': modifier_dict,
                'all_modifiers_in_sentence': all_modifiers
            })
            
            # 更新全局统计
            for target_word, mods in modifier_dict.items():
                global_modifier_dict[target_word].extend(mods)
            global_all_modifiers.update(all_modifiers)
            
            # 每100条打印一次进度
            if success_count % 100 == 0:
                print(f"✅ 已处理 {success_count} 条通过...")
        else:
            failed_count += 1
            print(f"❌ 校验失败 (单词/顺序问题): {image_filename}")
            failed_items.append({
                'image_filename': image_filename,
                'reason': 'Original words missing or order changed',
                'original_label': original_label,
                'polished_label': polished_label
            })

    # 5. 后处理全局统计：对每个被修饰词的修饰词去重并排序
    for target_word in global_modifier_dict:
        # 去重并按字母顺序排序
        global_modifier_dict[target_word] = sorted(list(set(global_modifier_dict[target_word])))
    global_modifier_dict = dict(global_modifier_dict)
    global_all_modifiers = sorted(list(global_all_modifiers))

    # 6. 保存结果文件
    if failed_items:
        with open(FAILED_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(failed_items, f, indent=2, ensure_ascii=False)
        print(f"\n💾 校验失败条目已保存至: {FAILED_JSON_PATH}")

    if all_modifiers_extraction:
        # 构建完整的修饰检索结果输出
        full_modifier_output = {
            'global_statistics': {
                'total_processed': total_count,
                'successfully_extracted': len(all_modifiers_extraction),
                'global_modifier_dict': global_modifier_dict,
                'all_modifiers_used': global_all_modifiers
            },
            'per_sentence_extraction': all_modifiers_extraction
        }
        with open(MODIFIERS_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(full_modifier_output, f, indent=2, ensure_ascii=False)
        print(f"💾 修饰检索结果已保存至: {MODIFIERS_JSON_PATH}")

    # 7. 输出完整统计报告
    print("\n========================================")
    print("📊 完整统计报告")
    print("========================================")
    print(f"【校验统计】")
    print(f"   总计处理: {total_count} 条")
    print(f"   ✅ 校验通过: {success_count} 条 ({success_count/total_count*100:.2f}%)")
    print(f"   ❌ 校验失败: {failed_count} 条 ({failed_count/total_count*100:.2f}%)")
    
    if global_all_modifiers:
        print(f"\n【修饰词统计】")
        print(f"   共使用修饰词: {len(global_all_modifiers)} 个")
        print(f"   所有修饰词: {', '.join(global_all_modifiers)}")
        print(f"\n   被修饰词及其修饰词:")
        for target_word, mods in global_modifier_dict.items():
            print(f"     - {target_word}: {', '.join(mods)}")
    print("========================================")

if __name__ == "__main__":
    main()