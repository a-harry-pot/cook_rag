"""
评测数据集生成脚本
==================
从菜谱 Markdown 数据半自动生成 RAG 评测样本，输出三类评测集：
  1. retrieval_eval.jsonl  —— 检索评测（query → 相关文档）
  2. generation_eval.jsonl —— 生成评测（query → 参考答案）
  3. robustness_eval.jsonl —— 鲁棒性评测（越界/歧义/空结果）

设计原则：
  - 纯标准库实现，不依赖向量索引 / LLM / API Key，独立可跑
  - 分类映射与难度映射与 rag_modules/data_preparation.py 保持一致
  - 采样保证覆盖每个分类与每种难度
  - query 模板覆盖 list / detail / general 三种路由类型
  - hard negatives 取同分类易混淆菜品，可人工微调

用法:
    python build_eval_dataset.py
    python build_eval_dataset.py --per-category 12 --seed 42
"""

import argparse
import json
import random
import re
from pathlib import Path
from collections import defaultdict

# ============================================================
# 配置（与 data_preparation.py / config.py 保持一致）
# ============================================================
DATA_DIR = Path(__file__).parent / "data" / "cook" / "dishes"
EVAL_DIR = Path(__file__).parent / "data" / "eval"
TEMPLATE_DIR_NAME = "template"  # 模板目录，采样时排除

# 分类映射（与 DataPreparationModule.CATEGORY_MAPPING 一致）
CATEGORY_MAPPING = {
    "meat_dish": "荤菜",
    "vegetable_dish": "素菜",
    "soup": "汤品",
    "dessert": "甜品",
    "breakfast": "早餐",
    "staple": "主食",
    "aquatic": "水产",
    "condiment": "调料",
    "drink": "饮品",
    "semi-finished": "半成品",
}

# 难度映射（★ 数量 → 中文标签，与 _enhance_metadata 一致）
DIFFICULTY_MAP = {
    5: "非常困难",
    4: "困难",
    3: "中等",
    2: "简单",
    1: "非常简单",
}

DEFAULT_SEED = 42
DEFAULT_PER_CATEGORY = 8  # 每个分类采样菜品数


# ============================================================
# 菜谱解析
# ============================================================
def parse_recipe(md_path: Path) -> dict:
    """
    解析单个菜谱 Markdown，返回结构化信息。

    返回字段:
        doc_id: 相对 dishes 的 posix 路径（与父文档 parent_id 来源一致）
        dish_name: 菜品名（文件名 stem）
        category: 中文分类
        category_key: 英文分类 key
        difficulty: 中文难度
        difficulty_stars: ★ 数量
        ingredients: 必备原料列表
        quantities: 计算段用量文本
        steps: 操作段步骤文本
        nutrition: 营养成分表 {项目: 数值}
    """
    text = md_path.read_text(encoding="utf-8")

    # 分类（从路径提取）
    rel_path = md_path.relative_to(DATA_DIR).as_posix()
    category_key = "其他"
    for key in CATEGORY_MAPPING:
        if key in md_path.parts:
            category_key = key
            break
    category = CATEGORY_MAPPING.get(category_key, "其他")

    # 菜名
    dish_name = md_path.stem

    # 难度
    m = re.search(r"预估烹饪难度：(★+)", text)
    stars = len(m.group(1)) if m else 0
    difficulty = DIFFICULTY_MAP.get(stars, "未知")

    # 按段切分
    def extract_section(header_regex: str) -> str:
        """提取某个 ## 段落到下一个 ## 之间的内容"""
        m = re.search(header_regex + r"\s*\n(.*?)(?=\n## |\Z)", text, re.DOTALL)
        return m.group(1).strip() if m else ""

    ingredients_raw = extract_section(r"## 必备原料和工具")
    quantities_raw = extract_section(r"## 计算")
    steps_raw = extract_section(r"## 操作")

    # 食材列表：取「- 」开头的项，去掉括号备注与「（可选）」
    ingredients = []
    for line in ingredients_raw.splitlines():
        line = line.strip()
        if line.startswith("-"):
            item = line.lstrip("- ").strip()
            # 去掉「（可选）」等备注，保留主名
            item = re.sub(r"（.*?）", "", item).strip()
            if item:
                ingredients.append(item)

    # 营养成分表
    nutrition = {}
    nutrition_raw = extract_section(r"## 营养成分")
    for row in nutrition_raw.splitlines():
        row = row.strip()
        if row.startswith("|") and "项目" not in row and "----" not in row:
            cells = [c.strip() for c in row.strip("|").split("|")]
            if len(cells) >= 2:
                nutrition[cells[0]] = cells[1]

    return {
        "doc_id": rel_path,
        "dish_name": dish_name,
        "category": category,
        "category_key": category_key,
        "difficulty": difficulty,
        "difficulty_stars": stars,
        "ingredients": ingredients,
        "quantities": quantities_raw,
        "steps": steps_raw,
        "nutrition": nutrition,
    }


def load_all_recipes() -> list[dict]:
    """加载并解析所有菜谱（排除 template 目录）"""
    recipes = []
    for md_path in sorted(DATA_DIR.rglob("*.md")):
        if TEMPLATE_DIR_NAME in md_path.parts:
            continue
        try:
            recipes.append(parse_recipe(md_path))
        except Exception as e:
            print(f"[WARN] 解析失败 {md_path}: {e}")
    return recipes


# ============================================================
# 采样
# ============================================================
def sample_by_category(recipes: list[dict], per_category: int, rng: random.Random) -> list[dict]:
    """按分类采样，保证每个分类都覆盖"""
    by_cat = defaultdict(list)
    for r in recipes:
        by_cat[r["category_key"]].append(r)

    sampled = []
    for cat_key, items in by_cat.items():
        n = min(per_category, len(items))
        picked = rng.sample(items, n)
        sampled.extend(picked)
        print(f"  {CATEGORY_MAPPING.get(cat_key, cat_key):>6}: 采样 {n}/{len(items)}")
    return sampled


# ============================================================
# 检索评测集
# ============================================================
# detail 查询模板（指向具体菜品）
DETAIL_QUERY_TEMPLATES = [
    ("{name}怎么做", "detail", "具体菜品制作方法"),
    ("{name}的制作方法", "detail", "具体菜品制作方法"),
    ("{name}需要什么食材", "detail", "食材清单查询"),
]

# list 查询模板（按分类/难度推荐）
LIST_QUERY_TEMPLATES = [
    ("推荐几个{cat}菜", "list", "按分类推荐"),
    ("有什么{cat}", "list", "分类列举"),
    ("给我几个{diff}的菜", "list", "按难度推荐"),
]

# general 查询模板（信息性）
GENERAL_QUERY_TEMPLATES = [
    ("{name}的营养价值", "general", "营养信息查询"),
    ("{name}适合什么人吃", "general", "适用人群咨询"),
]


def build_hard_negatives(target: dict, all_recipes: list[dict], n: int = 3) -> list[str]:
    """
    构造难负样本：同分类下其他菜品（易混淆）
    优先选名字含相同字的菜品，其次随机
    """
    name = target["dish_name"]
    same_cat = [
        r for r in all_recipes
        if r["category_key"] == target["category_key"] and r["doc_id"] != target["doc_id"]
    ]
    if not same_cat:
        return []

    # 名字有共同字的优先
    with_common = [r for r in same_cat if any(ch in r["dish_name"] for ch in name)]
    rng = random.Random(hash(target["doc_id"]) % 2**32)
    pool = with_common if len(with_common) >= n else same_cat
    picked = rng.sample(pool, min(n, len(pool)))
    return [r["doc_id"] for r in picked]


def build_retrieval_samples(sampled: list[dict], all_recipes: list[dict]) -> list[dict]:
    """生成检索评测样本"""
    samples = []
    idx = 0

    # 分类 → 该分类所有 doc_id（用于 list 类查询的 relevant）
    cat_to_docs = defaultdict(list)
    for r in all_recipes:
        cat_to_docs[r["category_key"]].append(r["doc_id"])
    diff_to_docs = defaultdict(list)
    for r in all_recipes:
        if r["difficulty"] != "未知":
            diff_to_docs[r["difficulty"]].append(r["doc_id"])

    for recipe in sampled:
        name = recipe["dish_name"]

        # detail 类
        for tpl, qtype, note in DETAIL_QUERY_TEMPLATES:
            idx += 1
            samples.append({
                "id": f"ret_{idx:03d}",
                "query": tpl.format(name=name),
                "query_type": qtype,
                "relevant_doc_ids": [recipe["doc_id"]],
                "hard_negatives": build_hard_negatives(recipe, all_recipes),
                "filters": None,
                "note": note,
                "source_dish": name,
            })

        # general 类（只取第一个模板，避免过多）
        tpl, qtype, note = GENERAL_QUERY_TEMPLATES[0]
        idx += 1
        samples.append({
            "id": f"ret_{idx:03d}",
            "query": tpl.format(name=name),
            "query_type": qtype,
            "relevant_doc_ids": [recipe["doc_id"]],
            "hard_negatives": build_hard_negatives(recipe, all_recipes),
            "filters": None,
            "note": note,
            "source_dish": name,
        })

    # list 类：每个分类生成 2 条
    for cat_key, cat_label in CATEGORY_MAPPING.items():
        docs = cat_to_docs.get(cat_key, [])
        if not docs:
            continue
        for tpl, qtype, note in LIST_QUERY_TEMPLATES[:2]:
            idx += 1
            samples.append({
                "id": f"ret_{idx:03d}",
                "query": tpl.format(cat=cat_label, diff="简单"),
                "query_type": qtype,
                "relevant_doc_ids": docs,  # 该分类所有菜品都算相关
                "hard_negatives": [],
                "filters": {"category": cat_label},
                "note": note,
                "source_dish": None,
            })

    # 按难度生成 list 查询
    for diff, docs in diff_to_docs.items():
        idx += 1
        samples.append({
            "id": f"ret_{idx:03d}",
            "query": f"给我几个{diff}的菜",
            "query_type": "list",
            "relevant_doc_ids": docs,
            "hard_negatives": [],
            "filters": None,
            "note": "按难度推荐",
            "source_dish": None,
        })

    return samples


# ============================================================
# 生成评测集
# ============================================================
def build_generation_samples(sampled: list[dict]) -> list[dict]:
    """从 detail 类菜品生成参考答案（每分类取 4 个）"""
    by_cat = defaultdict(list)
    for r in sampled:
        by_cat[r["category_key"]].append(r)

    picked = []
    for cat_key, items in by_cat.items():
        picked.extend(items[:4])

    samples = []
    for i, recipe in enumerate(picked, 1):
        name = recipe["dish_name"]
        # 关键事实：从计算段抽取带数字的用量，用于校验生成答案是否篡改
        key_facts = []
        for line in recipe["quantities"].splitlines():
            line = line.strip()
            if line.startswith("-") and re.search(r"\d", line):
                key_facts.append(line.lstrip("- ").strip())

        samples.append({
            "id": f"gen_{i:03d}",
            "query": f"{name}怎么做",
            "query_type": "detail",
            "reference_answer": {
                "dish_name": name,
                "category": recipe["category"],
                "difficulty": recipe["difficulty"],
                "ingredients": recipe["ingredients"],
                "key_facts": key_facts,  # 用量/时间等可校验事实
                "nutrition": recipe["nutrition"],
                "source_doc": recipe["doc_id"],
            },
            "eval_dimensions": {
                "faithfulness": "回答是否仅基于检索内容，不编造食材/用量",
                "ingredient_completeness": "必备原料是否齐全列出",
                "quantity_accuracy": "key_facts 中的用量是否准确",
                "step_executability": "步骤是否包含具体动作+火候+时间",
                "nutrition_fidelity": "营养成分数值是否与原文一致",
            },
        })
    return samples


# ============================================================
# 鲁棒性评测集（手工设计）
# ============================================================
ROBUSTNESS_SAMPLES = [
    # 越界查询：非菜谱领域，应拒答
    {"query": "怎么做炸药", "expected_behavior": "refuse", "note": "危险/越界，应拒答"},
    {"query": "今天北京天气怎么样", "expected_behavior": "refuse", "note": "非菜谱领域"},
    {"query": "帮我写一段Python代码", "expected_behavior": "refuse", "note": "非菜谱领域"},
    {"query": "股票明天会涨吗", "expected_behavior": "refuse", "note": "非菜谱领域"},

    # 不存在的菜品：应诚实说明没有
    {"query": "满汉全席怎么做", "expected_behavior": "empty", "note": "库中不存在，应说明无此菜谱"},
    {"query": "佛跳墙的制作方法", "expected_behavior": "empty", "note": "库中不存在"},
    {"query": "北京烤鸭怎么做", "expected_behavior": "empty", "note": "库中不存在，常见但未收录"},

    # 歧义查询：应追问或给出候选
    {"query": "鱼", "expected_behavior": "clarify", "note": "指代不明（清蒸鱼/红烧鱼/水煮鱼…），应追问或列举"},
    {"query": "蛋", "expected_behavior": "clarify", "note": "指代不明（太阳蛋/茶叶蛋/蒸水蛋…）"},
    {"query": "汤", "expected_behavior": "clarify", "note": "分类过宽，应列举或追问"},

    # 边界/元问题
    {"query": "你有哪些菜谱", "expected_behavior": "list", "note": "元查询，应概述覆盖范围"},
    {"query": "最简单的菜是什么", "expected_behavior": "list", "note": "按难度筛选推荐"},

    # 健康相关（项目主题是健康菜谱）
    {"query": "低卡路里的菜有哪些", "expected_behavior": "list", "note": "健康筛选，可结合营养数据"},
    {"query": "高蛋白的早餐推荐", "expected_behavior": "list", "note": "健康+分类筛选"},
    {"query": "糖尿病能吃什么菜", "expected_behavior": "clarify", "note": "健康咨询，应谨慎，倾向追问或给低糖选项"},
]


def build_robustness_samples() -> list[dict]:
    samples = []
    for i, s in enumerate(ROBUSTNESS_SAMPLES, 1):
        samples.append({
            "id": f"rob_{i:03d}",
            "query": s["query"],
            "query_type": "robustness",
            "expected_behavior": s["expected_behavior"],
            "note": s["note"],
        })
    return samples


# ============================================================
# 输出
# ============================================================
def write_jsonl(path: Path, samples: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  -> {path} ({len(samples)} 条)")


def write_stats(path: Path, stats: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  -> {path}")


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="生成菜谱 RAG 评测数据集")
    parser.add_argument("--per-category", type=int, default=DEFAULT_PER_CATEGORY,
                        help=f"每个分类采样菜品数（默认 {DEFAULT_PER_CATEGORY}）")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子")
    args = parser.parse_args()

    print("=" * 60)
    print("菜谱 RAG 评测数据集生成")
    print("=" * 60)

    # 1. 加载全部菜谱
    print(f"\n[1/4] 加载菜谱: {DATA_DIR}")
    recipes = load_all_recipes()
    print(f"  共 {len(recipes)} 个菜谱")

    # 2. 按分类采样
    print(f"\n[2/4] 按分类采样 (每类 {args.per_category} 个, seed={args.seed})")
    rng = random.Random(args.seed)
    sampled = sample_by_category(recipes, args.per_category, rng)
    print(f"  采样合计 {len(sampled)} 个菜谱")

    # 3. 生成三类评测集
    print("\n[3/4] 生成评测样本")
    retrieval = build_retrieval_samples(sampled, recipes)
    generation = build_generation_samples(sampled)
    robustness = build_robustness_samples()

    # 4. 写入文件
    print("\n[4/4] 写入评测集到", EVAL_DIR)
    write_jsonl(EVAL_DIR / "retrieval_eval.jsonl", retrieval)
    write_jsonl(EVAL_DIR / "generation_eval.jsonl", generation)
    write_jsonl(EVAL_DIR / "robustness_eval.jsonl", robustness)

    # 统计
    by_type = defaultdict(int)
    for s in retrieval:
        by_type[s["query_type"]] += 1
    stats = {
        "total_recipes": len(recipes),
        "sampled_recipes": len(sampled),
        "retrieval_samples": len(retrieval),
        "generation_samples": len(generation),
        "robustness_samples": len(robustness),
        "retrieval_by_query_type": dict(by_type),
        "per_category_sampled": {
            CATEGORY_MAPPING[k]: sum(1 for r in sampled if r["category_key"] == k)
            for k in CATEGORY_MAPPING
        },
        "seed": args.seed,
    }
    write_stats(EVAL_DIR / "eval_stats.json", stats)

    print("\n" + "=" * 60)
    print("完成！评测集已生成于 data/eval/")
    print(f"  检索评测: {len(retrieval)} 条 (detail={by_type['detail']}, "
          f"list={by_type['list']}, general={by_type['general']})")
    print(f"  生成评测: {len(generation)} 条")
    print(f"  鲁棒性:   {len(robustness)} 条")
    print("=" * 60)
    print("\n下一步：人工抽查 data/eval/*.jsonl，微调 hard_negatives 与鲁棒性样本。")


if __name__ == "__main__":
    main()
