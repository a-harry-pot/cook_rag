"""
生成评测入口脚本
==================
构建完整 RAG 系统（含 LLM），跑生成质量评测（LLM-as-Judge）+ 路由准确率。

需要 DEEPSEEK_API_KEY（从 .env 加载）。
因 FAISS save/load docstore 错位问题，每次重建向量索引。

用法:
    python rag_modules/evaluation/run_generation_eval.py
    python rag_modules/evaluation/run_generation_eval.py --limit-gen 10 --router-limit 100
"""

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# 脚本位于 rag_modules/evaluation/，需把项目根加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import DEFAULT_CONFIG
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule,
)
from rag_modules.evaluation import GenerationEvaluator

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GEN_EVAL_FILE = PROJECT_ROOT / "data" / "eval" / "generation_eval.jsonl"
RET_EVAL_FILE = PROJECT_ROOT / "data" / "eval" / "retrieval_eval.jsonl"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_jsonl(path: Path) -> list:
    """加载 jsonl，兼容单行与多行 pretty-print 两种格式"""
    content = path.read_text(encoding="utf-8")
    samples, decoder = [], json.JSONDecoder()
    idx, n = 0, len(content)
    while idx < n:
        while idx < n and content[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, idx = decoder.raw_decode(content, idx)
        samples.append(obj)
    return samples


def build_rag_system():
    """构建完整 RAG 系统：data → index → retrieval → generation"""
    data_path = PROJECT_ROOT / DEFAULT_CONFIG.data_path
    print(f"[1/5] 加载文档: {data_path}")
    data_module = DataPreparationModule(str(data_path))
    data_module.load_documents()
    chunks = data_module.chunk_documents()
    stats = data_module.get_statistics()
    print(f"   文档 {stats['total_documents']} 个, 分块 {stats['total_chunks']} 个")

    print("[2/5] 构建向量索引（重建，规避 save/load 错位）...")
    index_module = IndexConstructionModule(
        model_name=DEFAULT_CONFIG.embedding_model,
        index_save_path=str(PROJECT_ROOT / DEFAULT_CONFIG.index_save_path),
    )
    vectorstore = index_module.build_vector_index(chunks)

    print("[3/5] 初始化检索模块（含 jieba BM25）...")
    retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)
    retrieval_module.data_module = data_module

    print("[4/5] 初始化生成模块（LLM）...")
    generation_module = GenerationIntegrationModule(
        model_name=DEFAULT_CONFIG.llm_model,
        temperature=DEFAULT_CONFIG.temperature,
        max_tokens=DEFAULT_CONFIG.max_tokens,
    )

    print("[5/5] 就绪")
    return data_module, retrieval_module, generation_module


def main():
    parser = argparse.ArgumentParser(description="生成评测")
    parser.add_argument("--limit-gen", type=int, default=0,
                        help="生成评测样本上限（0=全部40条）")
    parser.add_argument("--router-limit", type=int, default=0,
                        help="路由评测样本上限（0=全部345条）")
    args = parser.parse_args()

    load_dotenv()  # 加载 DEEPSEEK_API_KEY
    print("=" * 64)
    print("菜谱 RAG 生成评测（P1）")
    print("=" * 64)

    data_module, retrieval_module, generation_module = build_rag_system()
    dishes_root = PROJECT_ROOT / DEFAULT_CONFIG.data_path / "dishes"
    evaluator = GenerationEvaluator(retrieval_module, generation_module,
                                    data_module, dishes_root, top_k=DEFAULT_CONFIG.top_k)

    # ---- 生成评测 ----
    print(f"\n加载生成评测集: {GEN_EVAL_FILE}")
    gen_samples = load_jsonl(GEN_EVAL_FILE)
    if args.limit_gen > 0:
        gen_samples = gen_samples[:args.limit_gen]
    print(f"  生成评测样本: {len(gen_samples)} 条")

    print("\n开始生成评测（每条需 3-4 次 LLM 调用，请耐心等待）...")
    t0 = time.time()
    gen_report = evaluator.run(gen_samples)
    gen_report["elapsed_seconds"] = round(time.time() - t0, 1)
    print(f"  生成评测完成，耗时 {gen_report['elapsed_seconds']}s")

    # ---- 路由准确率 ----
    print(f"\n加载检索评测集(用于路由): {RET_EVAL_FILE}")
    ret_samples = load_jsonl(RET_EVAL_FILE)
    if args.router_limit > 0:
        ret_samples = ret_samples[:args.router_limit]
    print(f"  路由评测样本: {len(ret_samples)} 条")

    print("\n开始路由准确率评测...")
    t1 = time.time()
    router_report = evaluator.eval_router_accuracy(ret_samples)
    router_report["elapsed_seconds"] = round(time.time() - t1, 1)
    print(f"  路由评测完成，耗时 {router_report['elapsed_seconds']}s")

    # ---- 输出报告 ----
    print("\n" + GenerationEvaluator.format_report(gen_report, router_report))

    # 保存
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "generation_eval_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"generation": gen_report, "router": router_report},
                  f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {out_path}")


if __name__ == "__main__":
    main()
