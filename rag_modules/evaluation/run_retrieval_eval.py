"""
检索评测入口脚本
=================
独立构建 data/index/retrieval 模块（不初始化 LLM，无需 API Key），
加载评测集跑三路检索对比 + 专项评测，输出报告。

用法:
    python run_retrieval_eval.py
    python run_retrieval_eval.py --routes vector bm25 hybrid --k 1 3 5
"""

import argparse
import json
import time
import sys
from pathlib import Path

# 脚本位于 rag_modules/evaluation/，需把项目根加入 sys.path 以便 import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import DEFAULT_CONFIG
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
)
from rag_modules.evaluation import RetrievalEvaluator

# 项目根目录（脚本所在目录）
PROJECT_ROOT = Path(__file__).parent.parent.parent
EVAL_FILE = PROJECT_ROOT / "data" / "eval" / "retrieval_eval.jsonl"
REPORTS_DIR = PROJECT_ROOT / "reports"


def build_retrieval_system():
    """构建检索系统：data → index → retrieval，不碰 LLM"""
    data_path = PROJECT_ROOT / DEFAULT_CONFIG.data_path
    print(f"[1/4] 加载文档: {data_path}")
    data_module = DataPreparationModule(str(data_path))
    data_module.load_documents()

    print("[2/4] 文本分块...")
    chunks = data_module.chunk_documents()
    stats = data_module.get_statistics()
    print(f"   文档 {stats['total_documents']} 个, 分块 {stats['total_chunks']} 个")

    print("[3/4] 构建向量索引（每次重建，规避 FAISS save/load docstore 错位问题）...")
    index_module = IndexConstructionModule(
        model_name=DEFAULT_CONFIG.embedding_model,
        index_save_path=str(PROJECT_ROOT / DEFAULT_CONFIG.index_save_path),
    )
    # 注意：FAISS load_local 在当前环境会令 docstore 与向量错位（检索结果错乱），
    # 故评测时始终重建索引，保证正确性。生产系统见 main.py，需另行排查 save/load。
    vectorstore = index_module.build_vector_index(chunks)

    print("[4/4] 初始化检索模块...")
    retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)
    # 挂载 data_module，供父子文档召回评测使用
    retrieval_module.data_module = data_module
    return data_module, retrieval_module


def load_eval_samples(path: Path) -> list:
    """加载评测样本，兼容单行 jsonl 与多行 pretty-print JSON 两种格式"""
    with open(path, encoding="utf-8") as f:
        content = f.read()
    samples = []
    decoder = json.JSONDecoder()
    idx, n = 0, len(content)
    while idx < n:
        while idx < n and content[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = decoder.raw_decode(content, idx)
        samples.append(obj)
        idx = end
    return samples


def main():
    parser = argparse.ArgumentParser(description="检索评测")
    parser.add_argument("--routes", nargs="+", default=["vector", "bm25", "hybrid"],
                        choices=["vector", "bm25", "hybrid"])
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--eval-file", default=str(EVAL_FILE))
    args = parser.parse_args()

    print("=" * 64)
    print("菜谱 RAG 检索评测")
    print("=" * 64)

    # 构建
    data_module, retrieval_module = build_retrieval_system()
    dishes_root = PROJECT_ROOT / DEFAULT_CONFIG.data_path / "dishes"

    # 加载评测集
    eval_path = Path(args.eval_file)
    print(f"\n加载评测集: {eval_path}")
    samples = load_eval_samples(eval_path)
    print(f"  共 {len(samples)} 条")

    # 评测器
    evaluator = RetrievalEvaluator(retrieval_module, dishes_root)

    # 主评测：三路 × 多 k
    print(f"\n开始评测 routes={args.routes} k={args.k}")
    t0 = time.time()
    report = evaluator.run(samples, args.routes, args.k)
    elapsed = time.time() - t0
    report["meta"]["elapsed_seconds"] = round(elapsed, 2)

    # 专项评测
    print("\n专项评测：父子文档召回...")
    report["parent_doc_recall"] = evaluator.eval_parent_doc_recall(samples, k=3)

    print("专项评测：元数据过滤准确率...")
    report["metadata_filter"] = evaluator.eval_metadata_filter_accuracy(samples, k=5)

    # 输出报告
    print("\n" + RetrievalEvaluator.format_report(report))

    print("-" * 64)
    print("【专项评测】")
    pdr = report["parent_doc_recall"]
    print(f"  父子文档召回 HitRate@3: {pdr['parent_hit_rate']:.4f} "
          f"({pdr['total_detail_samples']} 条 detail 样本)")
    mf = report["metadata_filter"]
    print(f"  元数据过滤准确率: {mf['filter_accuracy']:.4f} "
          f"({mf['passed_queries']}/{mf['checked_queries']} 条通过, "
          f"平均返回 {mf['avg_returned']:.1f} 个)")
    print(f"  耗时: {elapsed:.1f}s")

    # 保存 JSON 报告
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "retrieval_eval_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {out_path}")


if __name__ == "__main__":
    main()
