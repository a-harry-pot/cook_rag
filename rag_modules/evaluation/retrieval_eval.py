"""
检索评测模块
============

计算指标：Recall@k / Precision@k / MRR / NDCG@k / HitRate@k
支持三路检索对比：vector / bm25 / hybrid(RRF)
专项评测：父子文档召回率、元数据过滤准确率、top_k 敏感度

依赖现有 RetrievalOptimizationModule 的 vectorstore 与 chunks，
不依赖 LLM / API Key，可独立运行。

doc_id 匹配约定：
    检索返回的子块 Document，其 metadata['source'] 为绝对路径。
    反算相对 dishes 根目录的 posix 路径，与评测集 relevant_doc_ids 对齐。
"""

import json
import math
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

from langchain_core.documents import Document


# ============================================================
# 工具函数
# ============================================================
def doc_to_id(doc: Document, dishes_root: Path) -> str:
    """
    从检索子块反算 doc_id（相对 dishes 根目录的 posix 路径）。
    与 build_eval_dataset.py 中 doc_id 生成方式一致。

    兼容两种 source 格式：
      - 绝对路径：D:\\...\\dishes\\aquatic\\水煮鱼.md
      - 相对路径：data/cook/dishes/aquatic/水煮鱼.md（旧索引可能如此）
    """
    source = doc.metadata.get("source", "")
    p = Path(source)
    # 1) 绝对路径反算
    try:
        return p.resolve().relative_to(Path(dishes_root).resolve()).as_posix()
    except (ValueError, OSError):
        pass
    # 2) 从路径 parts 中找 'dishes' 之后的部分（兼容相对路径）
    parts = p.parts
    if "dishes" in parts:
        idx = parts.index("dishes")
        return "/".join(parts[idx + 1:])
    # 3) 兜底：dish_name（子块从父文档继承了该字段）
    return doc.metadata.get("dish_name", "")


# ============================================================
# 指标计算器
# ============================================================
class MetricsCalculator:
    """检索质量指标计算（全部为静态方法，二值相关性）"""

    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """Recall@k：前 k 个结果中命中的相关文档占比"""
        if not relevant_ids:
            return 0.0
        rel_set = set(relevant_ids)
        hits = sum(1 for rid in retrieved_ids[:k] if rid in rel_set)
        return hits / len(relevant_ids)

    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """Precision@k：前 k 个结果中相关的比例"""
        if k <= 0:
            return 0.0
        rel_set = set(relevant_ids)
        hits = sum(1 for rid in retrieved_ids[:k] if rid in rel_set)
        return hits / k

    @staticmethod
    def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """MRR：第一个相关结果的倒数排名"""
        rel_set = set(relevant_ids)
        for i, rid in enumerate(retrieved_ids):
            if rid in rel_set:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """NDCG@k：考虑排序位置的相关性增益（二值相关性）"""
        rel_set = set(relevant_ids)
        # DCG
        gains = [1.0 if rid in rel_set else 0.0 for rid in retrieved_ids[:k]]
        dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
        # IDCG：理想排序（相关文档全排前面）
        ideal_hits = min(len(relevant_ids), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def hit_rate_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """HitRate@k：前 k 个是否至少命中一个相关文档（0/1）"""
        rel_set = set(relevant_ids)
        return 1.0 if any(rid in rel_set for rid in retrieved_ids[:k]) else 0.0

    @staticmethod
    def all_metrics(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> Dict[str, float]:
        """一次性返回该 k 下的全部指标"""
        return {
            f"recall@{k}": MetricsCalculator.recall_at_k(retrieved_ids, relevant_ids, k),
            f"precision@{k}": MetricsCalculator.precision_at_k(retrieved_ids, relevant_ids, k),
            f"ndcg@{k}": MetricsCalculator.ndcg_at_k(retrieved_ids, relevant_ids, k),
            f"hit_rate@{k}": MetricsCalculator.hit_rate_at_k(retrieved_ids, relevant_ids, k),
            "mrr": MetricsCalculator.mrr(retrieved_ids, relevant_ids),
        }


# ============================================================
# 检索评测器
# ============================================================
class RetrievalEvaluator:
    """
    检索评测器：对现有检索系统跑评测集，输出三路检索对比报告。

    Args:
        retrieval_module: 已初始化的 RetrievalOptimizationModule
        dishes_root: data/cook/dishes 绝对路径，用于 doc_id 反算
        pool_size: 单路检索候选池大小（hybrid 的 RRF 在此范围内合并）
    """

    def __init__(self, retrieval_module, dishes_root, pool_size: int = 50):
        self.retrieval = retrieval_module
        self.dishes_root = Path(dishes_root)
        self.pool_size = pool_size

        # 从检索模块取出向量库与分块
        self.vectorstore = retrieval_module.vectorstore
        self.chunks = retrieval_module.chunks

        # 构建一个大 k 的 BM25 检索器（复用，避免重复构建）
        # 使用与生产系统一致的 jieba 中文分词
        from langchain_community.retrievers import BM25Retriever
        from ..retrieval_optimization import _tokenize_zh
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks, k=pool_size, preprocess_func=_tokenize_zh
        )

    # ---------- 三路检索 ----------
    def _vector_search(self, query: str, k: int) -> List[Document]:
        """向量检索，返回前 k 个"""
        docs = self.vectorstore.similarity_search(query, k=max(k, self.pool_size))
        return docs[:k]

    def _bm25_search(self, query: str, k: int) -> List[Document]:
        """BM25 检索，返回前 k 个"""
        docs = self.bm25_retriever.invoke(query)
        return docs[:k]

    def _hybrid_search(self, query: str, k: int) -> List[Document]:
        """
        混合检索：RRF 合并 vector 与 bm25 候选，取前 k。
        RRF 公式与 retrieval_optimization.py 一致：score = 1/(60+rank+1)
        """
        pool = max(k, self.pool_size)
        vector_docs = self._vector_search(query, pool)
        bm25_docs = self._bm25_search(query, pool)
        return self._rrf_merge(vector_docs, bm25_docs, k)

    def _rrf_merge(self, vector_docs: List[Document], bm25_docs: List[Document],
                   top_k: int, rrf_k: int = 60) -> List[Document]:
        """RRF 合并两路结果（与 RetrievalOptimizationModule._rrf_rerank 逻辑一致）"""
        scores: Dict[str, float] = {}
        objects: Dict[str, Document] = {}

        for rank, doc in enumerate(vector_docs):
            did = self._doc_key(doc)
            objects[did] = doc
            scores[did] = scores.get(did, 0) + 1.0 / (rrf_k + rank + 1)

        for rank, doc in enumerate(bm25_docs):
            did = self._doc_key(doc)
            objects[did] = doc
            scores[did] = scores.get(did, 0) + 1.0 / (rrf_k + rank + 1)

        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [objects[did] for did, _ in ordered[:top_k]]

    def _doc_key(self, doc: Document) -> str:
        """
        文档唯一键：优先用 doc_id（路径），路径相同则用内容哈希区分不同子块。
        这样同一菜品的不同子块不会误合并，但同一子块在两路中会合并。
        """
        did = doc_to_id(doc, self.dishes_root)
        content_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()[:8]
        return f"{did}#{content_hash}"

    def _docs_to_ids(self, docs: List[Document]) -> List[str]:
        """把检索子块列表转为 doc_id 列表（去重保留顺序）"""
        seen, ids = set(), []
        for doc in docs:
            did = doc_to_id(doc, self.dishes_root)
            if did and did not in seen:
                seen.add(did)
                ids.append(did)
        return ids

    # ---------- 评测主流程 ----------
    def search(self, query: str, route: str, k: int) -> List[Document]:
        """统一检索入口"""
        if route == "vector":
            return self._vector_search(query, k)
        elif route == "bm25":
            return self._bm25_search(query, k)
        elif route == "hybrid":
            return self._hybrid_search(query, k)
        raise ValueError(f"未知 route: {route}")

    def run(self, eval_samples: List[Dict], routes: List[str],
            k_values: List[int]) -> Dict[str, Any]:
        """
        跑完整检索评测集。

        Returns:
            {
                "by_route": {route: {k: {metric: mean}, ...}, ...},
                "by_route_type": {route_type: {route: {k: {metric: mean}}}},
                "per_query": [...],  # 每条 query 的明细
            }
        """
        # 累加器：by_route[route][metric_key] = [values]
        by_route = defaultdict(lambda: defaultdict(list))
        # by_route_type[qtype][route][metric_key] = [values]
        by_type = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        per_query = []

        total = len(eval_samples)
        for i, sample in enumerate(eval_samples, 1):
            query = sample["query"]
            relevant = sample["relevant_doc_ids"]
            qtype = sample.get("query_type", "unknown")

            for route in routes:
                for k in k_values:
                    docs = self.search(query, route, k)
                    retrieved = self._docs_to_ids(docs)
                    m = MetricsCalculator.all_metrics(retrieved, relevant, k)
                    for metric, val in m.items():
                        by_route[route][metric].append(val)
                        by_type[qtype][route][metric].append(val)

                # 记录首条 query 的明细（便于排查）
            if i <= 5 or i % 50 == 0:
                print(f"  进度 {i}/{total}: {query[:30]:<30} ({qtype})")

            per_query.append({
                "id": sample.get("id"),
                "query": query,
                "query_type": qtype,
                "relevant": relevant,
            })

        # 聚合均值
        report = {
            "by_route": self._aggregate(by_route),
            "by_route_type": {
                qt: self._aggregate(rt) for qt, rt in by_type.items()
            },
            "meta": {
                "total_samples": total,
                "routes": routes,
                "k_values": k_values,
                "pool_size": self.pool_size,
            },
        }
        return report

    @staticmethod
    def _aggregate(nested: defaultdict) -> Dict[str, Dict[str, float]]:
        """对累加器求均值"""
        out = {}
        for route, metrics in nested.items():
            out[route] = {m: (sum(v) / len(v) if v else 0.0) for m, v in metrics.items()}
        return out

    # ---------- 专项：父子文档召回 ----------
    def eval_parent_doc_recall(self, eval_samples: List[Dict], k: int = 3) -> Dict[str, Any]:
        """
        父子文档召回评测：
        检索命中子块后，get_parent_documents 是否取回正确父文档。
        现有系统最终用父文档喂给生成，故父文档召回比子块召回更关键。
        """
        hits, total = 0, 0
        for sample in eval_samples:
            if sample.get("query_type") != "detail":
                continue  # 仅对 detail 类评测（list 类 relevant 是多文档集合）
            total += 1
            relevant = set(sample["relevant_doc_ids"])
            docs = self._hybrid_search(sample["query"], k)
            parents = self.retrieval.data_module.get_parent_documents(docs) \
                if hasattr(self.retrieval, "data_module") and self.retrieval.data_module else []
            parent_ids = {doc_to_id(p, self.dishes_root) for p in parents}
            if parent_ids & relevant:
                hits += 1
        return {
            "parent_hit_rate": hits / total if total else 0.0,
            "total_detail_samples": total,
            "k": k,
        }

    # ---------- 专项：元数据过滤准确率 ----------
    def eval_metadata_filter_accuracy(self, eval_samples: List[Dict], k: int = 5) -> Dict[str, Any]:
        """
        元数据过滤准确率：
        对带 filters 的样本，metadata_filtered_search 返回结果是否都满足过滤条件。
        """
        checked, passed, total_returned = 0, 0, 0
        for sample in eval_samples:
            filters = sample.get("filters")
            if not filters:
                continue
            checked += 1
            docs = self.retrieval.metadata_filtered_search(
                sample["query"], filters, top_k=k
            )
            total_returned += len(docs)
            ok = True
            for doc in docs:
                for key, val in filters.items():
                    if doc.metadata.get(key) != val:
                        ok = False
                        break
                if not ok:
                    break
            if ok and docs:
                passed += 1
        return {
            "filter_accuracy": passed / checked if checked else 0.0,
            "checked_queries": checked,
            "passed_queries": passed,
            "avg_returned": total_returned / checked if checked else 0.0,
            "k": k,
        }

    # ---------- 报告格式化 ----------
    @staticmethod
    def format_report(report: Dict[str, Any]) -> str:
        """格式化报告为可读文本"""
        lines = []
        meta = report["meta"]
        lines.append("=" * 64)
        lines.append("检索评测报告")
        lines.append("=" * 64)
        lines.append(f"样本数: {meta['total_samples']}  routes: {meta['routes']}  k: {meta['k_values']}")
        lines.append("")

        # 按路由聚合
        lines.append("-" * 64)
        lines.append("【按检索路由聚合】")
        lines.append("-" * 64)
        header = f"{'route':<8}" + "".join(f"{m:<16}" for m in _metric_cols(meta["k_values"]))
        lines.append(header)
        for route, metrics in report["by_route"].items():
            row = f"{route:<8}"
            for m in _metric_cols(meta["k_values"]):
                row += f"{metrics.get(m, 0):.4f}{'':<10}"
            lines.append(row)
        lines.append("")

        # 按路由类型
        lines.append("-" * 64)
        lines.append("【按 query_type 分组】")
        lines.append("-" * 64)
        for qtype, rt_data in report["by_route_type"].items():
            lines.append(f"\n■ {qtype}")
            for route, metrics in rt_data.items():
                row = f"  {route:<8}"
                for m in _metric_cols(meta["k_values"]):
                    row += f"{metrics.get(m, 0):.4f}{'':<10}"
                lines.append(row)
        lines.append("")
        return "\n".join(lines)


def _metric_cols(k_values: List[int]) -> List[str]:
    """生成报告列名"""
    cols = []
    for k in k_values:
        cols.extend([f"recall@{k}", f"precision@{k}", f"ndcg@{k}", f"hit_rate@{k}"])
    cols.append("mrr")
    return cols
