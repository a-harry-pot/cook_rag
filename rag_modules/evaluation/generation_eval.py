"""
生成评测模块
============

P1 生成评测：评估 RAG 端到端生成质量 + 查询路由准确率。

两部分：
  1. 生成质量评测（LLM-as-Judge，多维度打分）
     - faithfulness 忠实度：回答是否仅基于检索内容（防幻觉）
     - answer_relevancy 答案相关性：回答是否切题
     - ingredient_completeness 食材完整性：必备原料是否齐全
     - quantity_accuracy 用量准确性：用量是否与参考一致
     - step_executability 步骤可执行性：步骤是否具体可操作
     - nutrition_fidelity 营养保真：营养数据是否正确
  2. 路由准确率：query_router 分类与标准答案一致率

不依赖 ragas，自实现 LLM-as-Judge（用现有 DeepSeek 作为评判）。
"""

import json
import re
import logging
from typing import List, Dict, Any, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


# ============================================================
# 评判 Prompt
# ============================================================
JUDGE_PROMPT = """你是菜谱问答系统的质量评测员。请根据以下信息对系统生成的回答进行多维度打分。

【用户问题】
{query}

【检索到的菜谱信息】
{context}

【系统生成的回答】
{answer}

【参考答案（来自菜谱原文）】
{reference}

请按以下 6 个维度打分，每项 0-5 分（0=极差，5=完美），并给出一句理由：

1. faithfulness 忠实度：回答中的事实是否都能在"检索到的菜谱信息"中找到依据，有无编造食材/用量/步骤
2. answer_relevancy 答案相关性：回答是否切题，是否直接回答了用户问题
3. ingredient_completeness 食材完整性：参考答案的"必备原料"在回答中是否齐全列出
4. quantity_accuracy 用量准确性：回答中的用量（克数/比例）是否与参考答案的 key_facts 一致
5. step_executability 步骤可执行性：步骤是否包含具体动作+火候+时间，能否照做
6. nutrition_fidelity 营养保真：营养成分数值是否与参考答案一致（若回答未涉及营养则给中性分3）

请严格只输出一个 JSON 对象，不要输出任何其它内容或代码块标记，格式如下：
{{"faithfulness": 分数, "answer_relevancy": 分数, "ingredient_completeness": 分数, "quantity_accuracy": 分数, "step_executability": 分数, "nutrition_fidelity": 分数, "reason": "简短说明"}}"""


# ============================================================
# 生成评测器
# ============================================================
class GenerationEvaluator:
    """
    生成评测器：跑完整 RAG 生成流程，用 LLM-as-Judge 评判回答质量。

    Args:
        retrieval_module: RetrievalOptimizationModule（已初始化）
        generation_module: GenerationIntegrationModule（已初始化，含 LLM）
        data_module: DataPreparationModule（用于 get_parent_documents）
        dishes_root: data/cook/dishes 绝对路径
        top_k: 检索返回数
    """

    DIMENSIONS = [
        "faithfulness", "answer_relevancy", "ingredient_completeness",
        "quantity_accuracy", "step_executability", "nutrition_fidelity",
    ]

    def __init__(self, retrieval_module, generation_module, data_module,
                 dishes_root, top_k: int = 3):
        self.retrieval = retrieval_module
        self.gen = generation_module
        self.data = data_module
        self.dishes_root = dishes_root
        self.top_k = top_k
        self.judge_llm = generation_module.llm  # 复用 LLM 做 judge

    # ---------- 复刻 RAG 生成流程 ----------
    def generate_with_context(self, query: str) -> Tuple[str, List[Document], str]:
        """
        复刻 main.ask_question 流程，返回 (回答, 上下文父文档, 路由类型)。
        保留上下文供评判使用。
        """
        # 1. 查询路由
        route = self.gen.query_router(query)

        # 2. 查询重写（list 类保持原样）
        if route == "list":
            rewritten = query
        else:
            rewritten = self.gen.query_rewrite(query)

        # 3. 检索（带元数据过滤）
        filters = self._extract_filters(query)
        if filters:
            docs = self.retrieval.metadata_filtered_search(rewritten, filters, top_k=self.top_k)
        else:
            docs = self.retrieval.hybrid_search(rewritten, top_k=self.top_k)

        # 4. 无结果
        if not docs:
            return "抱歉，没有找到相关的食谱信息。", [], route

        # 5. 取父文档并按路由生成
        parents = self.data.get_parent_documents(docs)
        if route == "list":
            answer = self.gen.generate_list_answer(query, parents)
        elif route == "detail":
            answer = self.gen.generate_step_by_step_answer(query, parents)
        else:
            answer = self.gen.generate_basic_answer(query, parents)
        return answer, parents, route

    def _extract_filters(self, query: str) -> Dict[str, str]:
        """复刻 main._extract_filters_from_query"""
        from rag_modules.data_preparation import DataPreparationModule
        filters = {}
        for cat in DataPreparationModule.get_supported_categories():
            if cat in query:
                filters["category"] = cat
                break
        for diff in sorted(DataPreparationModule.get_supported_difficulties(), key=len, reverse=True):
            if diff in query:
                filters["difficulty"] = diff
                break
        return filters

    # ---------- LLM-as-Judge ----------
    def judge(self, query: str, answer: str, context_docs: List[Document],
              reference: Dict) -> Dict[str, Any]:
        """用 LLM 对回答打分，返回各维度分数 + 理由"""
        context_text = self.gen._build_context(context_docs)
        ref_text = json.dumps(reference, ensure_ascii=False, indent=2)

        prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT)
        chain = prompt | self.judge_llm | StrOutputParser()

        try:
            resp = chain.invoke({
                "query": query,
                "context": context_text,
                "answer": answer,
                "reference": ref_text,
            })
            scores = self._parse_judge_json(resp)
        except Exception as e:
            logger.warning(f"评判失败: {e}")
            scores = {d: 0 for d in self.DIMENSIONS}
            scores["reason"] = f"评判异常: {e}"
        return scores

    @staticmethod
    def _parse_judge_json(text: str) -> Dict[str, Any]:
        """解析 judge 返回的 JSON（兼容代码块包裹）"""
        # 去掉可能的 markdown 代码块
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        # 提取第一个 {...}
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)
        data = json.loads(text)
        # 确保维度为数值
        out = {}
        for d in GenerationEvaluator.DIMENSIONS:
            v = data.get(d)
            try:
                out[d] = float(v)
            except (TypeError, ValueError):
                out[d] = 0.0
        out["reason"] = data.get("reason", "")
        return out

    # ---------- 生成评测主流程 ----------
    def run(self, samples: List[Dict]) -> Dict[str, Any]:
        """对生成评测集跑生成 + 评判"""
        results = []
        dim_sums = {d: [] for d in self.DIMENSIONS}
        route_match = 0

        total = len(samples)
        for i, s in enumerate(samples, 1):
            qid = s.get("id")
            query = s["query"]
            reference = s.get("reference_answer", {})
            expected_route = s.get("query_type", "detail")

            print(f"  [{i}/{total}] {query[:30]:<30} ", end="", flush=True)

            # 生成
            answer, context, route = self.generate_with_context(query)
            if route == expected_route:
                route_match += 1

            # 评判
            scores = self.judge(query, answer, context, reference)
            print(f"route={route} 平均分={sum(scores[d] for d in self.DIMENSIONS)/6:.1f}")

            for d in self.DIMENSIONS:
                dim_sums[d].append(scores[d])

            results.append({
                "id": qid,
                "query": query,
                "expected_route": expected_route,
                "actual_route": route,
                "answer_preview": answer[:200],
                "scores": {d: scores[d] for d in self.DIMENSIONS},
                "reason": scores.get("reason", ""),
            })

        # 聚合
        summary = {d: (sum(v) / len(v) if v else 0) for d, v in dim_sums.items()}
        overall = sum(summary.values()) / len(summary) if summary else 0
        return {
            "summary": summary,
            "overall_avg": overall,
            "route_match_rate": route_match / total if total else 0,
            "total": total,
            "details": results,
        }

    # ---------- 路由准确率 ----------
    def eval_router_accuracy(self, samples: List[Dict]) -> Dict[str, Any]:
        """评测 query_router 分类准确率"""
        correct = 0
        by_type: Dict[str, Dict[str, int]] = {}
        errors = []

        total = len(samples)
        for i, s in enumerate(samples, 1):
            query = s["query"]
            expected = s.get("query_type", "general")
            predicted = self.gen.query_router(query)

            by_type.setdefault(expected, {"correct": 0, "total": 0})
            by_type[expected]["total"] += 1
            if predicted == expected:
                correct += 1
                by_type[expected]["correct"] += 1
            else:
                errors.append({
                    "query": query, "expected": expected, "predicted": predicted,
                })

            if i % 50 == 0:
                print(f"  路由评测进度 {i}/{total}")

        per_type = {
            t: {"accuracy": v["correct"] / v["total"] if v["total"] else 0,
                "correct": v["correct"], "total": v["total"]}
            for t, v in by_type.items()
        }
        return {
            "accuracy": correct / total if total else 0,
            "correct": correct,
            "total": total,
            "per_type": per_type,
            "errors": errors[:20],  # 仅保留前 20 条错误样本
        }

    # ---------- 报告格式化 ----------
    @staticmethod
    def format_report(report: Dict, router_report: Dict) -> str:
        lines = []
        lines.append("=" * 64)
        lines.append("生成评测报告（LLM-as-Judge）")
        lines.append("=" * 64)
        lines.append(f"生成评测样本数: {report['total']}")
        lines.append("")

        lines.append("-" * 64)
        lines.append("【生成质量各维度均分（0-5）】")
        lines.append("-" * 64)
        for d in GenerationEvaluator.DIMENSIONS:
            lines.append(f"  {d:<26} {report['summary'][d]:.2f}")
        lines.append(f"  {'总体平均':<24} {report['overall_avg']:.2f}")
        lines.append(f"  路由匹配率(生成集)      {report['route_match_rate']:.2%}")
        lines.append("")

        lines.append("-" * 64)
        lines.append("【查询路由准确率】")
        lines.append("-" * 64)
        lines.append(f"  总体准确率: {router_report['accuracy']:.2%} "
                     f"({router_report['correct']}/{router_report['total']})")
        lines.append("  分类型:")
        for t, v in router_report["per_type"].items():
            lines.append(f"    {t:<10} {v['accuracy']:.2%} ({v['correct']}/{v['total']})")
        if router_report["errors"]:
            lines.append(f"  错误样本(前{len(router_report['errors'])}条):")
            for e in router_report["errors"]:
                lines.append(f"    [{e['expected']}→{e['predicted']}] {e['query']}")
        lines.append("")
        return "\n".join(lines)
