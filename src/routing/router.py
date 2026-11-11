"""
Module 4 - Query understanding & query-adaptive retrieval router.
"""
import os
import warnings
from typing import Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from src.routing.classifier import QueryClassification, classify_query
from src.routing.decomposer import DECOMPOSE_SYSTEM, heuristic_decompose
from src.routing.patient_context import HistoryExtractor, PatientContext, deidentify, separate_history
from src.routing.schemas import MODULES, QueryDecomposition, RoutedQuery, RoutingScores, SubQueryRoute
from src.routing.scorer import SCORE_SYSTEM, heuristic_scores


class QueryRouter:
    """Decompose a master query and decide, per sub-query, which retrieval modules to activate."""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        model_name: str = "gpt-4o-mini",
        tau: float = 0.5,
        fallback_to_best: bool = True,
        extractor: Optional[HistoryExtractor] = None,
    ) -> None:
        """
        tau              : activation threshold; a source is used when its score >= tau.
        fallback_to_best : if no source reaches tau, activate the highest-scoring one so the
                           pipeline never has zero evidence. Set False to allow an empty selection.
        """
        self.tau = tau
        self.fallback_to_best = fallback_to_best
        if llm is None and os.getenv("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=model_name, temperature=0)
        self.llm = llm
        self.extractor = extractor or HistoryExtractor(backend="auto", llm=llm)

        if llm is not None:
            decompose_prompt = ChatPromptTemplate.from_messages([("system", DECOMPOSE_SYSTEM), ("human", "{query}")])
            score_prompt = ChatPromptTemplate.from_messages([("system", SCORE_SYSTEM), ("human", "{query}")])
            self._decompose_chain = decompose_prompt | llm.with_structured_output(QueryDecomposition, method="function_calling")
            self._score_chain = score_prompt | llm.with_structured_output(RoutingScores, method="function_calling")
        else:
            warnings.warn("QueryRouter: no LLM / OPENAI_API_KEY - using the offline rule-based backend.")

    def decompose(self, master_query: str) -> QueryDecomposition:
        result = (self._decompose_chain.invoke({"query": master_query})
                  if self.llm is not None else heuristic_decompose(master_query))
        if not result.is_complex or not result.sub_queries:
            result.sub_queries = [master_query]
        return result

    def score(self, sub_queries: List[str]) -> List[RoutingScores]:
        if self.llm is not None:
            return self._score_chain.batch([{"query": q} for q in sub_queries])
        return [heuristic_scores(q) for q in sub_queries]

    def activated_modules(self, scores: RoutingScores) -> List[str]:
        """Modules whose score >= tau (optionally falling back to the single best module)."""
        values: Dict[str, float] = scores.model_dump()
        active = [MODULES[k] for k, v in values.items() if v >= self.tau]
        if not active and self.fallback_to_best:
            active = [MODULES[max(values, key=values.get)]]
        return active

    def classify(self, query: str, is_complex: bool = False) -> QueryClassification:
        """Classify the clinical intent of the query."""
        return classify_query(query, is_complex=is_complex)

    def route(
        self,
        master_query: str,
        context: Optional[PatientContext] = None,
        auto_split: bool = True,
    ) -> RoutedQuery:
        """Decompose + score the QUESTION, then activate sources.

        context    : patient history, kept out of decomposition and scoring.
        auto_split : if no context is given and the text mixes history sentences with a question, split them.
        """
        if context is None and auto_split:
            history, question = separate_history(master_query)
            if history:
                context, master_query = PatientContext.from_text(history, self.extractor), question
        if self.llm is not None:
            master_query = deidentify(master_query)

        decomposition = self.decompose(master_query)
        scores = self.score(decomposition.sub_queries)
        has_ctx = context is not None and not context.is_empty()
        routes = []
        for q, s in zip(decomposition.sub_queries, scores):
            active = self.activated_modules(s)
            enriched = context.enrich(q) if has_ctx and "Vector" in active else q
            routes.append(SubQueryRoute(sub_query=q, scores=s, activated=active, retrieval_query=enriched))
        return RoutedQuery(
            master_query=master_query,
            decomposition=decomposition,
            routes=routes,
            patient_context=context if has_ctx else None,
        )
