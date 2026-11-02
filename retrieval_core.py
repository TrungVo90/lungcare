from dataclasses import dataclass
import re
from typing import Any, Callable, Protocol, Sequence

import numpy as np


TOPIC_PATTERNS: dict[str, tuple[str, ...]] = {
    "mass": (r"\bmass(?:es)?\b", r"\btumou?rs?\b"),
    "nodule": (r"\bnodul(?:e|es|ar)\b",),
    "consolidation": (r"\bconsolidation\b", r"\bairspace opacity\b", r"\bopacity\b"),
    "effusion": (r"\beffusion\b", r"\bfluid\b"),
    "cardiomegaly": (r"\bcardiomegaly\b", r"\benlarged heart\b", r"\bheart silhouette\b"),
    "pneumothorax": (r"\bpneumothorax\b", r"\bcollapsed lung\b"),
    "fibrosis": (r"\bfibrosis\b", r"\bscarring\b"),
    "pleural_thickening": (r"\bpleural thickening\b",),
    "emphysema": (r"\bemphysema\b", r"\bhyperinflation\b"),
    "atelectasis": (r"\batelectasis\b", r"\bunder-expanded\b"),
    "fracture": (r"\bfracture\b", r"\bbroken bone\b", r"\brib injury\b"),
    "calcification": (r"\bcalcification\b", r"\bcalcified\b"),
}

LABEL_TO_TOPIC = {
    "Mass": "mass", "Nodule": "nodule", "Diffuse Nodule": "nodule",
    "Consolidation": "consolidation", "Effusion": "effusion",
    "Cardiomegaly": "cardiomegaly", "Pneumothorax": "pneumothorax",
    "Fibrosis": "fibrosis", "Pleural Thickening": "pleural_thickening",
    "Emphysema": "emphysema", "Atelectasis": "atelectasis",
    "Fracture": "fracture", "Calcification": "calcification",
}

RELATED_TOPICS = {"mass": ("nodule",), "nodule": ("mass",)}

TOPIC_QUERIES = {
    "mass": "chest radiograph pulmonary mass diagnostic evaluation",
    "nodule": "chest radiograph pulmonary nodule evaluation",
    "consolidation": "chest radiograph pulmonary consolidation evaluation",
    "effusion": "chest radiograph pleural effusion evaluation",
    "cardiomegaly": "chest radiograph enlarged cardiac silhouette evaluation",
    "pneumothorax": "chest radiograph pneumothorax evaluation",
    "fibrosis": "chest radiograph pulmonary fibrosis evaluation",
    "pleural_thickening": "chest radiograph pleural thickening evaluation",
    "emphysema": "chest radiograph pulmonary emphysema evaluation",
    "atelectasis": "chest radiograph pulmonary atelectasis evaluation",
    "fracture": "chest radiograph rib fracture evaluation",
    "calcification": "chest radiograph pulmonary calcification evaluation",
}

GENERAL_IMAGE = re.compile(
    r"\b(?:x[ -]?ray|radiograph|image|scan|region|area|highlight|finding|abnormal)\b", re.I
)
UNSUPPORTED_CONDITION = re.compile(
    r"\b(?:pneumonia|infection|tuberculosis|covid(?:-19)?|asthma|bronchitis|"
    r"cancer|carcinoma|malignan(?:t|cy)|sarcoma|lymphoma|metasta(?:sis|ses|tic))\b",
    re.I,
)
THORACIC = re.compile(
    r"\b(?:lung|pulmonary|chest|thoracic|pleur(?:a|al)|bronch(?:us|ial)?|"
    r"mediastin(?:al|um)|respiratory|radiograph|x-ray)\b", re.I
)
CXR = re.compile(r"\b(?:chest[ -](?:x[ -]?ray|radiograph(?:y|ic)?s?)|cxr)\b", re.I)
CT_ONLY = re.compile(r"\b(?:low-dose ct|computed tomography|ct screening|lung-rads)\b", re.I)
CASE_REPORT = re.compile(r"\b(?:case report|a case of|we present|we report|rare|unusual)\b", re.I)
GENERAL_EVIDENCE = re.compile(
    r"\b(?:review|guideline|algorithmic|approach|assessment|evaluation|diagnos(?:is|tic)|"
    r"management|performance|accuracy|systematic|meta-analysis|cohort|study)\b", re.I
)
DISEASE_SPECIFIC = re.compile(
    r"\b(?:cancer|carcinoma|malignan(?:t|cy)|tumou?rs?|sarcoma|lymphoma|metasta(?:sis|ses|tic))\b",
    re.I,
)

STOPWORDS = {
    "and", "caregiver", "chest", "does", "explain", "image", "patient",
    "please", "question", "radiograph", "show", "the", "what", "with", "xray",
}


class BM25Index(Protocol):
    def get_scores(self, query_tokens: Sequence[str]) -> np.ndarray: ...


class DenseIndex(Protocol):
    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class RetrievalPlan:
    query: str
    question_topics: list[str]
    selected_image_labels: list[str]
    scope: str


def infer_question_topics(question: str) -> list[str]:
    """Find only topics explicitly named by the caregiver."""
    normalized = re.sub(r"\s+", " ", question.strip())
    return [
        topic for topic, patterns in TOPIC_PATTERNS.items()
        if any(re.search(pattern, normalized, re.I) for pattern in patterns)
    ]


def classify_question(question: str) -> str:
    if infer_question_topics(question):
        return "specific"
    if UNSUPPORTED_CONDITION.search(question):
        return "unsupported"
    if GENERAL_IMAGE.search(question):
        return "general_image"
    return "unsupported"


def build_retrieval_plan(question: str, image_labels: Sequence[str]) -> RetrievalPlan:
    """Build a question-first query; image cues never replace question intent."""
    normalized = re.sub(r"\s+", " ", question.strip()).rstrip(".!?")
    topics = infer_question_topics(normalized)
    scope = classify_question(normalized)
    available = [(label, LABEL_TO_TOPIC.get(label)) for label in image_labels]
    available_topics = [topic for _, topic in available if topic]

    selected_topics: list[str] = []
    if topics:
        for topic in topics:
            if topic in available_topics:
                selected_topics.append(topic)
                continue
            related = next((x for x in RELATED_TOPICS.get(topic, ()) if x in available_topics), None)
            if related:
                selected_topics.append(related)
        retrieval_topics = topics
    elif scope == "general_image" and available_topics:
        selected_topics = available_topics[:1]
        retrieval_topics = selected_topics[:]
    else:
        retrieval_topics = []

    selected_labels = [label for label, topic in available if topic in selected_topics]
    query = f"Chest X-ray caregiver question: {normalized}"
    if retrieval_topics:
        query += ". Question topics: " + "; ".join(TOPIC_QUERIES[t] for t in retrieval_topics)
    if selected_topics:
        query += ". Image-review cues: " + "; ".join(TOPIC_QUERIES[t] for t in selected_topics)
    return RetrievalPlan(query, retrieval_topics, selected_labels, scope)


def _topic_matches(text: str, topic: str) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in TOPIC_PATTERNS.get(topic, ()))


def relevance_guard(title: str, abstract: str, topics: Sequence[str], allow_disease_specific: bool) -> dict[str, Any]:
    title = re.sub(r"\s+", " ", title).strip()
    body = re.sub(r"\s+", " ", f"{title} {abstract}").strip()
    title_topics = [topic for topic in topics if _topic_matches(title, topic)]
    body_topics = [topic for topic in topics if _topic_matches(body, topic)]
    passes = (
        bool(THORACIC.search(body))
        and bool(CXR.search(body))
        and bool(title_topics)
        and bool(GENERAL_EVIDENCE.search(title))
        and not bool(CT_ONLY.search(title))
        and not bool(CASE_REPORT.search(title))
        and (allow_disease_specific or not bool(DISEASE_SPECIFIC.search(title)))
    )
    return {
        "passes": passes,
        "title_topics": title_topics,
        "body_topics": body_topics,
        "lexical_score": 3 * len(title_topics) + len(body_topics),
    }


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, document_id in enumerate(ranking, 1):
            scores[int(document_id)] = scores.get(int(document_id), 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Index,
        faiss_index: DenseIndex,
        encode_queries: Callable[[list[str]], np.ndarray],
        medcpt_scores: Callable[[str, list[dict[str, Any]]], list[float]],
        read_document: Callable[[int], dict[str, Any]],
    ) -> None:
        self.bm25 = bm25
        self.faiss = faiss_index
        self.encode_queries = encode_queries
        self.medcpt_scores = medcpt_scores
        self.read_document = read_document

    @staticmethod
    def _bm25_terms(plan: RetrievalPlan) -> list[str]:
        question = plan.query.split(". Question topics:", 1)[0]
        terms = [
            token for token in re.findall(r"[a-z0-9]+", question.lower())
            if len(token) > 2 and token not in STOPWORDS
        ]
        for topic in plan.question_topics:
            terms.extend(re.findall(r"[a-z0-9]+", TOPIC_QUERIES[topic].lower()))
        return list(dict.fromkeys(terms))

    def retrieve(
        self,
        plan: RetrievalPlan,
        *,
        top_k: int = 3,
        candidate_k: int = 500,
        allow_disease_specific: bool = False,
    ) -> list[dict[str, Any]]:
        if not plan.question_topics:
            return []

        bm25_scores = self.bm25.get_scores(self._bm25_terms(plan))
        k = min(candidate_k, len(bm25_scores))
        bm25_ids = np.argpartition(bm25_scores, -k)[-k:]
        bm25_ids = bm25_ids[np.argsort(bm25_scores[bm25_ids])[::-1]].tolist()

        dense_queries = [plan.query] + [TOPIC_QUERIES[t] for t in plan.question_topics]
        _, dense_ids = self.faiss.search(self.encode_queries(dense_queries), k)
        rankings = [bm25_ids] + [[int(x) for x in row if int(x) >= 0] for row in dense_ids]
        fused = reciprocal_rank_fusion(rankings)

        candidates: list[dict[str, Any]] = []
        for document_id in sorted(fused, key=fused.get, reverse=True):
            document = self.read_document(document_id)
            title = str(document.get("title") or document.get("paper_title") or "")
            abstract = str(document.get("abstract") or document.get("content") or document.get("text") or "")
            relevance = relevance_guard(title, abstract, plan.question_topics, allow_disease_specific)
            if relevance["passes"]:
                candidates.append({
                    "document_id": document_id,
                    "title": title,
                    "abstract": abstract,
                    "fusion_score": fused[document_id],
                    "relevance": relevance,
                })

        scores = self.medcpt_scores(plan.query, candidates) if candidates else []
        for item, score in zip(candidates, scores):
            item["medcpt_score"] = float(score)
        candidates.sort(
            key=lambda x: (x["medcpt_score"] + 0.15 * x["relevance"]["lexical_score"], x["fusion_score"]),
            reverse=True,
        )

        selected: list[dict[str, Any]] = []
        used_ids: set[int] = set()
        for topic in plan.question_topics:
            match = next(
                (x for x in candidates if x["document_id"] not in used_ids and topic in x["relevance"]["title_topics"]),
                None,
            )
            if match:
                used_ids.add(match["document_id"])
                selected.append(match)
            if len(selected) == top_k:
                break

        return [
            {
                "rank": rank,
                "document_id": item["document_id"],
                "title": item["title"],
                "snippet": re.sub(r"\s+", " ", item["abstract"]).strip()[:240],
                "focus_topics": item["relevance"]["title_topics"],
                "selected_image_cues": plan.selected_image_labels,
                "retriever": "BM25 + BGE-M3/FAISS + RRF + MedCPT",
            }
            for rank, item in enumerate(selected, 1)
        ]

