"""
Legal Advisor Output Schema
"""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime

""" 
케이스 정보 
사건 ID, 범죄 유형, 기소 내용, 사건 요약
"""
class CaseInfo(BaseModel):
    case_id: str | int
    crime_type: str
    charges: List[str] = []
    description: str
    facts: str

""" 
검색 쿼리 및 필터링 옵션
success = 정상적으로 찾음,
partial = 일부만 찾음,
fail = 찾지 못함
"""
class RetrievalFilters(BaseModel):
    court_level: List[Literal["supreme", "high", "district"]] = []
    year_from: Optional[int] = None
    year_to: Optional[int] = None

class RetrievalQuery(BaseModel):
    crime_type: str
    keywords: List[str] = []
    filters: RetrievalFilters

class RetrievalStatus(BaseModel):
    laws: Literal["success", "partial", "fail"]
    sentencing_guidelines: Literal["success", "partial", "fail"]
    precedents: Literal["success", "partial", "fail"]

class RetrievalError(BaseModel):
    component: Literal["laws", "sentencing_guidelines", "precedents"]
    code: Optional[str] = None
    message: str
    retryable: bool = False

class RetrievalInfo(BaseModel):
    query: RetrievalQuery
    status: RetrievalStatus
    errors: List[RetrievalError] = []

""" 법령 (laws) """
class LawElement(BaseModel):
    name: str
    interpretation: str
    case_fit_hint: Optional[str] = None

class LawRelevance(BaseModel):
    label: Literal["core", "supporting", "related"]
    reason: str

class LawItem(BaseModel):
    citation_id: str
    law_name: str
    law_id: Optional[str] = None
    article_no: str
    article_title: str
    clause: Optional[str] = None
    text: str
    effective_date: Optional[str] = None  # YYYY-MM-DD
    relevance: LawRelevance
    elements: List[LawElement] = []

""" 양형기준 (sentencing_guidelines) """
class SentencingRange(BaseModel):
    min_months: int
    max_months: int
    text: str

class SentencingFactor(BaseModel):
    factor: str
    weight_hint: Optional[Literal["high", "mid", "low"]] = None
    applies_if: Optional[str] = None

class SentencingFactors(BaseModel):
    aggravating: List[SentencingFactor] = []
    mitigating: List[SentencingFactor] = []

class SentencingGuidelines(BaseModel):
    citation_id: str
    crime_type: str
    guideline_name: str
    base_range: SentencingRange
    factors: SentencingFactors
    notes: List[str] = []

""" Legal Opinion (analysis) """
class IssueSpotting(BaseModel):
    issue: str
    why_it_matters: str
    relevant_to: List[Literal["prosecution", "defense", "judge"]]

class ElementChecklistItem(BaseModel):
    element: str
    supporting_facts: List[str]
    risk_notes: Optional[str] = None

class CounterArgument(BaseModel):
    claim: str
    support: List[str]
    weakness: Optional[str] = None

class LegalOpinion(BaseModel):
    summary: str
    elements_checklist: List[ElementChecklistItem] = []
    counterarguments: List[CounterArgument] = []

class AnalysisResult(BaseModel):
    issue_spotting: List[IssueSpotting] = []
    legal_opinion: LegalOpinion


""" Citations(출처) """
class Citation(BaseModel):
    id: str
    type: Literal["law", "precedent", "guideline"]
    title: str
    source: Literal[
        "law.go.kr",
        "scourt.go.kr",
        "sentencing_commission",
        "internal_rag"
    ]
    url: Optional[str] = None
    pinpoint: Optional[str] = None
    retrieved_at: datetime


""" Quality(신뢰도) """
class Coverage(BaseModel):
    laws: float = 0.0
    guidelines: float = 0.0
    precedents: float = 0.0

class QualityInfo(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    coverage: Coverage
    notes: List[str] = []
    manual_review_required: bool = False


""" 최종 LegalAdvisor 출력 모델 """
class LegalOutputs(BaseModel):
    laws: List[LawItem] = []
    sentencing_guidelines: Optional[SentencingGuidelines] = None
    precedents: List[dict] = []

class LegalAdvisorOutput(BaseModel):
    schema_version: Literal["legal_advice.v1"] = "legal_advice.v1"
    case: CaseInfo
    retrieval: RetrievalInfo
    outputs: LegalOutputs
    analysis: AnalysisResult
    citations: List[Citation] = []
    quality: QualityInfo