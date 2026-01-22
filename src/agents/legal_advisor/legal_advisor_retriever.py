"""
Legal Advisor Retriever - 조회(쿼리 조회 및 반환용)
- (1) Postgres -> 사건 정보 조회
- (2) Pinecone -> 법령/조문 검색
- (3) Pinecone -> 양형기준 검색
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Literal, List, Tuple, Set
import re
import json

from pinecone import Pinecone
from sqlalchemy import create_engine, text as sql_text

from src.config.settings import settings
from src.agents.common.base_agent import BaseAgent, AgentMessage
from src.rag.embeddings import EmbeddingGenerator
from src.agents.legal_advisor.rerank_utils import rerank_matches_by_keywords, tokenize_ko, keyword_overlap_score


# -------------------------
# 공통 유틸
# -------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_str(x: Any) -> str:
    return (x or "").strip() if isinstance(x, str) else str(x or "").strip()


def _extract_keywords_from_description(description: str, max_keywords: int = 10) -> str:
    """
    description에서 핵심 키워드 추출
    법률 용어, 범죄 관련 키워드, 핵심 사실을 추출하여 쿼리에 활용
    """
    if not description:
        return ""
    
    # 한글 토큰화
    tokens = tokenize_ko(description)
    
    # 불필요한 단어 제거 (조사, 일반 동사 등)
    stopwords = {
        "이", "가", "을", "를", "에", "의", "로", "으로", "와", "과", "는", "은",
        "하다", "되다", "있다", "없다", "이다", "되다", "하다", "그", "것", "수",
        "때", "경우", "때문", "위해", "통해", "대해", "관련", "및", "또한", "또는"
    }
    
    # 의미 있는 키워드만 추출 (2글자 이상, stopwords 제외)
    keywords = [t for t in tokens if len(t) >= 2 and t not in stopwords]
    
    # 중복 제거 및 빈도 기반 정렬
    keyword_counts = Counter(keywords)
    
    # 상위 키워드만 선택
    top_keywords = [kw for kw, _ in keyword_counts.most_common(max_keywords)]
    
    return " ".join(top_keywords)


def _build_improved_query_text(crime_type: str, description: str = "") -> str:
    """
    개선된 쿼리 텍스트 생성
    - 형법을 명시하여 형법 조문 우선 검색
    - crime_type을 중심으로 구성
    - description에서 핵심 키워드 추출하여 포함
    - 법률 용어와 구성요건을 명확히 표현
    """
    # 형법을 더 강조하고 범죄 유형을 명확히
    query_parts = ["형법", f"{crime_type}죄", crime_type, "구성요건", "처벌", "조문", "법조문"]
    
    # description에서 핵심 키워드 추출
    if description:
        keywords = _extract_keywords_from_description(description, max_keywords=6)
        if keywords:
            query_parts.append(keywords)
    
    # 최종 쿼리 텍스트 구성 (형법을 여러 번 언급하여 강조)
    query_text = " ".join(query_parts)
    
    return query_text


def extract_article_numbers(text: str) -> Set[str]:
    """
    조문 텍스트에서 다른 조문 번호를 추출
    예: "제297조, 제297조의2, 제298조부터 제300조까지" → {"297", "297의2", "298", "299", "300"}
    """
    article_numbers: Set[str] = set()
    
    if not text:
        return article_numbers
    
    # 패턴 1: "제XXX조의Y" 형식 (먼저 처리하여 정확히 매칭)
    pattern1 = r"제(\d+)조의(\d+)"
    matches1 = re.findall(pattern1, text)
    for num, sub_num in matches1:
        article_numbers.add(f"{num}의{sub_num}")
    
    # 패턴 2: "제XXX조부터 제YYY조까지" 형식 (범위)
    pattern2 = r"제(\d+)조부터\s*제(\d+)조까지"
    matches2 = re.findall(pattern2, text)
    for start, end in matches2:
        try:
            start_num = int(start)
            end_num = int(end)
            for num in range(start_num, end_num + 1):
                article_numbers.add(str(num))
        except (ValueError, TypeError):
            pass
    
    # 패턴 3: "제XXX조" 형식 (일반 조문, "의"가 없는 것만)
    pattern3 = r"제(\d+)조(?!의)"
    matches3 = re.findall(pattern3, text)
    for match in matches3:
        # 이미 "의"가 포함된 것은 제외
        if f"{match}의" not in text or not any(f"{match}의" in num for num in article_numbers):
            article_numbers.add(match)
    
    return article_numbers


ActionType = Literal[
    "get_case_info",
    "search_legal_provisions",
    "get_sentencing_guidelines",
    "get_all",  # case + laws + guidelines
]


class LegalAdvisorRetriever(BaseAgent):
    """
    조회 전용
    다른 에이전트가 action과 파라미터를 주면 필요한 조회만 수행해서 dict로 반환
    """

    def __init__(self):
        super().__init__(name="LegalAdvisorRetriever", role="legal_advisor_retriever")

        # Embedding (OpenAI)
        self.embedding = EmbeddingGenerator()

        # Postgres 엔진 (사건 정보 조회)
        if not settings.database_url:
            raise RuntimeError("settings.database_url is not set")
        # Convert to psycopg driver URL
        psycopg_url = settings.database_url.replace("postgresql://", "postgresql+psycopg://")
        self.db_engine = create_engine(psycopg_url, pool_pre_ping=True)

        # Pinecone - 공통 인덱스 (형법 조문용)
        # 형법은 공통 pinecone의 law_statue_criminal namespace에 있음
        # 공통 Pinecone API 키 사용 (pinecone_api_key 또는 law_pinecone_api_key)
        if not settings.pinecone_api_key:
            raise RuntimeError("pinecone_api_key or law_pinecone_api_key must be set in .env file")
        
        self.common_pc = Pinecone(api_key=settings.pinecone_api_key)
        self.common_index = self.common_pc.Index(settings.pinecone_index_name)
        self.criminal_law_namespace = "law_statue_criminal"

        # Pinecone - 법령/조문 인덱스 (legacy, 사용 안 함)
        # 공통 Pinecone만 사용하므로 초기화만 하고 사용하지 않음
        if settings.law_pinecone_api_key:
            self.law_pc = Pinecone(api_key=settings.law_pinecone_api_key)
            self.law_index = self.law_pc.Index(settings.law_pinecone_index_name) if settings.law_pinecone_index_name else None
            self.law_namespace = settings.law_pinecone_namespace or ""
        else:
            self.law_pc = None
            self.law_index = None
            self.law_namespace = ""

        # Pinecone - 양형기준 인덱스
        self.sentence_pc = Pinecone(api_key=settings.sentence_pinecone_api_key)
        self.sentence_index = self.sentence_pc.Index(settings.sentence_pinecone_index_name)
        self.sentence_namespace = settings.sentence_pinecone_namespace

    # ============================
    # 요청 라우터
    # ============================
    def handle_request(
        self,
        *,
        action: ActionType,
        case_id: Optional[str | int] = None,
        crime_type: Optional[str] = None,
        description: str = "",
        top_k: int = 5,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        오케스트레이터/다른 에이전트가 action과 파라미터를 주면 요청받은 조회 수행 후 dict로 반환
        """

        if action == "get_case_info":
            if case_id is None:
                return {"ok": False, "error": "case_id is required"}
            return {"ok": True, "action": action, "data": self.get_case_info(case_id), "retrieved_at": _now_iso()}

        # case_id -> crime_type/description로 세팅되도록 함
        if case_id is not None and (crime_type is None or description == ""):
            base_case = self.get_case_info(case_id)
            if base_case.get("error"):
                return {"ok": False, "action": action, "error": base_case["error"], "case": base_case}
            crime_type = crime_type or _safe_str(base_case.get("casename") or "UNKNOWN")
            description = description or _safe_str(base_case.get("description") or "")

        crime_type = _safe_str(crime_type or "UNKNOWN")

        if action == "search_legal_provisions":
            data = self.search_legal_provisions(crime_type=crime_type, description=description, top_k=top_k)
            return {
                "ok": True,
                "action": action,
                "query": {"crime_type": crime_type, "description_excerpt": description[:200]},
                "data": data,
                "retrieved_at": _now_iso(),
            }

        if action == "get_sentencing_guidelines":
            if namespace:
                self.sentence_namespace = namespace
            data = self.get_sentencing_guidelines(crime_type=crime_type, description=description, top_k=min(top_k, 10))
            return {
                "ok": True,
                "action": action,
                "query": {
                    "crime_type": crime_type,
                    "namespace": self.sentence_namespace,
                    "description_excerpt": description[:200],
                },
                "data": data,
                "retrieved_at": _now_iso(),
            }

        if action == "get_all":
            case = self.get_case_info(case_id) if case_id is not None else None
            laws = self.search_legal_provisions(crime_type=crime_type, description=description, top_k=top_k)
            guides = self.get_sentencing_guidelines(crime_type=crime_type, description=description, top_k=min(top_k, 10))
            return {"ok": True, "action": action, "case": case, "laws": laws, "guidelines": guides, "retrieved_at": _now_iso()}

        return {"ok": False, "error": f"unknown action: {action}"}

    # =========================================================
    # 에이전트 표준 진입점: context.action 기반으로 조회
    # =========================================================
    async def generate_response(self, case_info: Dict[str, Any], context: Dict[str, Any]) -> AgentMessage:
        action: ActionType = context.get("action", "get_all")
        case_id = case_info.get("id") or case_info.get("case_id")

        result = self.handle_request(
            action=action,
            case_id=case_id,
            crime_type=context.get("crime_type"),
            description=context.get("description", ""),
            top_k=context.get("top_k", 5),
            namespace=context.get("namespace"),
        )

        return AgentMessage(role=self.role, content=str(result), metadata={"ok": result.get("ok", False)})

    # =========================================================
    # (1) Postgres: 사건 정보 가져오기
    # =========================================================
    def get_case_info(self, case_id: str | int) -> Dict[str, Any]:
        query = sql_text("""
            SELECT
                id,
                casetype,
                casename,
                facts,
                description,
                sentencing_factors
            FROM cases
            WHERE id = :case_id
            LIMIT 1
        """)

        with self.db_engine.connect() as conn:
            row = conn.execute(query, {"case_id": case_id}).mappings().first()

        if not row:
            return {
                "id": case_id,
                "casetype": None,
                "casename": None,
                "facts": "",
                "description": "",
                "sentencing_factors": None,
                "error": "case not found",
            }

        # sentencing_factors 파싱 (TEXT 타입으로 저장된 JSON 문자열)
        sentencing_factors = None
        if row.get("sentencing_factors"):
            try:
                if isinstance(row["sentencing_factors"], str):
                    sentencing_factors = json.loads(row["sentencing_factors"])
                else:
                    # 이미 dict인 경우
                    sentencing_factors = row["sentencing_factors"]
            except (json.JSONDecodeError, TypeError) as e:
                # JSON 파싱 실패 시 None 유지
                sentencing_factors = None

        return {
            "id": row["id"],
            "casetype": _safe_str(row["casetype"]),
            "casename": _safe_str(row["casename"]),
            "facts": _safe_str(row["facts"]),
            "description": _safe_str(row["description"]),
            "sentencing_factors": sentencing_factors,
        }

    # =========================================================
    # (2) Pinecone: 법령/조문 검색
    # =========================================================
    def search_legal_provisions(
        self, 
        crime_type: str, 
        description: str = "", 
        top_k: int = 5,
        min_score: float = 0.50,  # 초기 필터링용 (낮게 설정)
        search_candidates: int = 30  # 더 많은 후보 검색
    ) -> Dict[str, Any]:
        """
        법령/조문 검색 (개선된 버전)
        
        Args:
            crime_type: 범죄 유형
            description: 사건 설명
            top_k: 최종 반환할 결과 수
            min_score: 초기 필터링 최소 유사도 임계값 (낮게 설정하여 후보 확보)
            search_candidates: 초기 검색 후보 수 (기본 30, re-ranking 후 필터링)
        """
        crime_type = _safe_str(crime_type)
        if not crime_type or crime_type == "UNKNOWN":
            return {"status": "fail", "laws": [], "citations": [], "error": "crime_type empty"}

        # 개선된 쿼리 텍스트 생성 (형법 명시)
        query_text = _build_improved_query_text(crime_type, description)
        
        # Re-ranking을 위한 키워드 쿼리 텍스트 (형법 + 범죄명 강조)
        keyword_query_text = f"형법 {crime_type}" + (f" {description[:200]}" if description else "")

        try:
            vector = self.embedding.encode_query(query_text)
        except Exception as e:
            return {"status": "fail", "laws": [], "citations": [], "error": f"embedding failed: {e}"}

        # 공통 Pinecone의 형법 namespace만 사용하여 검색
        matches = []
        error_details = None
        debug_info = []
        
        # 여러 방법으로 검색 시도
        search_attempts = [
            # 시도 1: 필터 없이 검색 (가장 관대한 방법)
            {
                "filter": None,
                "description": "without filter"
            },
            # 시도 2: type 필터와 함께 검색
            {
                "filter": {"type": {"$in": ["article", "paragraph"]}},
                "description": "with type filter"
            },
            # 시도 3: law_name 필터로 형법만 검색
            {
                "filter": {"law_name": {"$eq": "형법"}},
                "description": "with law_name filter"
            },
        ]
        
        for attempt in search_attempts:
            try:
                query_params = {
                    "vector": vector,
                    "top_k": search_candidates,
                    "include_metadata": True,
                    "namespace": self.criminal_law_namespace,
                }
                if attempt["filter"]:
                    query_params["filter"] = attempt["filter"]
                
                res = self.common_index.query(**query_params)
                attempt_matches = res.get("matches") or []
                
                debug_info.append(f"{attempt['description']}: {len(attempt_matches)} matches")
                
                if attempt_matches:
                    matches = attempt_matches
                    # 검색 성공하면 중단
                    break
            except Exception as e:
                error_details = str(e)
                debug_info.append(f"{attempt['description']}: error - {str(e)}")
                # 다음 시도 계속
        
        if not matches:
            # 모든 시도 실패 - 디버깅 정보 포함
            error_msg = f"no matches found in namespace '{self.criminal_law_namespace}' (index: bupblessyou-judgments)"
            if debug_info:
                error_msg += f" [debug: {', '.join(debug_info)}]"
            if error_details:
                error_msg += f" (last error: {error_details})"
            return {"status": "fail", "laws": [], "citations": [], "error": error_msg}

        # 1단계: 최소 유사도로 초기 필터링 (너무 낮은 점수 제거)
        # 하지만 결과가 없으면 임계값을 낮춰서라도 결과 반환
        filtered_matches = [m for m in matches if (m.get("score") or 0.0) >= min_score]
        
        if not filtered_matches:
            # 최소한 상위 결과는 유지 (임계값 무시)
            filtered_matches = matches[:min(search_candidates, len(matches))]
            # 점수가 있는 결과라도 모두 포함
            if not filtered_matches:
                filtered_matches = [m for m in matches if m.get("score") is not None]

        # 2단계: 1차 필터링 - law_name, law_id, title로 형법 우선 선택
        def is_criminal_law(match: Dict[str, Any]) -> bool:
            """형법 관련 법령인지 확인"""
            md = match.get("metadata") or {}
            law_name = _safe_str(md.get("law_name") or "")
            law_id = _safe_str(md.get("law_id") or "")
            article_title = _safe_str(md.get("article_title") or "")
            
            # law_name에 "형법"이 포함되어 있으면 형법
            if "형법" in law_name:
                return True
            
            # law_id에 "형법"이 포함되어 있으면 형법
            if law_id and "형법" in law_id:
                return True
            
            # article_title에 범죄 관련 키워드가 있으면 형법일 가능성 높음
            crime_keywords = ["죄", "처벌", "범죄", "형사"]
            if any(keyword in article_title for keyword in crime_keywords):
                return True
            
            return False

        # 형법 우선 분류
        criminal_law_matches = []
        other_law_matches = []
        
        for m in filtered_matches:
            if is_criminal_law(m):
                criminal_law_matches.append(m)
            else:
                other_law_matches.append(m)
        
        # 형법이 있으면 형법 우선, 없으면 다른 법령도 포함
        if criminal_law_matches:
            primary_matches = criminal_law_matches
            secondary_matches = other_law_matches
        else:
            # 형법이 없으면 모든 결과 사용
            primary_matches = filtered_matches
            secondary_matches = []

        # 3단계: 2차 필터링 - 조문 내용(text)으로 유사도 높은 것 선택
        def calculate_text_relevance_score(match: Dict[str, Any]) -> float:
            """조문 내용 기반 관련성 점수 계산"""
            md = match.get("metadata") or {}
            text_body = _safe_str(md.get("text") or md.get("content") or "")
            law_name = _safe_str(md.get("law_name") or "")
            is_criminal = is_criminal_law(match)
            
            if not text_body:
                return 0.0
            
            score = 0.0
            
            # 형법 조문이면 기본 점수 부여 (매우 중요!)
            if is_criminal:
                score += 0.4
            else:
                # 형법이 아니면 감점
                score -= 0.2
            
            # crime_type이 조문에 직접 포함되면 매우 높은 점수
            if crime_type in text_body:
                score += 0.6 if is_criminal else 0.3
            
            # 형법 + crime_type 조합이면 추가 보너스
            if is_criminal and crime_type in text_body:
                score += 0.2
            
            # 키워드 오버랩 점수
            keyword_score = keyword_overlap_score(keyword_query_text, text_body)
            # 키워드 점수를 0~0.3 범위로 정규화
            normalized_keyword = min(keyword_score / 25.0, 0.3)
            score += normalized_keyword
            
            # description의 핵심 키워드가 포함되면 추가 점수
            if description:
                desc_keywords = _extract_keywords_from_description(description, max_keywords=5)
                if desc_keywords:
                    for keyword in desc_keywords.split():
                        if keyword in text_body:
                            score += 0.05
            
            return max(0.0, min(score, 1.0))  # 0 이상으로 제한
        
        # 조문 내용 기반 점수 계산 및 정렬
        scored_primary: List[Tuple[float, Dict[str, Any]]] = []
        for m in primary_matches:
            text_score = calculate_text_relevance_score(m)
            vector_score = m.get("score") or 0.0
            is_criminal = is_criminal_law(m)
            
            # 형법이면 더 높은 가중치, 형법이 아니면 낮은 가중치
            if is_criminal:
                # 형법: 조문 내용 점수에 더 높은 가중치
                combined_score = (vector_score * 0.3) + (text_score * 0.7)
                # 형법 보너스
                combined_score += 0.2
            else:
                # 형법 아님: 벡터 점수에 더 높은 가중치 (낮은 점수)
                combined_score = (vector_score * 0.6) + (text_score * 0.4)
                # 형법 아님 페널티
                combined_score -= 0.3
            
            scored_primary.append((max(0.0, combined_score), m))
        
        scored_primary.sort(key=lambda x: x[0], reverse=True)
        
        # secondary matches도 점수 계산
        scored_secondary: List[Tuple[float, Dict[str, Any]]] = []
        for m in secondary_matches:
            text_score = calculate_text_relevance_score(m)
            vector_score = m.get("score") or 0.0
            # 형법이 아니면 낮은 점수
            combined_score = (vector_score * 0.6) + (text_score * 0.4) - 0.3
            scored_secondary.append((max(0.0, combined_score), m))
        
        scored_secondary.sort(key=lambda x: x[0], reverse=True)
        
        # 4단계: 최종 선택 - 형법 우선, combined_score >= 0.6 필터 적용
        final_matches = []
        final_scored_matches: List[Tuple[float, Dict[str, Any]]] = []
        
        # 형법 결과를 우선 추가 (combined_score >= 0.6 필터 적용)
        for combined_score, m in scored_primary:
            text_score = calculate_text_relevance_score(m)
            is_criminal = is_criminal_law(m)
            
            # Faithfulness 향상을 위해 combined_score >= 0.6 필터 적용
            if combined_score >= 0.6:
                if is_criminal:
                    # 형법이면 더 관대한 text_score 기준 (하지만 combined_score는 이미 0.6 이상)
                    if text_score >= 0.2 or len(final_scored_matches) < top_k:
                        final_scored_matches.append((combined_score, m))
                        if len(final_scored_matches) >= top_k:
                            break
                else:
                    # 형법이 아니면 더 엄격한 text_score 기준
                    if text_score >= 0.5 and len(final_scored_matches) < top_k:
                        final_scored_matches.append((combined_score, m))
                        if len(final_scored_matches) >= top_k:
                            break
        
        # 형법 결과가 부족하면 다른 법령도 추가 (하지만 매우 높은 기준: combined_score >= 0.6)
        if len(final_scored_matches) < top_k:
            for combined_score, m in scored_secondary:
                text_score = calculate_text_relevance_score(m)
                # 다른 법령은 매우 높은 기준 적용 (combined_score >= 0.6, text_score >= 0.6)
                if combined_score >= 0.6 and text_score >= 0.6:
                    final_scored_matches.append((combined_score, m))
                    if len(final_scored_matches) >= top_k:
                        break
        
        # 최종 상위 top_k개만 선택 (combined_score 기준으로 정렬)
        final_scored_matches.sort(key=lambda x: x[0], reverse=True)
        final_matches = [m for _, m in final_scored_matches[:top_k]]

        laws = []
        citations = []
        seen_ids = {m.get("id") for m in final_matches if m.get("id")}  # 중복 제거용

        for i, m in enumerate(final_matches, start=1):
            md = (m.get("metadata") or {})
            law_name = _safe_str(md.get("law_name"))
            article_no_raw = _safe_str(md.get("article_no"))
            article_title = _safe_str(md.get("article_title"))
            text_body = _safe_str(md.get("text") or md.get("content"))
            paragraph_no = md.get("paragraph_no")
            law_id = _safe_str(md.get("law_id") or "")

            if not (law_name and article_no_raw and text_body):
                continue

            # article_no 포맷팅
            article_no = article_no_raw
            if not article_no.startswith("제"):
                article_no = f"제{article_no}조"
            if article_no.startswith("제") and not article_no.endswith("조"):
                article_no = f"{article_no}조"
            
            # paragraph_no가 있으면 추가
            if paragraph_no:
                try:
                    para_num = int(paragraph_no)
                    article_no = f"{article_no} {para_num}항"
                except (ValueError, TypeError):
                    pass

            cid = f"C_LAW_{i}"
            laws.append(
                {
                    "citation_id": cid,
                    "law_name": law_name,
                    "article_no": article_no,
                    "article_title": article_title,
                    "text": text_body,
                    "score": m.get("score"),
                    "law_id": law_id if law_id else None,
                    "paragraph_no": paragraph_no if paragraph_no else None,
                }
            )
            citations.append(
                {
                    "id": cid,
                    "type": "law",
                    "title": f"{law_name} {article_no}",
                    "source": "internal_rag",
                    "pinpoint": m.get("id"),
                    "retrieved_at": _now_iso(),
                }
            )

        # 연계 조문 검색: 조문 텍스트에서 다른 조문 번호 추출하여 추가 검색
        related_articles = set()
        for law in laws:
            text = law.get("text", "")
            article_nums = extract_article_numbers(text)
            related_articles.update(article_nums)
        
        # 연계 조문이 있으면 공통 Pinecone에서 추가 검색
        related_laws = []
        if related_articles:
            try:
                # 각 조문 번호로 검색
                for article_num in list(related_articles)[:10]:  # 최대 10개만 검색
                    # 조문 번호로 필터링하여 검색
                    try:
                        # article_no로 필터링하여 정확한 조문 검색
                        related_res = self.common_index.query(
                            vector=vector,  # 같은 벡터 사용
                            top_k=3,
                            include_metadata=True,
                            namespace=self.criminal_law_namespace,
                            filter={
                                "type": {"$in": ["article", "paragraph"]},
                                "article_no": {"$eq": article_num}
                            },
                        )
                        
                        for related_match in related_res.get("matches") or []:
                            # 이미 포함된 조문은 제외
                            if related_match.get("id") in seen_ids:
                                continue
                            
                            related_md = related_match.get("metadata") or {}
                            related_law_name = _safe_str(related_md.get("law_name") or "")
                            related_article_no = _safe_str(related_md.get("article_no") or "")
                            related_text = _safe_str(related_md.get("text") or related_md.get("content") or "")
                            
                            if related_law_name == "형법" and related_article_no and related_text:
                                # 이미 추가된 조문 번호와 중복 체크
                                is_duplicate = False
                                for existing_law in laws:
                                    if existing_law.get("article_no", "").replace("제", "").replace("조", "").split()[0] == related_article_no:
                                        is_duplicate = True
                                        break
                                
                                if not is_duplicate:
                                    related_article_formatted = related_article_no
                                    if not related_article_formatted.startswith("제"):
                                        related_article_formatted = f"제{related_article_formatted}조"
                                    
                                    related_paragraph_no = related_md.get("paragraph_no")
                                    if related_paragraph_no:
                                        try:
                                            para_num = int(related_paragraph_no)
                                            related_article_formatted = f"{related_article_formatted} {para_num}항"
                                        except (ValueError, TypeError):
                                            pass
                                    
                                    related_law_id = _safe_str(related_md.get("law_id") or "")
                                    
                                    related_laws.append({
                                        "citation_id": f"C_LAW_RELATED_{len(related_laws) + 1}",
                                        "law_name": related_law_name,
                                        "article_no": related_article_formatted,
                                        "article_title": _safe_str(related_md.get("article_title") or ""),
                                        "text": related_text,
                                        "score": related_match.get("score"),
                                        "law_id": related_law_id if related_law_id else None,
                                        "paragraph_no": related_paragraph_no if related_paragraph_no else None,
                                        "is_related": True,  # 연계 조문 표시
                                    })
                    except Exception:
                        # 개별 조문 검색 실패는 무시하고 계속
                        continue
            except Exception:
                # 연계 조문 검색 실패는 무시하고 기존 결과만 반환
                pass
        
        # 연계 조문을 laws에 추가 (score >= 0.6인 것만)
        for related_law in related_laws:
            score = related_law.get("score") or 0.0
            if score >= 0.6:
                laws.append(related_law)
                # citation도 추가
                citations.append({
                    "id": related_law.get("citation_id", ""),
                    "type": "law",
                    "title": f"{related_law.get('law_name', '')} {related_law.get('article_no', '')}",
                    "source": "internal_rag",
                    "pinpoint": related_law.get("law_id", ""),
                    "retrieved_at": _now_iso(),
                })

        # 품질 지표 계산 (디버깅 및 모니터링용)
        quality_metrics = {}
        if laws:
            # 벡터 유사도 점수의 평균
            vector_scores = [law.get("score", 0.0) for law in laws if law.get("score") is not None]
            avg_vector_score = sum(vector_scores) / len(vector_scores) if vector_scores else 0.0
            
            # 키워드 오버랩 점수 계산
            keyword_scores = []
            for law in laws:
                text = law.get("text", "")
                if text:
                    overlap = keyword_overlap_score(keyword_query_text, text)
                    keyword_scores.append(overlap)
            avg_keyword_score = sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0.0
            
            # 형법 비율
            criminal_law_count = sum(1 for law in laws if "형법" in law.get("law_name", ""))
            
            quality_metrics = {
                "average_score": avg_vector_score,
                "avg_keyword_score": avg_keyword_score,
                "total_laws": len(laws),
                "criminal_law_count": criminal_law_count,
                "criminal_law_ratio": criminal_law_count / len(laws) if laws else 0.0
            }
        else:
            quality_metrics = {
                "average_score": 0.0,
                "avg_keyword_score": 0.0,
                "total_laws": 0,
                "criminal_law_count": 0,
                "criminal_law_ratio": 0.0
            }

        return {
            "status": "success" if laws else "fail",
            "laws": laws,
            "citations": citations,
            "quality_metrics": quality_metrics
        }

    # =========================================================
    # (3) Pinecone: 양형기준 검색
    # =========================================================
    def get_sentencing_guidelines(
        self,
        crime_type: str,
        description: str = "",
        top_k: int = 3,
        sentencing_factors: Optional[Dict[str, Any]] = None,
        crime_number: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        양형기준 검색 (개선된 버전: Reranking 및 score 필터링 적용)

        Args:
            crime_type: 범죄 유형
            description: 사건 설명 (case_attributes 포함 가능)
            top_k: 최종 반환할 결과 수
            sentencing_factors: 양형 고려 요소 (선택적, reranking 점수 계산에 활용)
            crime_number: 양형기준 필터링용 (예: "criterion_10", "criterion_03")

        Returns:
            검색 결과 딕셔너리: {"status": "...", "guidelines": [...], "error": ...}
        """
        crime_type = _safe_str(crime_type)
        if not crime_type or crime_type == "UNKNOWN":
            return {"status": "fail", "guidelines": [], "error": "crime_type empty"}

        # crime_type으로 crime_number 자동 매핑 (명시적으로 제공되지 않은 경우)
        if not crime_number:
            crime_type_lower = crime_type.lower()
            if "사기" in crime_type_lower:
                crime_number = "criterion_10"
            elif "강제추행" in crime_type_lower or "성범죄" in crime_type_lower:
                crime_number = "criterion_03"
            # 다른 케이스 추가 가능

        query_text = f"양형기준 {crime_type} 기준형 범위 가중 감경 요소"
        if description:
            query_text += f" | 사건개요: {description[:200]}"

        try:
            vector = self.embedding.encode_query(query_text)
        except Exception as e:
            return {"status": "fail", "guidelines": [], "error": f"embedding failed: {e}"}

        # Pinecone 쿼리 파라미터 구성
        query_params = {
            "vector": vector,
            "top_k": min(top_k * 3, 15),  # reranking을 위해 더 많이 검색
            "include_metadata": True,
            "namespace": self.sentence_namespace,
        }

        # crime_number 필터 추가
        if crime_number:
            query_params["filter"] = {"crime_number": {"$eq": crime_number}}

        try:
            res = self.sentence_index.query(**query_params)
        except Exception as e:
            return {"status": "fail", "guidelines": [], "error": f"pinecone query failed: {e}"}

        matches = (res.get("matches") or [])
        if not matches:
            return {"status": "fail", "guidelines": [], "error": "no matches"}

        # Reranking: crime_type, description, sentencing_factors 기반 점수 계산
        def calculate_guideline_score(match: Dict[str, Any]) -> float:
            """양형기준 관련성 점수 계산"""
            md = match.get("metadata") or {}
            raw_text = _safe_str(md.get("text") or md.get("content") or "")
            guideline_name = _safe_str(md.get("guideline_name") or md.get("title") or "")
            match_crime_type = _safe_str(md.get("crime_type") or "")
            
            score = 0.0
            vector_score = match.get("score") or 0.0
            
            # 벡터 점수 기반 점수 (0.4 가중치로 조정)
            score += vector_score * 0.4
            
            # crime_type 매칭 (0.25 가중치로 조정)
            if crime_type in match_crime_type or crime_type in guideline_name:
                score += 0.25
            if crime_type in raw_text:
                score += 0.15
            
            # description 키워드 매칭 (0.15 가중치로 조정)
            if description:
                desc_keywords = _extract_keywords_from_description(description, max_keywords=5)
                if desc_keywords:
                    keyword_matches = sum(1 for kw in desc_keywords.split() if kw in raw_text)
                    score += min(keyword_matches * 0.03, 0.15)  # 최대 0.15
            
            # sentencing_factors 매칭 (0.2 가중치 추가)
            if sentencing_factors:
                factors_score = 0.0
                raw_text_lower = raw_text.lower()
                
                # 전과 정보 매칭
                if "전력_관련" in sentencing_factors:
                    전력 = sentencing_factors["전력_관련"]
                    if 전력.get("동종_전과_횟수") and 전력.get("동종_전과_횟수") > 0:
                        if any(kw in raw_text_lower for kw in ["전과", "전력", "처벌받은", "동종"]):
                            factors_score += 0.06
                    elif 전력.get("집행유예_이상_전과_존재") is True:
                        if any(kw in raw_text_lower for kw in ["전과", "전력", "처벌받은", "집행유예"]):
                            factors_score += 0.06
                
                # 반성/인정 매칭
                if "책임_인정_관련" in sentencing_factors:
                    책임 = sentencing_factors["책임_인정_관련"]
                    if 책임.get("범행_인정") is True:
                        if any(kw in raw_text_lower for kw in ["범행 인정", "인정", "자백"]):
                            factors_score += 0.04
                    if 책임.get("반성_여부") is True:
                        if any(kw in raw_text_lower for kw in ["반성", "뉘우침", "자책"]):
                            factors_score += 0.04
                
                # 합의/피해 회복 매칭
                if "기타_참작사유" in sentencing_factors:
                    기타 = sentencing_factors["기타_참작사유"]
                    if 기타.get("합의") is True:
                        if any(kw in raw_text_lower for kw in ["합의", "조정", "화해"]):
                            factors_score += 0.04
                    if 기타.get("피해_회복") is True or (기타.get("피해_회복") and isinstance(기타.get("피해_회복"), (int, float)) and 기타.get("피해_회복") > 0):
                        if any(kw in raw_text_lower for kw in ["피해 회복", "배상", "보상", "회복"]):
                            factors_score += 0.04
                    if 기타.get("자수") is True:
                        if any(kw in raw_text_lower for kw in ["자수", "출석"]):
                            factors_score += 0.02
                
                score += min(factors_score, 0.2)  # 최대 0.2
            
            return min(score, 1.0)
        
        # Reranking 수행
        scored_matches: List[Tuple[float, Dict[str, Any]]] = []
        for match in matches:
            rerank_score = calculate_guideline_score(match)
            scored_matches.append((rerank_score, match))
        
        # 점수 순으로 정렬
        scored_matches.sort(key=lambda x: x[0], reverse=True)
        
        # score >= 0.6인 결과만 필터링 (Faithfulness 향상)
        filtered_guidelines = []
        for rerank_score, match in scored_matches:
            if rerank_score >= 0.6:
                filtered_guidelines.append((rerank_score, match))
                if len(filtered_guidelines) >= top_k:
                    break
        
        # 필터링된 결과가 없으면 상위 결과라도 포함 (하지만 최소 0.5 이상)
        if not filtered_guidelines:
            for rerank_score, match in scored_matches:
                if rerank_score >= 0.5:
                    filtered_guidelines.append((rerank_score, match))
                    if len(filtered_guidelines) >= min(top_k, 1):  # 최소 1개는 반환
                        break
        
        # 양형기준 딕셔너리 리스트 생성
        guidelines = []
        for i, (rerank_score, match) in enumerate(filtered_guidelines, start=1):
            md = (match.get("metadata") or {})
            raw_text = _safe_str(md.get("text") or md.get("content"))

            guideline_name = _safe_str(md.get("guideline_name") or md.get("title") or md.get("category") or f"양형기준 > {crime_type}")
            min_months = md.get("min_months")
            max_months = md.get("max_months")
            range_text = _safe_str(md.get("range_text"))

            try:
                min_months = int(min_months) if min_months is not None else None
            except Exception:
                min_months = None
            try:
                max_months = int(max_months) if max_months is not None else None
            except Exception:
                max_months = None

            guideline = {
                "citation_id": f"C_GUIDE_{i}",
                "crime_type": _safe_str(md.get("crime_type") or crime_type),
                "guideline_name": guideline_name,
                "base_range": {
                    "min_months": min_months,
                    "max_months": max_months,
                    "text": range_text or "",
                },
                "factors": {
                    "aggravating": md.get("aggravating") or [],
                    "mitigating": md.get("mitigating") or [],
                },
                "raw_text_excerpt": raw_text[:400] if raw_text else "",
                "text": raw_text,  # 전체 텍스트 포함
                "tab_title": _safe_str(md.get("tab_title") or ""),  # 탭 제목 포함
                "score": rerank_score,  # reranking 점수 사용
                "vector_score": match.get("score"),  # 원본 벡터 점수도 포함
            }
            guidelines.append(guideline)

        return {
            "status": "success" if guidelines else "fail",
            "guidelines": guidelines,
            "error": None
        }


if __name__ == "__main__":
    """
    Legal Advisor Retriever 테스트
    사용법: python -m src.agents.legal_advisor.legal_advisor_retriever
    """
    import json
    
    print("Legal Advisor Retriever 테스트 시작...")
    print("=" * 60)
    
    try:
        retriever = LegalAdvisorRetriever()
        
        # 테스트 1: 법령 검색
        print("\n[테스트 1] 법령 검색 테스트")
        print("-" * 60)
        result = retriever.search_legal_provisions(
            crime_type="폭행",
            description="피해자가 장애인인 폭행 사건으로, 가해자가 피해자의 신체를 폭행하여 상해를 입혔습니다.",
            top_k=5
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        # 테스트 2: handle_request를 통한 검색
        print("\n[테스트 2] handle_request를 통한 검색")
        print("-" * 60)
        result2 = retriever.handle_request(
            action="search_legal_provisions",
            crime_type="사기",
            description="피해자에게 거짓말을 하여 재물을 편취한 사기 사건",
            top_k=3
        )
        print(json.dumps(result2, ensure_ascii=False, indent=2))
        
        print("\n" + "=" * 60)
        print("테스트 완료!")
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
