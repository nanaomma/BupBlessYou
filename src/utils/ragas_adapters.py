"""
RAGAS 평가용 데이터 변환 어댑터 모듈

각 RAG 타입(법령, 양형기준, 판례)별로 검색 결과를
RAGAS 평가 데이터 형식으로 변환하는 어댑터 함수들을 제공합니다.

Usage:
    from src.utils.ragas_adapters import prepare_sentencing_guideline_evaluation

    eval_data = prepare_sentencing_guideline_evaluation(
        crime_type="사기",
        description="피해액 5000만원",
        guideline_result={"status": "success", "guideline": {...}}
    )

    if eval_data:
        evaluate_rag_quality(**eval_data)
"""
from typing import Dict, List, Optional, Any


def prepare_sentencing_guideline_evaluation(
    crime_type: str,
    description: str,
    guideline_result: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    양형기준 검색 결과를 RAGAS 평가용 데이터로 변환

    Args:
        crime_type: 범죄 유형 (예: "사기", "폭행")
        description: 사건 설명
        guideline_result: get_sentencing_guidelines() 반환 결과

    Returns:
        RAGAS 평가용 데이터 딕셔너리
    """
    # 'guidelines' 리스트에서 첫 번째 항목 추출 (가장 관련성 높은 결과)
    guidelines = guideline_result.get("guidelines", [])
    if not guidelines:
        # 하위 호환성: 'guideline' 키가 직접 있는 경우 (단일 결과)
        guideline = guideline_result.get("guideline")
        if not guideline:
            return None
    else:
        guideline = guidelines[0]

    # ========================================
    # 1. 질문 구성 (검색 쿼리와 동일한 형식)
    # ========================================
    question = f"양형기준 {crime_type} 기준형 범위 가중 감경 요소"
    if description:
        question += f" | 사건개요: {description[:200]}"

    # ========================================
    # 2. 답변 구성 (검색된 양형기준 정보)
    # ========================================
    answer_parts = []

    # 기준형 범위
    base_range = guideline.get("base_range", {})
    if base_range.get("text"):
        answer_parts.append(f"기준형 범위: {base_range['text']}")

    # 가중/감경 요소
    factors = guideline.get("factors", {})
    if factors.get("aggravating"):
        aggravating_str = ", ".join(factors["aggravating"][:3])
        answer_parts.append(f"가중요소: {aggravating_str}")
    if factors.get("mitigating"):
        mitigating_str = ", ".join(factors["mitigating"][:3])
        answer_parts.append(f"감경요소: {mitigating_str}")

    # 원문 발췌
    if guideline.get("raw_text_excerpt"):
        excerpt = guideline["raw_text_excerpt"][:200]
        answer_parts.append(f"상세내용: {excerpt}")

    # 답변이 비어있으면 최소한 양형기준명 사용
    answer = "\n".join(answer_parts) if answer_parts else guideline.get("guideline_name", "")

    # ========================================
    # 3. 컨텍스트 구성 (검색에 사용된 원본 데이터)
    # ========================================
    contexts = []

    if guideline.get("raw_text_excerpt"):
        contexts.append(guideline["raw_text_excerpt"])

    if guideline.get("guideline_name"):
        contexts.append(f"양형기준명: {guideline['guideline_name']}")

    # ========================================
    # 4. 유효성 검증
    # ========================================
    if not contexts or not answer:
        return None

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts
    }


def prepare_legal_provision_evaluation(
    crime_type: str,
    description: str,
    laws_result: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    법령 검색 결과를 RAGAS 평가용 데이터로 변환

    Args:
        crime_type: 범죄 유형
        description: 사건 설명
        laws_result: search_legal_provisions() 반환 결과

    Returns:
        RAGAS 평가용 데이터 또는 None
    """
    laws = laws_result.get("laws", [])
    if not laws:
        return None

    # ========================================
    # 1. 질문 구성
    # ========================================
    question = f"{crime_type} 관련 처벌 조항 및 법령"
    if description:
        question += f" | 사건개요: {description[:200]}"

    # ========================================
    # 2. 답변 구성 (상위 3개 법령 요약)
    # ========================================
    answer_parts = []
    
    # 상위 3개만 사용
    top_laws = laws[:3]
    
    for law in top_laws:
        law_name = law.get("law_name", "")
        article_no = law.get("article_no", "")
        # article_title이 있으면 사용, 없으면 생략
        title = f" ({law['article_title']})" if law.get("article_title") else ""
        text = law.get("text", "")[:100] + "..." if law.get("text") else ""
        
        answer_parts.append(f"[{law_name} {article_no}{title}] {text}")

    answer = "\n\n".join(answer_parts)

    # ========================================
    # 3. 컨텍스트 구성 (법령 원문)
    # ========================================
    contexts = []
    for law in top_laws:
        full_text = law.get("text", "")
        if full_text:
            contexts.append(full_text)
            
    # 컨텍스트가 없으면 제목이라도 넣어서 평가 가능하게 함
    if not contexts:
        for law in top_laws:
             contexts.append(f"{law.get('law_name')} {law.get('article_no')}")

    # ========================================
    # 4. 유효성 검증
    # ========================================
    if not contexts or not answer:
        return None

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts
    }


def prepare_precedent_evaluation(
    crime_type: str,
    description: str,
    precedents_result: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    판례 검색 결과를 RAGAS 평가용 데이터로 변환

    Args:
        crime_type: 범죄 유형
        description: 사건 설명
        precedents_result: search_similar_precedents() 반환 결과

    Returns:
        RAGAS 평가용 데이터 또는 None

    Note:
        향후 판례 검색 기능 구현 및 RAGAS 통합 시 구현 예정
    """
    # TODO: 판례 검색 RAGAS 통합 시 구현
    # 현재는 양형기준만 지원
    return None
