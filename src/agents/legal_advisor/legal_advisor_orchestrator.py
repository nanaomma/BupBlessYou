"""
Legal Advisor Orchestrator - 자문/해석기 (모델: GPT 사용)
- 자연어 질문을 이해하고 Retriever를 어떤 방식으로 호출할지 결정
- Retriever 결과를 근거로 답변 생성
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.agents.common.base_agent import BaseAgent, AgentMessage
from src.config.settings import settings

from src.agents.legal_advisor.legal_advisor_retriever import LegalAdvisorRetriever


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json_loads(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None


class LegalAdvisorOrchestrator(BaseAgent):
    """
    자문 에이전트
    - 판사/검사/변호사 에이전트가 자연어로 질문 시
      (1) 조회 계획(plan) 수립
      (2) Retriever를 호출하여 정보 수집
      (3) 모댈(GPT) 기반으로 답변 생성해서 반환
    """

    def __init__(self):
        super().__init__(name="LegalAdvisorOrchestrator", role="legal_advisor_orchestrator")

        self.retriever = LegalAdvisorRetriever()

        self.gpt_model = getattr(settings, "legal_advisor_gpt_model", None) or getattr(settings, "gpt_model", None) or "gpt-4.1-mini"

        try:
            from openai import OpenAI  # type: ignore
            api_key = getattr(settings, "openai_api_key", None) or getattr(settings, "api_key", None)
            base_url = getattr(settings, "openai_base_url", None) or getattr(settings, "base_url", None)

            if base_url:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                self.client = OpenAI(api_key=api_key)
        except Exception:
            self.client = None

    # ---------------------------
    # (1) 질문 → 조회 플랜 생성
    # ---------------------------
    def _build_plan_prompt(self, question: str, case: Optional[Dict[str, Any]]) -> str:
        case_block = ""
        if case:
            case_block = (
                f"\n[CASE]\n"
                f"- casetype: {case.get('casetype')}\n"
                f"- casename: {case.get('casename')}\n"
                f"- description: {str(case.get('description',''))[:300]}\n"
                f"- facts_excerpt: {str(case.get('facts',''))[:300]}\n"
            )

        return f"""
너는 법률자문 시스템의 '조회 계획 생성기'다.
이 시스템은 형법(형사법) 범죄 사건만 다룬다. 민사 사건은 처리하지 않는다.
사용자의 자연어 질문을 보고, Retriever(조회기)를 어떻게 호출할지 계획(plan)을 JSON으로만 출력해라.
절대 설명하지 말고 JSON만 출력해라.

[QUESTION]
{question}
{case_block}

[OUTPUT JSON SCHEMA]
{{
  "crime_type": "string (형법 범죄 유형만: 폭행, 사기, 강제추행, 절도, 살인, 상해, 협박, 감금, 횡령, 배임, 공갈 등. 민사 키워드나 'civil_', 'insurance' 같은 것은 절대 사용하지 말 것)",
  "description_hint": "string (검색에 섞을 요약 힌트, 0~200자)",
  "need_laws": true|false,
  "need_guidelines": true|false,
  "law_top_k": 5,
  "guideline_top_k": 3,
  "guideline_namespaces": ["sentence_criteria","suspended_sentence"]  // 필요시
}}

규칙:
- 질문이 '감경/가중/양형/집행유예' 중심이면 need_guidelines=true
- 질문이 '조문/구성요건/위법성/책임' 중심이면 need_laws=true
- crime_type은 반드시 형법에 있는 범죄 유형만 추출: 폭행, 사기, 강제추행, 절도, 살인, 상해, 협박, 감금, 횡령, 배임, 공갈, 강도, 강간, 명예훼손, 모욕, 방화 등
- 민사 사건 관련 키워드(보험, 손해배상, 계약, 부동산 등)나 'civil_'로 시작하는 단어는 절대 사용하지 말 것
- description_hint는 질문 핵심 키워드로 200자 이내
""".strip()

    def _gpt_make_plan(self, question: str, case: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.client:
            return {"ok": False, "error": "OpenAI client not configured"}

        prompt = self._build_plan_prompt(question, case)

        resp = self.client.chat.completions.create(
            model=self.gpt_model,
            messages=[
                {"role": "system", "content": "You output only valid JSON. No markdown."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        text = (resp.choices[0].message.content or "").strip()
        plan = _safe_json_loads(text)
        if not plan:
            return {"ok": False, "error": "plan_json_parse_failed", "raw": text}

        # 기본값 보강
        plan.setdefault("need_laws", False)
        plan.setdefault("need_guidelines", True)
        plan.setdefault("law_top_k", 5)
        plan.setdefault("guideline_top_k", 3)
        plan.setdefault("guideline_namespaces", ["sentence_criteria", "suspended_sentence"])
        plan.setdefault("description_hint", "")
        plan.setdefault("crime_type", "")

        return {"ok": True, "plan": plan, "raw": text}

    # ---------------------------
    # (2) Retriever 호출
    # ---------------------------
    def _retrieve(self, *, plan: Dict[str, Any], case_id: Optional[str | int], case: Optional[Dict[str, Any]], case_attributes: Optional[List[str]] = None) -> Dict[str, Any]:
        crime_type = plan.get("crime_type") or (case.get("casename") if case else None) or "UNKNOWN"
        desc = plan.get("description_hint") or (case.get("description") if case else "") or ""
        
        # 사건 속성이 있으면 description에 포함
        if case_attributes:
            attributes_text = ", ".join(case_attributes)
            if desc:
                desc = f"{desc} (속성: {attributes_text})"
            else:
                desc = f"속성: {attributes_text}"

        out: Dict[str, Any] = {"crime_type": crime_type, "description": desc}
        if case_attributes:
            out["case_attributes"] = case_attributes

        if plan.get("need_laws"):
            out["laws"] = self.retriever.search_legal_provisions(
                crime_type=crime_type,
                description=desc,
                top_k=int(plan.get("law_top_k", 5)),
            )

        if plan.get("need_guidelines"):
            namespaces: List[str] = plan.get("guideline_namespaces") or [self.retriever.sentence_namespace]
            guides = []
            for ns in namespaces:
                self.retriever.sentence_namespace = ns
                guides.append(
                    {
                        "namespace": ns,
                        "result": self.retriever.get_sentencing_guidelines(
                            crime_type=crime_type,
                            description=desc,  # 속성 정보가 포함된 description
                            top_k=int(plan.get("guideline_top_k", 3)),
                        ),
                    }
                )
            out["guidelines"] = guides

        return out

    # ---------------------------
    # (3) 근거 기반 답변 생성
    # ---------------------------
    def _build_answer_prompt(self, question: str, retrieved: Dict[str, Any]) -> str:
        laws = retrieved.get("laws")
        guides = retrieved.get("guidelines")
        case_attributes = retrieved.get("case_attributes", [])

        return f"""
    너는 모의법정 시뮬레이터에서 '판사 에이전트'를 보조하는 자문/자료제공 담당(재판연구관 역할)이다.
    너의 목적은 판단을 대신하는 것이 아니라, 판사가 판단할 수 있도록 관련 조문/기준/판례(또는 양형기준)를 정리해 제공하는 것이다.

    중요 규칙 (절대 준수):
    1. 제공된 컨텍스트만 사용: 아래 [RETRIEVED]에 포함된 내용만 근거로 작성한다. 외부 지식, 일반 상식, 추측은 절대 사용하지 말 것.
    2. Hallucination 금지: 검색 결과에 없는 조문, 판례, 기준을 만들어내지 말 것. 없다고 말하라.
    3. 인용 필수: 모든 법령/조문/기준 언급 시 반드시 citation_id와 score를 함께 표시하라.
    4. 컨텍스트 재확인: 답변 생성 전에 [RETRIEVED]의 내용을 다시 한 번 확인하고, 그 내용만 사용하라.
    5. 근거 부족 시 명시: 검색 결과가 비었거나(no matches), score가 낮거나(0.6 미만), 서로 충돌하면 반드시 그 사실을 명시한다.
    6. 속성 반영: 사건 속성 정보가 있으면 양형기준 설명에 반영하라 (예: "초범인 경우", "피해 회복된 경우" 등).
    7. 결론 단정 금지: 유/무죄 또는 적용 결론을 단정하지 말고, "검토 포인트" 형태로 판사에게 선택권을 남겨라.

    출력 형식(반드시 이 구조를 지켜라):
    1) [판사용 1문장 브리핑] : 이 질문의 핵심 쟁점 1~2개를 한 문장으로 정리할 것. 출력 시 [판사용 1문장 브리핑] 이라는 제목을 붙이지 말 것.
    2) [관련 법령/조문] : 최대 3개
    - 각 항목마다: (조문명/단위) + 핵심요건 요약(1~2줄) + (citation_id, score)
    - score가 0.6 미만인 경우 "(참고: 관련성 낮음)" 표시
    3) [관련 기준/양형기준] : 최대 3개 (없으면 "해당 없음/추가 근거 필요")
    - 각 항목마다: 기준의 포인트 2~3개 + (citation_id, score)
    {f"- 사건 속성 정보가 있으면 이를 양형기준 설명에 반영하라 (예: '초범인 경우', '피해 회복된 경우' 등)" if case_attributes else ""}
    4) [검토 포인트] : 판사가 판단할 때 확인할 질문 형태로 3~5개
    - 예: "피해자 지위가 법령상 가중/감경 요소로 명시되어 있는가?"

    톤/말투:
    - 존댓말, 정중한 자료제공 톤("정리해드립니다", "참고하실 수 있습니다")
    - 결론 단정 금지("~로 판단됩니다" 금지)

    [QUESTION]
    {question}

    [RETRIEVED]
    {json.dumps({"laws": laws, "guidelines": guides}, ensure_ascii=False)}
    """.strip()

    def _gpt_answer(self, question: str, retrieved: Dict[str, Any]) -> Dict[str, Any]:
        if not self.client:
            return {"ok": False, "error": "OpenAI client not configured"}

        prompt = self._build_answer_prompt(question, retrieved)

        resp = self.client.chat.completions.create(
            model=self.gpt_model,
            messages=[
                {"role": "system", "content": "Answer in Korean. Be conservative. No hallucinations."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        answer = (resp.choices[0].message.content or "").strip()
        return {"ok": True, "answer": answer}

    # ---------------------------
    # 형사 사건 필터링 및 속성 추출
    # ---------------------------
    def _validate_criminal_case(self, case: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        형사 사건 여부 검증 및 사건 속성 추출
        
        Returns:
            {"valid": bool, "error": str | None, "attributes": List[str]}
        """
        if not case:
            return {"valid": True, "error": None, "attributes": []}  # case가 없으면 통과
        
        casetype = case.get("casetype", "").lower() if case.get("casetype") else ""
        
        # 형사 사건이 아니면 에러 반환
        if casetype != "criminal":
            return {
                "valid": False,
                "error": f"이 시스템은 형사 사건만 처리합니다. 현재 사건 유형: {case.get('casetype')}",
                "attributes": []
            }
        
        # 사건 속성 추출 (전과 여부, 피해자 관계, 범행 동기 등)
        attributes = []
        facts = case.get("facts", "") or ""
        description = case.get("description", "") or ""
        combined_text = f"{facts} {description}".lower()
        
        # 전과 여부 추출
        if any(keyword in combined_text for keyword in ["초범", "처음", "전과 없음", "범죄 경력 없음", "범죄 이력 없음"]):
            attributes.append("초범")
        elif any(keyword in combined_text for keyword in ["재범", "전과", "범죄 경력", "유죄 판결"]):
            attributes.append("재범")
        
        # 피해자 관계 추출
        if any(keyword in combined_text for keyword in ["친족", "가족", "배우자", "직계존속", "형제"]):
            attributes.append("친족 관계")
        elif any(keyword in combined_text for keyword in ["지인", "아는 사람", "동료", "친구"]):
            attributes.append("지인 관계")
        
        # 범행 동기 추출
        if any(keyword in combined_text for keyword in ["갈등", "다툼", "싸움", "시비"]):
            attributes.append("갈등 기반")
        elif any(keyword in combined_text for keyword in ["금전", "재물", "이익", "돈"]):
            attributes.append("금전 목적")
        elif any(keyword in combined_text for keyword in ["우발", "갑작스럽", "순간적"]):
            attributes.append("우발적")
        
        # 피해 회복 여부
        if any(keyword in combined_text for keyword in ["합의", "피해 회복", "사과", "배상", "조정"]):
            attributes.append("피해 회복")
        
        return {"valid": True, "error": None, "attributes": attributes}

    # ---------------------------
    # 외부 호출: 자연어 질문 처리
    # ---------------------------
    def answer_question(
        self,
        *,
        question: str,
        case_id: Optional[str | int] = None,
        include_case: bool = True,
    ) -> Dict[str, Any]:
        case = None
        if include_case and case_id is not None:
            case = self.retriever.get_case_info(case_id)
        
        # 형사 사건 필터링
        validation = self._validate_criminal_case(case if isinstance(case, dict) else None)
        if not validation["valid"]:
            return {
                "ok": False,
                "stage": "validation",
                "error": validation["error"],
                "retrieved_at": _now_iso()
            }
        
        case_attributes = validation["attributes"]

        plan_res = self._gpt_make_plan(question, case if isinstance(case, dict) else None)
        if not plan_res.get("ok"):
            return {"ok": False, "stage": "plan", **plan_res, "retrieved_at": _now_iso()}

        plan = plan_res["plan"]
        
        # crime_type 검증 및 정제 (민사 키워드 제거)
        crime_type = plan.get("crime_type", "") or ""
        if crime_type:
            # 민사 사건 키워드 필터링
            civil_keywords = ["civil", "insurance", "claim", "contract", "property", "보험", "손해배상", "계약", "부동산"]
            crime_type_lower = crime_type.lower()
            if any(keyword in crime_type_lower for keyword in civil_keywords):
                # 민사 키워드가 포함되어 있으면 casename이나 기본값 사용
                if case and isinstance(case, dict):
                    fallback_type = case.get("casename") or "UNKNOWN"
                    if fallback_type and fallback_type.lower() not in [k for k in civil_keywords]:
                        plan["crime_type"] = fallback_type
                    else:
                        plan["crime_type"] = "UNKNOWN"
                else:
                    plan["crime_type"] = "UNKNOWN"
        
        # 사건 속성을 description_hint에 포함
        if case_attributes:
            attributes_text = ", ".join(case_attributes)
            current_hint = plan.get("description_hint", "") or ""
            if current_hint:
                plan["description_hint"] = f"{current_hint} (사건 속성: {attributes_text})"
            else:
                plan["description_hint"] = f"사건 속성: {attributes_text}"

        retrieved = self._retrieve(plan=plan, case_id=case_id, case=case if isinstance(case, dict) else None, case_attributes=case_attributes)
        ans_res = self._gpt_answer(question, retrieved)
        if not ans_res.get("ok"):
            return {"ok": False, "stage": "answer", **ans_res, "plan": plan, "retrieved": retrieved, "retrieved_at": _now_iso()}

        return {
            "ok": True,
            "question": question,
            "case_id": case_id,
            "plan": plan,
            "retrieved": retrieved,
            "answer": ans_res["answer"],
            "retrieved_at": _now_iso(),
        }

    # ---------------------------
    # BaseAgent 호환 진입점
    # ---------------------------
    async def generate_response(self, case_info: Dict[str, Any], context: Dict[str, Any]) -> AgentMessage:
        """
        예: 판사 에이전트가
        - case_info: {"id": 10200}
        - context: {"question": "폭행인데 가해자가 장애인이면 감형?"}
        로 호출하면 여기서 처리 가능
        """
        question = (context.get("question") or "").strip()
        if not question:
            return AgentMessage(role=self.role, content="question이 없습니다.", metadata={"ok": False})

        case_id = case_info.get("id") or case_info.get("case_id")
        result = self.answer_question(question=question, case_id=case_id, include_case=True)

        return AgentMessage(role=self.role, content=json.dumps(result, ensure_ascii=False), metadata={"ok": result.get("ok", False)})


if __name__ == "__main__":
    # 간단 수동 테스트
    orch = LegalAdvisorOrchestrator()
    q = "폭행 사건인데 가해자가 장애인이면 감형 사유가 되나?"
    print(json.dumps(orch.answer_question(question=q, case_id=None), ensure_ascii=False, indent=2))

