"""
Facts Guardrail - 사실 관계 기반 가드레일

변호사/검사 에이전트가 확인된 사실만 사용하도록 강제하는 시스템
Hallucination 방지 및 사실 관계 검증
"""
from typing import Dict, Any, List
from src.agents.common.state import CourtSimulationState, CaseAttribute, LegalContext
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FactsGuardrail:
    """
    사실 관계 검증 및 추출 클래스

    Purpose:
        LLM이 확인된 사실만 사용하도록 강제하여 Hallucination 방지
    """

    @staticmethod
    def extract_verified_facts(state: CourtSimulationState) -> Dict[str, Any]:
        """
        State에서 검증된 사실만 추출

        Args:
            state: 현재 법정 시뮬레이션 상태

        Returns:
            검증된 사실 딕셔너리:
            {
                "case_summary": str,
                "verified_attributes": Dict[str, Dict],
                "legal_basis": Dict[str, List],
                "sentencing_factors": Dict[str, List]
            }
        """
        # 1. Case Summary (사건 개요)
        case_summary = state.get("case_summary", "")

        # 2. Verified Attributes (검증된 피고인 속성)
        case_attributes = state.get("case_attributes", [])
        verified_attributes = {}

        for attr in case_attributes:
            key = attr.get("key", "")
            value = attr.get("value")
            description = attr.get("description", "")

            if key:
                verified_attributes[key] = {
                    "value": value,
                    "description": description,
                    "verified": True,
                    "type": type(value).__name__
                }

        # 3. Legal Basis (법률 근거)
        legal_context = state.get("legal_context", {})
        legal_basis = {
            "laws": legal_context.get("relevant_laws", []),
            "guidelines": legal_context.get("sentencing_guidelines", []),
            "precedents": legal_context.get("similar_precedents_summary", "")
        }

        # 4. Sentencing Factors (양형 인자 - 구조화된 정보)
        sentencing_factors = FactsGuardrail._extract_sentencing_factors(
            case_attributes, legal_context
        )

        return {
            "case_summary": case_summary,
            "verified_attributes": verified_attributes,
            "legal_basis": legal_basis,
            "sentencing_factors": sentencing_factors
        }

    @staticmethod
    def _extract_sentencing_factors(
        attributes: List[CaseAttribute],
        legal_context: LegalContext
    ) -> Dict[str, List[str]]:
        """
        양형 인자 추출 (가중/감경 요소)

        Returns:
            {
                "aggravating": ["계획적 범행", "피해액 거액"],
                "mitigating": ["초범", "반성"]
            }
        """
        aggravating = []
        mitigating = []

        # Case attributes에서 양형 인자 추출
        for attr in attributes:
            key = attr.get("key", "")
            value = attr.get("value")
            desc = attr.get("description", "")

            # 가중 요소
            if key in ["planned_crime", "multiple_victims", "large_damage"] and value:
                aggravating.append(desc)

            # 감경 요소
            if key in ["first_offender", "remorse", "victim_agreement"] and value:
                mitigating.append(desc)

        # Legal context에서 양형 기준 추출
        guidelines = legal_context.get("sentencing_guidelines", [])
        for guideline in guidelines:
            if isinstance(guideline, dict):
                # 구조화된 양형기준에서 factors 추출
                factors = guideline.get("factors", {})
                if factors:
                    agg = factors.get("aggravating", [])
                    mit = factors.get("mitigating", [])
                    aggravating.extend(agg)
                    mitigating.extend(mit)

        return {
            "aggravating": list(set(aggravating)),  # 중복 제거
            "mitigating": list(set(mitigating))
        }

    @staticmethod
    def create_facts_guard_prompt(facts: Dict[str, Any]) -> str:
        """
        사실 관계 가드레일 프롬프트 생성

        Args:
            facts: extract_verified_facts()의 반환값

        Returns:
            구조화된 사실 관계 프롬프트 (LLM에 주입)
        """
        prompt_parts = []

        # ==========================================
        # 헤더: 중요 경고
        # ==========================================
        prompt_parts.append("=" * 60)
        prompt_parts.append("⚠️  **[중요: 사실 관계 준수 필수]**")
        prompt_parts.append("=" * 60)
        prompt_parts.append("")
        prompt_parts.append("❌ **금지**: 아래에 없는 사실을 상상하거나 추측하지 마세요.")
        prompt_parts.append("✅ **허용**: 오직 아래 정보만 사용하세요.")
        prompt_parts.append("")

        # ==========================================
        # 1. 사건 개요
        # ==========================================
        prompt_parts.append("## 📋 사건 개요 (Case Summary)")
        prompt_parts.append(facts["case_summary"])
        prompt_parts.append("")

        # ==========================================
        # 2. 확인된 피고인 속성
        # ==========================================
        verified_attrs = facts["verified_attributes"]
        prompt_parts.append("## ✅ 확인된 피고인 속성 (Verified Attributes)")
        prompt_parts.append("**이것만 사용 가능합니다:**")
        prompt_parts.append("")

        if not verified_attrs:
            prompt_parts.append("- (확인된 속성 없음)")
        else:
            for key, attr_info in verified_attrs.items():
                value = attr_info["value"]
                desc = attr_info["description"]
                attr_type = attr_info["type"]

                # 값 타입에 따라 포맷팅
                if attr_type == "bool":
                    status = "✓ 해당함" if value else "✗ 해당 안 됨"
                    prompt_parts.append(f"- **{desc}**: {status}")
                elif attr_type in ["int", "float"]:
                    prompt_parts.append(f"- **{desc}**: {value:,} (확인된 수치)")
                else:
                    prompt_parts.append(f"- **{desc}**: {value}")

        prompt_parts.append("")

        # ==========================================
        # 3. 양형 인자 (가중/감경 요소)
        # ==========================================
        sentencing_factors = facts["sentencing_factors"]
        prompt_parts.append("## ⚖️  양형 인자 (Sentencing Factors)")
        prompt_parts.append("")

        # 가중 요소
        aggravating = sentencing_factors.get("aggravating", [])
        prompt_parts.append("**가중 처벌 요소** (검사 유리):")
        if aggravating:
            for factor in aggravating:
                prompt_parts.append(f"  🔺 {factor}")
        else:
            prompt_parts.append("  - (없음)")
        prompt_parts.append("")

        # 감경 요소
        mitigating = sentencing_factors.get("mitigating", [])
        prompt_parts.append("**감경 요소** (변호사 유리):")
        if mitigating:
            for factor in mitigating:
                prompt_parts.append(f"  🔻 {factor}")
        else:
            prompt_parts.append("  - (없음)")
        prompt_parts.append("")

        # ==========================================
        # 4. 법률 근거
        # ==========================================
        legal_basis = facts["legal_basis"]
        prompt_parts.append("## 📚 법률 근거 (Legal Basis)")
        prompt_parts.append("")

        # 관련 법령
        laws = legal_basis.get("laws", [])
        if laws:
            prompt_parts.append("**관련 법령**:")
            for law in laws[:3]:  # 최대 3개
                if isinstance(law, dict):
                    law_name = law.get("law_name", "")
                    article = law.get("article_no", "")
                    summary = law.get("summary", "")
                    prompt_parts.append(f"  - {law_name} {article}: {summary}")
                else:
                    prompt_parts.append(f"  - {law}")
            prompt_parts.append("")

        # 양형 기준
        guidelines = legal_basis.get("guidelines", [])
        if guidelines:
            prompt_parts.append("**양형 기준**:")
            for guideline in guidelines[:2]:  # 최대 2개
                if isinstance(guideline, dict):
                    name = guideline.get("guideline_name", "")
                    summary = guideline.get("summary", "")
                    prompt_parts.append(f"  - {name}: {summary}")
                else:
                    prompt_parts.append(f"  - {guideline}")
            prompt_parts.append("")

        # 유사 판례
        precedents = legal_basis.get("precedents", "")
        if precedents:
            prompt_parts.append("**유사 판례 경향**:")
            prompt_parts.append(f"  {precedents}")
            prompt_parts.append("")

        # ==========================================
        # 푸터: 재차 경고
        # ==========================================
        prompt_parts.append("=" * 60)
        prompt_parts.append("⚠️  **경고: 위에 명시되지 않은 사실은 절대 언급 금지!**")
        prompt_parts.append("=" * 60)
        prompt_parts.append("")
        prompt_parts.append("**금지 사항 예시**:")
        prompt_parts.append("❌ '피고인은 과거에도 유사한 범행을 저질렀습니다' (확인 안 됨)")
        prompt_parts.append("❌ '피해자는 노인이었습니다' (사건 개요에 없음)")
        prompt_parts.append("❌ '피고인은 범행 후 도주했습니다' (Case attributes에 없음)")
        prompt_parts.append("")
        prompt_parts.append("**올바른 주장 예시**:")
        prompt_parts.append("✅ '피고인의 피해액이 5천만원으로 확인되었습니다' (Verified)")
        prompt_parts.append("✅ '계획적 범행이라는 점은 가중 요소입니다' (Sentencing Factors)")
        prompt_parts.append("✅ '초범이라는 점은 참작할 여지가 있습니다' (Mitigating Factor)")

        return "\n".join(prompt_parts)

    @staticmethod
    def validate_argument(
        argument: str,
        facts: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        생성된 주장이 사실 관계를 준수하는지 검증 (향후 확장용)

        Args:
            argument: LLM이 생성한 주장
            facts: 검증된 사실 딕셔너리

        Returns:
            {
                "valid": bool,
                "violations": List[str],
                "warnings": List[str]
            }

        Note:
            현재는 Placeholder. 향후 NLI 모델 또는 규칙 기반 검증 추가 가능
        """
        # TODO: 향후 구현
        # - NLI (Natural Language Inference) 모델로 모순 검증
        # - 규칙 기반: 키워드 매칭으로 미확인 사실 탐지
        # - RAGAS faithfulness score로 사실 충실도 측정

        return {
            "valid": True,  # 현재는 항상 통과
            "violations": [],
            "warnings": []
        }
