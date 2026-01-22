import json
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, AIMessage

from src.config.settings import settings
from src.utils.llm_factory import create_llm
from src.agents.common.state import CourtSimulationState, Role


"""
src/agents/judge/judge_agent.py
Judge Agent - 재판장 페르소나 (진행, 평가, 채점 담당)
"""

# 구조화된 출력을 위한 Pydantic 모델 정의
class SuggestedQuestions(BaseModel):
    questions: List[str] = Field(description="판사가 던질 수 있는 날카로운 후속 질문 3가지")

class EvaluationResult(BaseModel):
    turn : int = Field(description = "현재 Round")
    speaker : str = Field(description="발언자")
    score: int = Field(description="논리적 설득력 점수 (0-10점)")
    fact_check: str = Field(description="사실관계 부합 여부 평가와 이유")
    logical_flaw: str = Field(description="논리적 허점이나 비약 지적과 이유")
    feedback: str = Field(description="짧은 한 줄 평가")

class RoundSummary(BaseModel):
    prosecutor_summary: str
    defense_summary: str
    prosecutor_score: float
    defense_score: float
    verdict: str = Field(description="prosecutor | defense | tie")
    reason: str = Field(description="종합 판단 이유 (1~2문장)")


class JudgeAgent:
    """
    판사 에이전트 (Judge Agent)
    역할:
    1. 유저 개입 지원 (질문 추천) - [Flow Support]
    2. 실시간 검사/변호사 공방 평가 (Argument Evaluation) - [Feature 1]
    3. 유저 판결 정확도 채점 (Verdict Scoring) - [Feature 2]
    """

    def __init__(self):
        # 평가와 분석은 정확도가 중요하므로 temperature를 낮게 설정
        self.llm = create_llm(temperature=0.1)
        # 질문 생성은 창의성이 필요하므로 약간 높게 설정할 수 있음 (여기선 인스턴스 공유)
        self.creative_llm = create_llm(temperature=0.7)
    
    
    # =================================================================
    # [Role 1 / Feature 1] 배치 단위 공방 평가
    # =================================================================

    async def batch_evaluate_argument(self, state: CourtSimulationState) -> List[Dict[str, Any]]:
        """
        최근 검사/변호사 발언을 abatch를 사용하여 병렬로 일괄 평가합니다.
        """
        recent = state.get("messages", [])[-5:]
        round_num = state.get("judge_round", 1)
        
        # 평가를 수행할 Chain 준비 (반복문 밖에서 정의)
        system_prompt = """당신은 공정하고 엄격한 재판관입니다.
        제시된 [사건 개요]와 [법률 정보]를 기준으로, 현재 발언자의 주장을 평가하세요.
        
        평가 기준:
        1. Fact Check: 사건 개요와 모순되는 내용을 날조하지 않았는가?
        2. Legal Logic: 인용된 법령이나 양형기준, 판례가 상황에 적절한가?
        3. Persuasiveness: 논리가 명확하고 설득력이 있는가?
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        # Structured Output
        structured_llm = self.llm.with_structured_output(EvaluationResult)
        evaluation_chain = prompt | structured_llm

        # 입력 데이터(Batch Inputs)와 메타데이터 준비
        batch_inputs = []
        meta_data_list = [] # 결과와 매칭할 발화자/라운드 정보 저장용

        case_summary = state.get('case_summary', '정보 없음')
        legal_context = state.get('legal_context', '정보 없음')

        for msg in recent:
            try:
                if isinstance(msg.content, str):
                    try:
                        data = json.loads(msg.content)
                    except json.JSONDecodeError:
                        continue 
                else:
                    data = msg.content 

                role = data.get("role")
                content = data.get("content")

                if role not in [Role.PROSECUTOR.value, Role.DEFENSE.value]:
                    continue

                # LLM에 들어갈 프롬프트 내용 구성
                input_content = f"""
                    [사건 개요]
                    {case_summary}

                    [법률 정보 (RAG)]
                    {legal_context}

                    [현재 발언자]: {role}
                    [발언 내용]: {content}
                    """
                
                batch_inputs.append({"input": input_content})
                meta_data_list.append({"role": role, "round": round_num})

            except Exception as e:
                print(f"메시지 전처리 중 오류: {e}")
                continue

        # 평가할 내용이 없으면 빈 리스트 반환
        if not batch_inputs:
            return []
        
        try:
            # 병렬 실행
            eval_results = await evaluation_chain.abatch(batch_inputs)
        except Exception as e:
            print(f"Batch Evaluation 전체 실패: {e}")
            return []

        final_results = []
        for meta, result in zip(meta_data_list, eval_results):
            # result: EvaluationResult 객체
            result_dict = result.model_dump()
            
            final_results.append({
                "round": meta["round"],
                "speaker": meta["role"],
                "score": result_dict.get("score", 0),
                "fact_check": result_dict.get("fact_check", ""),
                "logical_flaw": result_dict.get("logical_flaw", ""),
                "feedback": result_dict.get("feedback", "")
            })

        return final_results
    
    async def generate_round_summary(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        발언별 평가(evaluations)를 바탕으로
        라운드 종합 평가 생성
        """
        system_prompt = """
            당신은 형사 재판의 재판장입니다.

            아래에는 특정 라운드에서 오간 검사(Prosecutor)와 변호인(Defense)의 발언에 대한 개별 평가 기록이 주어집니다.

            각 평가 항목은 다음 의미를 가집니다:
            - speaker: 발언 주체 (prosecutor | defense)
            - score: 해당 발언의 설득력 점수 (0~10)
            - fact_check: 사실관계 적합성 평가
            - logical_flaw: 논리적 허점
            - feedback: 판사의 코멘트

            [당신의 임무]
            1. 검사 측 발언들을 종합하여 강점과 한계를 요약
            2. 변호사 측 발언들을 종합하여 강점과 한계를 요약
            3. 각 측의 평균 설득력 점수를 계산
            4. 이번 라운드에서 어느 측 주장이 더 설득력 있었는지 판단하고, 그 이유를 1~2문장으로 명확히 설명

            발언 수가 많을 경우, 반복되는 주장보다는 핵심 논점 위주로 종합하십시오.
            판결이 아닌 “공방 평가”임을 명심하십시오.
            """

        input_content = json.dumps(evaluations, ensure_ascii=False, indent=2)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "다음은 이번 라운드 발언별 평가 기록입니다:\n{input}")
        ])

        structured_llm = self.llm.with_structured_output(RoundSummary)
        chain = prompt | structured_llm

        result: RoundSummary = await chain.ainvoke({"input": input_content})
        return result.model_dump()
    

    # =================================================================
    # [Role 2] 유저 개입 질문 추천
    # =================================================================
    async def suggest_questions(self, state: CourtSimulationState) -> List[Dict[str, str]]:
        """
        현재까지의 공방 내용을 바탕으로 유저(판사)가 할 만한 질문 3가지를 추천합니다.
        """
        # 최근 대화 요약 (마지막 6턴)
        history_text = "\n".join([
            f"{m.type}: {m.content[:200]}" 
            for m in state.get("messages", [])[-6:]
        ])

        system_prompt = (
            "당신은 재판장을 보좌하는 베테랑 AI 판사입니다. "
            "현재 진행 중인 재판의 흐름을 파악하고, 재판장(사용자)이 검사나 변호인에게 "
            "확인해야 할 핵심적인 쟁점 질문 3가지를 한국어로 제안해주세요. "
            "질문은 20자 내외로 간결하고 날카로워야 합니다."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "대화 기록:\n{history}")
        ])

        # Structured Output 사용
        structured_llm = self.creative_llm.with_structured_output(SuggestedQuestions)
        chain = prompt | structured_llm

        try:
            result = await chain.ainvoke({"history":history_text})
            questions = result.questions
        except Exception as e:
            # 실패 시 기본 질문 제공
            print(f"[JudgeAgent] 질문 생성 실패: {e}")
            questions = ["피고인의 평소 행실은?", "피해자와 합의되었습니까?", "반성의 기미가 있습니까?"]

        # UI에서 사용할 후속 질문 선택지 포맷으로 변환
        return questions
    
    
    # -------------------------------------------------
    # 유저 개입 핸들러 (node_user_judge)
    # -------------------------------------------------

    async def handle_user_judge(self, state: CourtSimulationState) -> dict:
        """
        graph.py의 node_user_judge에서 직접 호출
        return 구조까지 전부 책임짐
        """
        evaluations = await self.batch_evaluate_argument(state)
        questions = await self.suggest_questions(state)
    
        
        # 평가 결과 누적
        accumulated = state.get("evaluations", [])
        updated = accumulated + evaluations
        round_summary = await self.generate_round_summary(evaluations)

        judge_message = {
            "role": Role.JUDGE.value,
            "content": (
                "지금까지의 심리 내용을 검토하였습니다. "
                "재판장으로서 추가로 확인하고 싶은 사항을 질문하시거나, "
                "충분한 심리가 이루어졌다고 판단되면 최종 판결을 선고하실 수 있습니다."
            ),
            "emotion": "neutral",
            "references": []
        }

        return {
            "current_phase": "user_judge",
            "phase_turn": 2,
            "choices": [
                {"id": f"q_{i}", "label": q, "value": q}
                for i, q in enumerate(questions)
            ],
            "evaluations": evaluations,
            "evaluations_log": updated,
            "round_summary": round_summary,
            "messages": [
                AIMessage(content=json.dumps(judge_message, ensure_ascii=False))
            ]
        }

    

    # # =================================================================
    # # [Role 3 / Feature 2] 판결 정확도 채점 (vs LBox Data)
    # # =================================================================
    # async def score_user_verdict(self, state: CourtSimulationState) -> Dict[str, Any]:
    #     """
    #     유저 판결과 실제 데이터(LBox)를 비교하여 점수와 피드백을 생성합니다.
    #     """
    #     actual_data = state.get("actual_case_data", {})
        
    #     # 실제 데이터가 없는 경우 (커스텀 케이스 등) 예외 처리
    #     if not actual_data or not actual_data.get("actual_label"):
    #         return {
    #             "total_score": 0,
    #             "comparison_analysis": "비교할 실제 판결 데이터가 없습니다. (커스텀 사건인 경우 정상)"
    #         }

    #     system_prompt = """당신은 판결 분석 AI입니다.
    #     사용자(User Judge)가 내린 판결과 실제 법원의 판결(Actual Ruling)을 정밀 비교하여 채점하세요.

    #     [채점 기준 - 총 100점]
    #     1. **형량 일치도 (50점)**: 
    #        - 집행유예 여부, 징역 기간 등이 유사하면 고득점.
    #        - 실형 vs 집행유예 차이는 큰 감점 요인.
    #     2. **논리 유사도 (50점)**: 
    #        - 양형 사유(감경/가중 요소) 판단이 실제 판결문과 유사한가?

    #     분석 코멘트는 사용자가 무엇을 놓쳤는지, 어떤 점이 훌륭했는지 구체적으로 작성하세요.
    #     """

    #     input_content = f"""
    #     [사건 개요]
    #     {state.get('case_summary')}

    #     [사용자 판결]
    #     - 형량: {state.get('user_verdict')}
    #     - 이유: {state.get('user_reasoning')}

    #     [실제 판결 (정답)]
    #     - 실제 형량: {actual_data.get('actual_label')}
    #     - 실제 양형 이유: {actual_data.get('actual_reason')}
    #     """

    #     prompt = ChatPromptTemplate.from_messages([
    #         ("system", system_prompt),
    #         ("human", input_content)
    #     ])

    #     structured_llm = self.llm.with_structured_output(VerdictScore)
    #     chain = prompt | structured_llm

    #     try:
    #         result: VerdictScore = await chain.ainvoke({})
    #         return result.model_dump()
    #     except Exception as e:
    #         print(f"[JudgeAgent] 채점 실패: {e}")
    #         return {
    #             "total_score": 0,
    #             "verdict_match_score": 0,
    #             "reasoning_match_score": 0,
    #             "comparison_analysis": "채점 중 오류가 발생했습니다."
    #         }