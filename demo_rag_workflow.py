import os
import sys
import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import warnings

# --- 1. 환경 설정 및 라이브러리 로드 ---
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from pinecone import Pinecone
from opentelemetry import trace

# Issue 3 Fix: Arize/Phoenix의 공식 등록 함수 사용
try:
    from arize.otel import register as register_arize
except ImportError:
    print("❌ 'arize-otel' 패키지를 찾을 수 없습니다. pip install arize-otel")
    sys.exit(1)

# RAGAS
from ragas import evaluate, EvaluationDataset, SingleTurnSample
# Use ragas.metrics.collections for v0.4
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import json
from src.services.case_service import get_case_by_id

# 콘솔 출력용 색상
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    CYAN = '\033[36m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# 환경변수 로드
load_dotenv()

# --- 설정값 검증 ---
REQUIRED_KEYS = [
    "OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME",
    "SENTENCE_PINECONE_INDEX_NAME", "PHOENIX_SPACE_ID", "PHOENIX_API_KEY", "LANGSMITH_PROJECT"
]

def check_env():
    missing = [key for key in REQUIRED_KEYS if not os.getenv(key)]
    if missing:
        print(f"{Colors.FAIL}❌ 필수 환경변수가 없습니다: {', '.join(missing)}{Colors.ENDC}")
        sys.exit(1)
    print(f"{Colors.GREEN}✅ 환경변수 확인 완료{Colors.ENDC}")

# --- 2. Arize (Phoenix) Tracer 설정 (개선) ---
def setup_arize_tracer():
    """Arize Phoenix의 공식 register 함수를 사용하여 Tracer를 안전하게 설정"""
    print(f"\n{Colors.BLUE}📡 [Observability] Arize Phoenix 설정 중...{Colors.ENDC}")
    try:
        register_arize(
            space_id=os.getenv("PHOENIX_SPACE_ID"),
            api_key=os.getenv("PHOENIX_API_KEY"),
            project_name=os.getenv("LANGSMITH_PROJECT") # LangSmith와 프로젝트명 통일
        )
        print(f"{Colors.GREEN}   -> Arize Phoenix 연결 성공!{Colors.ENDC}")
        return trace.get_tracer("demo_rag_workflow")
    except Exception as e:
        print(f"{Colors.FAIL}   -> Arize Phoenix 연결 실패: {e}{Colors.ENDC}")
        return None

# --- 3. Pinecone 검색 (RAG) ---
def search_pinecone_index(
    tracer, pc: Pinecone, index_name: str, namespace: str, query_vector: List[float], 
    top_k: int = 3, filter_dict: Optional[Dict] = None, category: str = "General"
) -> List[str]:
    print(f"   -> [{category}] Index '{index_name}' (NS: {namespace}) 검색 중...")
    if filter_dict:
        print(f"      Filter: {filter_dict}")
    
    with tracer.start_as_current_span(f"pinecone_search_{category}") as span:
        span.set_attribute("pinecone.index", index_name)
        span.set_attribute("pinecone.namespace", namespace)
        span.set_attribute("pinecone.top_k", top_k)
        if filter_dict:
            span.set_attribute("pinecone.filter", str(filter_dict))
            
        try:
            index = pc.Index(index_name)
            results = index.query(
                vector=query_vector, namespace=namespace, filter=filter_dict,
                top_k=top_k, include_metadata=True
            )
            contexts = []
            for match in results['matches']:
                meta = match['metadata']
                text = meta.get('text') or meta.get('raw_text') or str(meta)
                score = match['score']
                contexts.append(f"[{category}] {text}")
                print(f"      - {category} 문서(Score {score:.4f}): {text[:50]}...")
            
            span.set_attribute("pinecone.result_count", len(contexts))
            if not contexts:
                print(f"      ⚠️ 검색된 문서가 없습니다 (Score < 0.6 or No matches).")
            return contexts
        except Exception as e:
            print(f"{Colors.FAIL}      ⚠️ 검색 실패: {e}{Colors.ENDC}")
            span.record_exception(e)
            return []

def perform_dual_search(tracer, query: str, crime_code: Optional[str] = None) -> List[str]:
    print(f"\n{Colors.BLUE}🔍 [RAG] 듀얼 벡터 검색 시작 (법령 + 양형기준)...{Colors.ENDC}")
    
    with tracer.start_as_current_span("Step 1: Retrieval (Dual Search)") as span:
        current_trace_id = span.get_span_context().trace_id
        print(f"   -> [Trace ID: {current_trace_id:x}] Retrieval Step Started")
        
        # Standard Attributes for Input/Output Tab
        span.set_attribute("input.value", query)
        
        # Custom Attributes
        span.set_attribute("rag.query", query)
        
        openai_client = OpenAI()
        print(f"   -> 쿼리 임베딩 생성: '{query}'")
        embedding_resp = openai_client.embeddings.create(input=query, model="text-embedding-3-small")
        query_vector = embedding_resp.data[0].embedding
        
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        all_contexts = []
        
        law_index = os.getenv("PINECONE_INDEX_NAME") or "bupblessyou-judgments"
        law_namespace = os.getenv("LAW_PINECONE_NAMESPACE") or "law_statue_criminal"
        
        # Pass tracer to search_pinecone_index
        laws = search_pinecone_index(
            tracer, pc, index_name=law_index, namespace=law_namespace, query_vector=query_vector,
            top_k=2, category="법령", filter_dict=None
        )
        all_contexts.extend(laws)
        
        if crime_code:
            sentence_index = os.getenv("SENTENCE_PINECONE_INDEX_NAME") or "bupblessyou-sentence-v1"
            sentence_namespace = os.getenv("SENTENCE_PINECONE_NAMESPACE") or "sentence_criteria"
            guidelines = search_pinecone_index(
                tracer, pc, index_name=sentence_index, namespace=sentence_namespace, query_vector=query_vector,
                top_k=2, filter_dict={"crime_number": crime_code}, category="양형기준"
            )
            all_contexts.extend(guidelines)
            
        print(f"   -> 총 {len(all_contexts)}개 유효 문서 발견")
        span.set_attribute("rag.total_documents", len(all_contexts))
        
        # Standard Attribute for Output (Convert list to string representation)
        import json
        span.set_attribute("output.value", json.dumps(all_contexts, ensure_ascii=False))
        
        return all_contexts

# --- 4. LLM 답변 생성 ---
def generate_answer(tracer, query: str, contexts: List[str]) -> str:
    print(f"\n{Colors.BLUE}🤖 [LLM] 답변 생성 중...{Colors.ENDC}")
    
    with tracer.start_as_current_span("Step 2: LLM Generation") as span:
        current_trace_id = span.get_span_context().trace_id
        print(f"   -> [Trace ID: {current_trace_id:x}] Generation Step Started")
        
        client = OpenAI()
        if not contexts:
            return "죄송합니다. 관련 정보를 찾을 수 없습니다."
            
        context_text = "\n\n".join(contexts)
        system_prompt = "당신은 유능한 법률 조력자입니다. 주어진 [법령]와 [양형기준] 정보를 종합하여 질문에 대해 논리적이고 명확하게 답변하세요. 출처(법령, 양형기준)를 명시하면 더 좋습니다."
        user_prompt = f"질문: {query}\n\n[참고 자료]\n{context_text}"
        
        # Standard Attributes for Input
        span.set_attribute("input.value", user_prompt)
        import json
        span.set_attribute("llm.input_messages", json.dumps([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], ensure_ascii=False))

        # Custom Attributes
        span.set_attribute("llm.system_prompt", system_prompt)
        span.set_attribute("llm.user_prompt", user_prompt)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0,
            max_tokens=4000 # Increase output limit to avoid truncation
        )
        answer = response.choices[0].message.content
        print(f"   -> 생성된 답변 미리보기:\n{Colors.CYAN}{answer[:150]}...{Colors.ENDC}")
        
        # Standard Attribute for Output
        span.set_attribute("output.value", answer)
        span.set_attribute("llm.response", answer)
        
        return answer

# --- 5. RAGAS 평가 및 Arize 전송 ---
async def evaluate_and_log(tracer, query: str, answer: str, contexts: List[str], scenario_name: str, reference: Optional[str] = None):
    print(f"\n{Colors.WARNING}⚖️ [Evaluation] '{scenario_name}' RAGAS 평가 시작...{Colors.ENDC}")
    if not contexts:
        print(f"{Colors.FAIL}❌ 컨텍스트가 없어 평가를 건너뜁니다.{Colors.ENDC}")
        return
    if not tracer:
        print(f"{Colors.FAIL}❌ Tracer가 없어 평가 및 전송을 건너뜁니다.{Colors.ENDC}")
        return

    # Create Ragas LLM and Embeddings using factory (requires AsyncOpenAI client for async execution)
    eval_openai_client = AsyncOpenAI()
    ragas_llm = llm_factory(model="gpt-4o-mini", client=eval_openai_client)
    ragas_embeddings = embedding_factory(model="text-embedding-3-small", client=eval_openai_client)
    
    # Instantiate metrics with Ragas LLM
    f_metric = Faithfulness(llm=ragas_llm)
    ar_metric = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    cp_metric = ContextPrecision(llm=ragas_llm)
    
    with tracer.start_as_current_span(f"Step 3: RAGAS Evaluation") as span:
        current_trace_id = span.get_span_context().trace_id
        print(f"   -> [Trace ID: {current_trace_id:x}] Evaluation Step Started")
        
        start_time = time.time()
        
        # Run manual evaluation using ascore (async)
        try:
            # Faithfulness
            f_result = await f_metric.ascore(
                user_input=query,
                response=answer,
                retrieved_contexts=contexts
            )
            faith_score = f_result.value
            
            # Answer Relevancy
            ar_result = await ar_metric.ascore(
                user_input=query,
                response=answer
            )
            relevancy_score = ar_result.value
            
            # Context Precision (Requires reference if strict, but let's try passing what we have)
            cp_score = 0.0
            if reference:
                cp_result = await cp_metric.ascore(
                    user_input=query,
                    retrieved_contexts=contexts,
                    reference=reference
                )
                cp_score = cp_result.value
            
            scores = {
                "faithfulness": faith_score,
                "answer_relevancy": relevancy_score,
                "context_precision": cp_score
            }
        except Exception as e:
            print(f"{Colors.FAIL}⚠️ 평가 실패: {e}{Colors.ENDC}")
            span.record_exception(e)
            scores = {}
            faith_score = 0.0
            relevancy_score = 0.0
            cp_score = 0.0

        duration = time.time() - start_time
        
        print(f"{Colors.GREEN}✅ 평가 완료 ({duration:.2f}초){Colors.ENDC}")
        print(f"   -> Faithfulness: {faith_score:.4f}")
        print(f"   -> Answer Relevancy: {relevancy_score:.4f}")
        print(f"   -> Context Precision: {cp_score:.4f}")
        
        span.set_attribute("ragas.faithfulness", faith_score)
        span.set_attribute("ragas.answer_relevancy", relevancy_score)
        span.set_attribute("ragas.context_precision", cp_score)
        span.set_attribute("rag.scenario", scenario_name)
        span.set_attribute("rag.question", query)
        span.set_attribute("rag.answer", answer)
        span.set_attribute("rag.scores", scores)
        span.set_attribute("rag.context_count", len(contexts))
        span.set_attribute("ragas.status", "success")
        
        print(f"\n{Colors.BLUE}📡 [Observability] Arize Trace 전송 완료{Colors.ENDC}")
        print(f"   -> Span ID: {span.get_span_context().span_id:x}")
        print(f"   -> Trace ID: {span.get_span_context().trace_id:x}")

# --- 메인 실행 흐름 ---
async def main():
    print(f"{Colors.BOLD}=================================================={Colors.ENDC}")
    print(f"{Colors.BOLD}   BupBlessYou Dual-Source RAG Demo Script        {Colors.ENDC}")
    print(f"{Colors.BOLD}=================================================={Colors.ENDC}")
    
    check_env()
    tracer = setup_arize_tracer()
    
    scenarios = [
        {
            "name": "강제추행 (Indecent Act)",
            "query": "이 사건의 처벌 법규와 양형 인자는 무엇인가?",
            "code": "criterion_03",
            "case_id": 750,
            "reference": "강제추행죄(형법 제298조)는 10년 이하의 징역 또는 1천500만원 이하의 벌금에 처합니다."
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{Colors.HEADER}##################################################")
        print(f"# Scenario: {scenario['name']} (Filter: {scenario['code']})")
        print(f"##################################################{Colors.ENDC}")
        
        # Start a root span for the scenario
        with tracer.start_as_current_span(f"Workflow: {scenario['name']}") as root_span:
            root_trace_id = root_span.get_span_context().trace_id
            print(f"📍 [Root Trace ID: {root_trace_id:x}] Started workflow for '{scenario['name']}'")
            
            root_span.set_attribute("scenario.name", scenario['name'])
            root_span.set_attribute("scenario.code", scenario['code'])
            
            # Standard Attribute for Root Span Input
            root_span.set_attribute("input.value", scenario['query'])
            
            contexts = perform_dual_search(tracer, scenario['query'], scenario['code'])
            answer = generate_answer(tracer, scenario['query'], contexts)
            
            # Standard Attribute for Root Span Output
            root_span.set_attribute("output.value", answer)
            
            await evaluate_and_log(tracer, scenario['query'], answer, contexts, scenario['name'])
            
            print(f"📍 [Root Trace ID: {root_trace_id:x}] Workflow completed")
    
    print(f"\n{Colors.BOLD}🎉 모든 시나리오 테스트가 완료되었습니다.{Colors.ENDC}")
    
    # 프로그램 종료 전 Trace 데이터 강제 전송
    print(f"{Colors.BLUE}⏳ Trace 데이터 전송 중...{Colors.ENDC}")
    try:
        provider = trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush()
        elif hasattr(provider, "shutdown"):
            provider.shutdown()
        
        # 안전을 위해 잠시 대기
        time.sleep(2)
        print(f"{Colors.GREEN}✅ 전송 완료 및 종료{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.FAIL}⚠️ Trace 전송 중 오류: {e}{Colors.ENDC}")

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(main())