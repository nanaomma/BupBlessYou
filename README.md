
# ⚖️ BupBlessYou
**멀티에이전트를 활용한 모의법정 시뮬레이션**

BupBlessYou는 사용자가 판사가 되어 검사와 변호사 AI 에이전트의 공방을 듣고 판결을 내리는 모의 법정 시뮬레이션 플랫폼입니다. LangGraph 기반의 멀티 에이전트 시스템과 RAG(Retrieval-Augmented Generation)를 활용하여 실제 판례와 법령에 기반한 사실적인 법정 경험을 제공합니다.

<img width="1340" height="723" alt="스크린샷 2026-01-07 오후 9 30 41" src="https://github.com/user-attachments/assets/8c7bfc42-38ab-4e88-812d-15453e84961e" />



---

## 🎯 목표 (Goals)

*   **RAG 기반 멀티 에이전트 시스템 구현**: 다양한 역할의 AI 에이전트가 협업하고 대립하는 모의법정 시뮬레이터를 구축합니다.
*   **법적 편향 및 불균형 이해**: AI를 활용해 판례와 양형 근거를 구조적으로 비교함으로써 법원의 판단 과정에서 발생할 수 있는 편향과 불균형 요소를 이해합니다.
*   **합리적 법적 판단 체험**: 사용자가 직접 양형을 결정해보는 체험을 통해 합리적인 법적 판단 절차와 사고 과정을 경험합니다.

## 🏆 성과 (Achievements)

### 🎓 교육적 성과
*   사용자는 AI와의 모의재판을 통해 **양형 근거·판결 구조·법리 적용**을 실제 사례 기반으로 체득합니다.
*   사건별 양형 비교 분석을 통해 사법 절차의 **비판적 법률 사고력**을 강화합니다.
*   판사 역할을 수행하며 **독립적 판단 능력**과 **논리적 의사결정** 역량을 기릅니다.

### 🛠️ 기술적 성과
*   **도메인 특화 RAG 시스템**: 한국 판례 기반의 대규모 데이터셋을 구축하고 활용합니다.
*   **법률 특화 파이프라인 완성**: 판결문, 사실관계, 양형 이유 등 비정형 텍스트를 처리하는 전처리·청킹·임베딩 파이프라인을 구축했습니다.
*   **멀티에이전트 구조 구현**: AI 변호사·AI 검사·AI 자문 시스템 간의 유기적인 협업 구조를 설계하고 구현했습니다.
      

### 🖥️ 시스템 성과
*   **시나리오 기반 인터랙션**: 실제 법정 절차를 모사하는 몰입감 있는 시나리오를 제공합니다.
*   **양형 비교 대시보드**: 판사(사용자)가 사건 전체를 평가하고 편향을 분석할 수 있는 도구를 제공합니다.
*   **근거 기반 판결 훈련**: RAG 기반 법령 및 양형기준과 판례 근거 제시로 신뢰성 있는 판결 훈련 시스템을 완성했습니다.

      |  |  |
      | :--- | :--- |
      | <img width="600" height="528" alt="Image" src="https://github.com/user-attachments/assets/f3ed7cfe-b9a1-42fe-8d55-80dac3c8a51c" />| <img width="925" height="600" alt="Image" src="https://github.com/user-attachments/assets/da54c750-94fb-4344-8e8b-c21fb6034dce" />|
*  **멀티 에이전트 성능 트랙킹 Dashboard**

    <img width="800" height="511" alt="스크린샷 2026-01-07 오후 9 29 24" src="https://github.com/user-attachments/assets/807646b5-c22c-4ccd-a2ff-886fb3f32304" />

---

## ✨ 주요 기능 (Key Features)

*   **👥 멀티 에이전트 시뮬레이션**: 검사, 변호사, 판사, 법률 자문가 등 전문 역할을 수행하는 AI 에이전트들이 상호작용하며 재판을 진행합니다.
*   **📚 RAG 기반 법률 정보**: Pinecone 벡터 DB를 활용해 관련 법령과 양형 기준을 실시간으로 검색하여 논리적인 주장을 펼칩니다.
*   **🔍 실시간 관측 및 평가 (Observability)**: LangSmith와 Arize Phoenix를 통해 에이전트의 추론 과정과 RAG 성능(Faithfulness, Relevance 등)을 실시간으로 모니터링합니다.
*   **🕹️ 인터랙티브 판결**: 사용자는 재판 과정에서 질문을 던지거나 최종 판결(형량 선고)을 내릴 수 있습니다.

## 🛠️ 기술 스택 (Tech Stack)

*   **Backend Framework**: FastAPI
*   **Frontend**: Streamlit (Demo/Orchestration), Jinja2 Templates
*   **LLM & Agent**: LangChain, LangGraph, OpenAI (GPT-4o)
*   **Vector DB**: Pinecone
*   **Observability**: LangSmith, Arize Phoenix, RAGAS (Evaluation)
*   **Database**: PostgreSQL
*   **Tools**: Python 3.10+
  <img width="1368" height="569" alt="Image" src="https://github.com/user-attachments/assets/55faf091-9f61-49d3-b265-33a0562eb032" />

## 👥 팀원 및 역할 (Roles)

| 이름 | 역할 | 담당 업무 |
| :--- | :--- | :--- |
| **김상현 (팀장)** | PM / AI Dev | 일정 관리, 진행 상황 체크, ```자문 에이전트 구현``` |
| **양은빈** | Documentation / AI Dev | 회의록 작성, 노션 관리, 프로젝트 문서화, ```자문 에이전트 구현``` |
| **안재현** | GitHub / Code Integration | Git 브랜치 관리, 머지 충돌 해결, ```판사 에이전트 구현``` |
| **김윤교** | Main Architect / AI Dev | 메인 구조 구현, 아키텍처 설계, ```판사 에이전트 구현``` |
| **조수진** | UI/UX / AI Dev | 시각화 자료 디자인, PPT 제작, ```검사/변호사 에이전트 구현``` |
| **한가은** | Data Research / AI Dev | 시나리오 구체화 및 관리, ```검사/변호사 에이전트 구현``` |

---

## 🚀 시작 가이드 (Getting Started)

### 1. 환경 변수 설정
`.env` 파일을 생성하고 필요한 API 키를 입력하세요. (`.env.example` 참고)
```bash
cp .env.example .env
# .env 파일 수정 (OPENAI_API_KEY, PINECONE_API_KEY 등)
```

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. 서버 실행 (FastAPI)
```bash
uvicorn src.api.main:app --reload
```
서버가 실행되면 `http://localhost:8000`에서 시나리오 선택 화면을 볼 수 있습니다.

### 4. 오케스트레이션 데모 실행 (Streamlit)
LangGraph의 동작 과정을 시각적으로 확인하려면 Streamlit 앱을 실행하세요.
```bash
streamlit run streamlit_app.py
```

### 5. RAG 성능 평가 실행
RAG 파이프라인의 검색 및 답변 성능을 테스트하려면 다음 스크립트를 실행하세요.
```bash
python demo_rag_workflow.py
```

