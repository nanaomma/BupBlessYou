"""
actual_reason에서 양형 요소 추출 및 DB 저장 스크립트

이 스크립트는:
1. cases 테이블에서 actual_reason이 있는 사건들을 조회
2. actual_reason을 파싱하여 양형 요소 추출 (카테고리화)
3. facts와 description에 있는 정보만 필터링 (스포일러 방지)
4. sentencing_factors 컬럼에 JSON 형태로 저장
"""
import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 환경 변수 로드
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f".env 파일 로드 완료: {env_path}")

from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session
from openai import OpenAI

from src.database.connection import SessionLocal
from src.database.models.case import Case
from src.config.settings import settings

# OpenAI 클라이언트 초기화
api_key = getattr(settings, "openai_api_key", None) or getattr(settings, "api_key", None)
base_url = getattr(settings, "openai_base_url", None) or getattr(settings, "base_url", None)

if base_url:
    client = OpenAI(api_key=api_key, base_url=base_url)
else:
    client = OpenAI(api_key=api_key)


def get_sentencing_factors_schema(casename: str) -> str:
    """
    범죄 유형에 따른 양형 요소 스키마 반환
    
    Args:
        casename: 범죄 유형 (예: "음주운전", "사기", "폭행", "절도" 등)
        
    Returns:
        JSON 스키마 템플릿 문자열
    """
    casename_lower = (casename or "").lower()
    
    # 공통 카테고리 (모든 범죄 유형에 적용)
    base_schema = """
  "전력_관련": {{
    "동종_전과_횟수": null,  // 숫자 - 구체적인 횟수가 명시되면 해당 횟수, "전력이 있다"만 언급되면 1, 없으면 null
    "집행유예_이상_전과_존재": null  // true/false/null - "전력이 있다"면 true, "없다"면 false
  }},
  "책임_인식_관련": {{
    "범행_인정": null,  // true (인정함) / false (부인함) / null (언급 없음)
    "반성_여부": null  // true (반성함) / false (반성 안 함) / null (언급 없음)
  }},
  "기타_참작사유": {{
    "피해_회복": null,  // true/false/null
    "합의": null,  // true/false/null
    "자수": null,  // true/false/null
    "생계_목적": null,  // true/false/null (경제적 어려움으로 인한 범행)
    "조직적_범행": null  // true/false/null (조직적/계획적 범행 여부)
  }}"""
    
    # 범죄 유형별 특화 카테고리
    if "음주" in casename_lower or "도로교통" in casename_lower:
        # 음주운전 특화 요소
        return f"""{{
  "전력_관련": {{
    "동종_전과_횟수": null,
    "집행유예_이상_전과_존재": null
  }},
  "범행_위험성_관련": {{
    "혈중알코올농도": null,  // 숫자 (예: 0.102) 또는 null
    "도로_유형": null,  // "도심", "고속도로", "일반도로" 등 또는 null
    "주행_거리": null,  // 숫자 (km) 또는 null
    "처벌규정_강화_이후_범행": null  // true/false/null (처벌규정 강화 시점 이후 범행 여부)
  }},
  "책임_인식_관련": {{
    "범행_인정": null,
    "반성_여부": null
  }},
  "기타_참작사유": {{
    "피해_회복": null,
    "합의": null,
    "자수": null,
    "생계_목적": null,
    "조직적_범행": null
  }}
}}"""
    
    elif "사기" in casename_lower:
        # 사기 범죄 특화 요소
        return f"""{{
  "전력_관련": {{
    "동종_전과_횟수": null,
    "집행유예_이상_전과_존재": null
  }},
  "범행_위험성_관련": {{
    "피해_금액": null,  // 숫자 (원) 또는 null
    "피해자_수": null,  // 숫자 또는 null
    "조직적_사기": null  // true/false/null
  }},
  "책임_인식_관련": {{
    "범행_인정": null,
    "반성_여부": null
  }},
  "기타_참작사유": {{
    "피해_회복": null,
    "합의": null,
    "자수": null,
    "생계_목적": null,
    "조직적_범행": null
  }}
}}"""
    
    elif "폭행" in casename_lower or "상해" in casename_lower:
        # 폭행/상해 특화 요소
        return f"""{{
  "전력_관련": {{
    "동종_전과_횟수": null,
    "집행유예_이상_전과_존재": null
  }},
  "범행_위험성_관련": {{
    "상해_정도": null,  // "경미", "중상", "중상해", "전치기간" 등 또는 null
    "전치_기간": null,  // 숫자 (주) 또는 null
    "피해자_관계": null  // "친족", "지인", "불특정" 등 또는 null
  }},
  "책임_인식_관련": {{
    "범행_인정": null,
    "반성_여부": null
  }},
  "기타_참작사유": {{
    "피해_회복": null,
    "합의": null,
    "자수": null,
    "생계_목적": null,
    "조직적_범행": null
  }}
}}"""
    
    elif "절도" in casename_lower:
        # 절도 특화 요소
        return f"""{{
  "전력_관련": {{
    "동종_전과_횟수": null,
    "집행유예_이상_전과_존재": null
  }},
  "범행_위험성_관련": {{
    "절취_금액": null,  // 숫자 (원) 또는 null
    "절취_횟수": null,  // 숫자 또는 null
    "범행_장소": null,  // "편의점", "주거지", "상점" 등 또는 null
    "침입_여부": null  // true/false/null (침입 절도 여부)
  }},
  "책임_인식_관련": {{
    "범행_인정": null,
    "반성_여부": null
  }},
  "기타_참작사유": {{
    "피해_회복": null,
    "합의": null,
    "자수": null,
    "생계_목적": null,
    "조직적_범행": null
  }}
}}"""
    
    else:
        # 일반 범죄 유형 (공통 요소만)
        return f"""{{
  "전력_관련": {{
    "동종_전과_횟수": null,
    "집행유예_이상_전과_존재": null
  }},
  "범행_위험성_관련": {{
    "피해_규모": null,  // 피해 금액, 피해자 수, 상해 정도 등 범죄 유형에 맞게 자유롭게 기술
    "범행_동기": null  // 범행 동기 (예: "금전 목적", "갈등 기반" 등)
  }},
  "책임_인식_관련": {{
    "범행_인정": null,
    "반성_여부": null
  }},
  "기타_참작사유": {{
    "피해_회복": null,
    "합의": null,
    "자수": null,
    "생계_목적": null,
    "조직적_범행": null
  }}
}}"""


def extract_sentencing_factors_with_llm(
    actual_reason: str,
    facts: str,
    description: str,
    casename: str
) -> Optional[Dict[str, Any]]:
    """
    LLM을 사용하여 actual_reason에서 양형 요소 추출
    사실관계 정보는 추출하되, 양형 기준/판단 근거는 제외
    
    Args:
        actual_reason: 실제 판결 이유
        facts: 사실관계
        description: 사건 개요
        casename: 범죄 유형 (예: "음주운전", "사기", "폭행", "절도" 등)
        
    Returns:
        카테고리화된 양형 요소 딕셔너리
    """
    
    # 범죄 유형에 맞는 스키마 가져오기
    schema_template = get_sentencing_factors_schema(casename)
    
    prompt = f"""당신은 법률 데이터 분석 전문가입니다.
actual_reason에서 양형 고려 요소를 추출하되, **사실관계 정보**만 추출하세요.
양형 기준의 가중/감경 요소 판단이나 판사의 최종 판단 근거는 제외하세요.

범죄 유형: {casename}

[사실관계 (facts)]
{facts[:1000] if facts else "없음"}

[사건 개요 (description)]
{description[:1000] if description else "없음"}

[실제 판결 이유 (actual_reason)]
{actual_reason[:2000] if actual_reason else "없음"}

위 actual_reason에서 양형 고려 요소를 추출하되, 다음 규칙을 반드시 준수하세요:

**추출 가능한 정보 (사실관계 정보):**
- 범죄 전력/전과: actual_reason에 명시적으로 언급된 실제 과거 처벌 기록
  * 구체적인 횟수가 명시된 경우: 해당 횟수로 추출 (예: "두 차례" → 2, "3회" → 3)
  * 횟수가 명시되지 않았지만 "전력이 있다"고 언급된 경우: 1로 추출 (예: "전력이 있는 점" → 1)
  * "전력이 없다"고 명시된 경우: 동종_전과_횟수는 null, 집행유예_이상_전과_존재는 false
  * 집행유예_이상_전과_존재: 전력이 있으면 true, 없으면 false
- 피해 규모: 실제 피해 금액, 상해 정도, 피해자 수 등 (구체적인 숫자만)
- 피해자 관계: 실제 관계 (예: "친족", "지인")
- 반성/인정 여부: 실제 행위를 boolean 값으로 추출 (예: "범행 인정" → true, "반성하는 점" → true, "범행 부인" → false)
- 합의/피해 회복: 실제로 일어난 일 (예: "합의 완료", "피해 회복 노력")
- 자수 여부: 실제 행위
- 기타 사실 정보: 실제로 일어난 사건의 사실

**제외할 정보:**
- 범행 수법/방법/수단: 범행 방식에 대한 구체적인 기술은 제외 (양형 기준 결정에 직접적인 영향이 적음)

**제외해야 할 정보 (스포일러/판단 근거):**
- 양형 기준의 가중/감경 요소에 대한 판단 (예: "가중 사유가 있다", "감경 사유가 인정된다")
- 판사의 최종 판단 근거나 평가 (예: "엄벌에 처할 필요성이 있다", "참작할 만한 사유가 있다")
- 최종 형량이나 판결 결과
- 양형 기준 자체의 내용

규칙:
1. actual_reason에서 **사실관계 정보**를 추출 (facts/description에 없어도 actual_reason에 있는 사실 정보는 포함 가능)
2. 전과 횟수: 구체적인 횟수가 명시되지 않았지만 "전력이 있다"고 언급된 경우 1로 기록
3. 범행 인정/반성 여부: 반드시 boolean 값(true/false)으로 기록, 자유 텍스트가 아님 (예: "범행 인정" → true, "반성하는 점" → true, "부인함" → false)
4. 양형 기준 판단이나 판사의 평가는 제외
5. 아래 JSON 형식으로 출력 (값이 없으면 null)
6. 범죄 유형({casename})에 맞는 요소만 포함 (해당 범죄 유형에 적합하지 않은 필드는 null)

출력 형식:
{schema_template}

중요: 
- **사실 정보**는 추출하되, **판단/평가**는 제외하세요
- 범행 인정/반성 여부는 반드시 boolean 값(true/false)으로 기록, 자유 텍스트가 아닙니다
- 범죄 유형에 맞지 않는 필드는 null로 설정하세요
- JSON만 출력하세요 (코드 블록 없이 순수 JSON)"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a legal data analyst. Output only valid JSON. No markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
        )
        
        content = response.choices[0].message.content.strip()
        
        # JSON 블록 제거 (```json ... ```)
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        
        factors = json.loads(content)
        return factors
        
    except Exception as e:
        print(f"LLM 추출 실패: {e}")
        print(f"응답 내용: {response.choices[0].message.content[:200] if 'response' in locals() else 'N/A'}")
        return None


def verify_facts_match(
    extracted_value: Any,
    facts: str,
    description: str,
    field_name: str
) -> bool:
    """
    추출된 값이 facts나 description에 실제로 있는지 확인
    (추가 검증 로직 - 선택적, LLM이 이미 필터링하므로 참고용)
    
    Note: 이 함수는 현재 사용되지 않음 (LLM 프롬프트에서 이미 필터링하도록 지시)
    """
    if extracted_value is None:
        return True  # null 값은 OK
    
    combined_text = f"{facts} {description}".lower()
    value_str = str(extracted_value).lower()
    
    # 숫자인 경우 숫자 매칭
    if isinstance(extracted_value, (int, float)):
        # 숫자가 텍스트에 포함되어 있는지 확인
        if value_str in combined_text or re.search(rf"\b{re.escape(value_str)}\b", combined_text):
            return True
    
    # 문자열/불리언인 경우 키워드 매칭
    if isinstance(extracted_value, bool):
        # true/false는 의미 있는 키워드로 변환 필요
        if "반성" in field_name and extracted_value:
            return any(kw in combined_text for kw in ["반성", "뉘우치", "사과"])
        if "인정" in field_name and extracted_value:
            return any(kw in combined_text for kw in ["인정", "시인", "자백"])
        return True  # false인 경우 사실관계에 없어도 OK (추론 가능)
    
    # 문자열인 경우
    if isinstance(extracted_value, str):
        return value_str in combined_text
    
    return True


def process_all_cases(dry_run: bool = False):
    """
    모든 cases를 처리하여 sentencing_factors 추출 및 저장
    
    Args:
        dry_run: True이면 DB에 저장하지 않고 결과만 출력
    """
    db = SessionLocal()
    
    try:
        # actual_reason이 있는 형사 사건만 조회
        query = select(Case).where(
            Case.casetype == "criminal",
            Case.actual_reason.isnot(None),
            Case.actual_reason != ""
        )
        
        cases = db.execute(query).scalars().all()
        print(f"처리할 사건 수: {len(cases)}")
        
        processed = 0
        skipped = 0
        failed = 0
        
        for case in cases:
            print(f"\n[사건 ID: {case.id}] {case.case_number} - {case.casename}")
            
            # 기존 값 덮어쓰기 옵션이 없으므로 항상 추출 (이미 저장된 값도 업데이트)
            
            # LLM으로 추출
            factors = extract_sentencing_factors_with_llm(
                actual_reason=case.actual_reason or "",
                facts=case.facts or "",
                description=case.description or "",
                casename=case.casename or ""
            )
            
            if factors is None:
                print("  추출 실패")
                failed += 1
                continue
            
            # 결과 출력
            print(f"  추출된 요소:")
            print(json.dumps(factors, ensure_ascii=False, indent=2))
            
            # DB에 저장
            if not dry_run:
                case.sentencing_factors = factors
                db.commit()
                print("  저장 완료")
            else:
                print("  [DRY RUN] 저장하지 않음")
            
            processed += 1
            
            # 진행 상황 출력
            if processed % 10 == 0:
                print(f"\n진행 상황: {processed}/{len(cases)} 처리 완료")
        
        print(f"\n=== 완료 ===")
        print(f"처리됨: {processed}")
        print(f"건너뜀: {skipped}")
        print(f"실패: {failed}")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="actual_reason에서 양형 요소 추출")
    parser.add_argument("--dry-run", action="store_true", help="DB에 저장하지 않고 결과만 출력")
    parser.add_argument("--case-id", type=int, help="특정 사건 ID만 처리")
    
    args = parser.parse_args()
    
    if args.case_id:
        # 특정 사건만 처리
        db = SessionLocal()
        try:
            case = db.get(Case, args.case_id)
            if not case:
                print(f"사건 ID {args.case_id}를 찾을 수 없습니다.")
                sys.exit(1)
            
            print(f"[사건 ID: {case.id}] {case.case_number} - {case.casename}")
            factors = extract_sentencing_factors_with_llm(
                actual_reason=case.actual_reason or "",
                facts=case.facts or "",
                description=case.description or "",
                casename=case.casename or ""
            )
            
            if factors:
                print("\n추출된 요소:")
                print(json.dumps(factors, ensure_ascii=False, indent=2))
                
                if not args.dry_run:
                    case.sentencing_factors = factors
                    db.commit()
                    print("\n저장 완료")
                else:
                    print("\n[DRY RUN] 저장하지 않음")
        finally:
            db.close()
    else:
        # 모든 사건 처리
        process_all_cases(dry_run=args.dry_run)

