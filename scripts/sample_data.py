"""Insert sample data for testing"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.database.connection import get_db
from src.database.models import Case


def insert_sample_cases():
    """Insert sample legal cases"""
    db = next(get_db())

    sample_cases = [
        Case(
            case_number="2023고단1234",
            title="투자 사기 사건",
            casename="사기",
            description="피고인이 고수익 투자를 약속하며 금전을 편취한 사건",
            facts="""피고인은 2023년 3월, SNS를 통해 피해자들에게 접근하여
                    '월 10% 수익 보장' 투자 상품을 홍보하였습니다.
                    총 20명의 피해자로부터 5천만원을 편취하였으나,
                    실제 투자는 이루어지지 않았습니다.""",
            actual_label="징역 2년",
            actual_reason=["피해 금액이 크다", "다수의 피해자", "일부 피해 회복 노력"],
            source="LBox Open"
        ),
        Case(
            case_number="2023고단5678",
            title="폭행 사건",
            casename="폭행",
            description="피고인이 피해자를 폭행하여 상해를 입힌 사건",
            facts="""피고인은 2023년 6월, 술에 취한 상태에서
                    피해자와 시비가 붙어 주먹으로 얼굴을 수회 가격하였습니다.
                    피해자는 안와골절 등 전치 4주의 상해를 입었습니다.""",
            actual_label="징역 1년, 집행유예 2년",
            actual_reason=["피해자와 합의", "초범", "깊이 반성"],
            source="LBox Open"
        ),
        Case(
            case_number="2023고단9012",
            title="절도 사건",
            casename="절도",
            description="피고인이 편의점에서 물건을 절취한 사건",
            facts="""피고인은 2023년 8월, 경제적 어려움으로 인해
                    편의점에서 식품 및 생필품(약 5만원 상당)을 절취하였습니다.
                    범행 직후 자수하였고 전액 변상하였습니다.""",
            actual_label="벌금 300만원",
            actual_reason=["피해 금액이 적다", "자수", "전액 변상", "생계형 범죄"],
            source="LBox Open"
        )
    ]

    try:
        for case in sample_cases:
            db.add(case)

        db.commit()

        print("Sample cases inserted successfully!")
        print(f"\nInserted {len(sample_cases)} cases:")
        for case in sample_cases:
            print(f"  - {case.case_number}: {case.title}")

    except Exception as e:
        db.rollback()
        print(f"Error inserting sample data: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    insert_sample_cases()
