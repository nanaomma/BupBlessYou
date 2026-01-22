"""Case service - 사건 데이터 조회 서비스"""
import random
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models.case import Case
from src.database.connection import async_session_maker
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def get_random_case_by_scenario(scenario_type: str) -> Optional[dict]:
    """
    시나리오 타입에 맞는 케이스를 랜덤으로 조회

    Args:
        scenario_type: 시나리오 타입 (예: '강제추행', '사기')

    Returns:
        케이스 정보 딕셔너리 또는 None
    """
    try:
        async with async_session_maker() as session:
            # USE_YN='Y'이고 casename이 시나리오 타입과 일치하는 케이스 조회
            stmt = select(Case).where(
                Case.use_yn == 'Y',
                Case.casename == scenario_type
            )
            result = await session.execute(stmt)
            cases = result.scalars().all()

            if not cases:
                logger.warning(f"No cases found for scenario: {scenario_type}")
                return None

            # 랜덤으로 하나 선택
            selected_case = random.choice(cases)

            # 딕셔너리로 변환
            return {
                "id": selected_case.id,
                "case_number": selected_case.case_number,
                "title": selected_case.title,
                "casetype": selected_case.casetype,
                "casename": selected_case.casename,
                "description": selected_case.description,
                "facts": selected_case.facts,
                "actual_label": selected_case.actual_label,
                "actual_rule": selected_case.actual_rule,
                "actual_reason": selected_case.actual_reason,
                "sentencing_factors": selected_case.sentencing_factors,
                "source": selected_case.source,
                "case_metadata": selected_case.case_metadata
            }

    except Exception as e:
        logger.error(f"Error fetching random case for scenario {scenario_type}: {e}")
        return None


async def get_available_scenarios() -> list[str]:
    """
    사용 가능한 시나리오 목록 조회 (USE_YN='Y'인 케이스의 casename 유니크 값)

    Returns:
        시나리오 타입 리스트
    """
    try:
        async with async_session_maker() as session:
            stmt = select(Case.casename).where(Case.use_yn == 'Y').distinct()
            result = await session.execute(stmt)
            scenarios = [row[0] for row in result.fetchall() if row[0]]

            logger.info(f"Available scenarios: {scenarios}")
            return scenarios

    except Exception as e:
        logger.error(f"Error fetching available scenarios: {e}")
        return []


async def get_case_by_id(case_id: int) -> Optional[dict]:
    """
    ID로 케이스 조회

    Args:
        case_id: 케이스 ID

    Returns:
        케이스 정보 딕셔너리 또는 None
    """
    try:
        async with async_session_maker() as session:
            stmt = select(Case).where(Case.id == case_id)
            result = await session.execute(stmt)
            selected_case = result.scalars().first()

            if not selected_case:
                logger.warning(f"Case not found with ID: {case_id}")
                return None

            return {
                "id": selected_case.id,
                "case_number": selected_case.case_number,
                "title": selected_case.title,
                "casetype": selected_case.casetype,
                "casename": selected_case.casename,
                "description": selected_case.description,
                "facts": selected_case.facts,
                "actual_label": selected_case.actual_label,
                "actual_rule": selected_case.actual_rule,
                "actual_reason": selected_case.actual_reason,
                "sentencing_factors": selected_case.sentencing_factors,
                "source": selected_case.source,
                "case_metadata": selected_case.case_metadata
            }

    except Exception as e:
        logger.error(f"Error fetching case with ID {case_id}: {e}")
        return None
