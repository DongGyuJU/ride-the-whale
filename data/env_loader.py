"""
.env 파일에서 환경변수 로드.

bash special character (!@$) 안전 처리 — Python으로 직접 파싱.
"""

from __future__ import annotations
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_env(env_file: str = '.env', override: bool = True) -> dict:
    """
    .env 파일에서 환경변수 로드.

    Args:
        env_file: .env 파일 경로 (상대/절대)
        override: True면 이미 설정된 환경변수 덮어씀

    Returns:
        로드된 변수들의 dict
    """
    env_path = Path(env_file)
    if not env_path.exists():
        # 프로젝트 루트에서 찾기
        env_path = Path(__file__).resolve().parent.parent / env_file
        if not env_path.exists():
            logger.warning(f".env 파일 없음 — 환경변수만 사용")
            return {}

    loaded = {}
    with open(env_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            # 따옴표 제거 (양 끝만)
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            if key in os.environ and not override:
                continue
            os.environ[key] = value
            loaded[key] = value

    logger.info(f".env 로드: {len(loaded)}개 변수")
    return loaded


def get_required(key: str) -> str:
    """필수 환경변수 — 없으면 에러."""
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(
            f"환경변수 {key} 필요. "
            f".env에 설정 후 load_env() 호출 또는 export {key}=..."
        )
    return val
