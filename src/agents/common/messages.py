from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class ServiceRequest:
    action: str # "case" | "law" | "sentencing"
    case_id: int | str
    requester_role: str # "judge" | "prosecutor" | "defense"
    params: Optional[Dict[str, Any]] = None # top_k, namespace override 등

@dataclass
class ServiceResponse:
    ok: bool
    action: str
    requester_role: str
    case_id: int | str
    data: Any = None
    error: Optional[str] = None
