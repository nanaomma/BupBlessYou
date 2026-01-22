from __future__ import annotations

import re
from typing import List, Dict, Any, Tuple

# 토큰화 + 매칭
def tokenize_ko(text: str) -> List[str]:
    if not text:
        return []
    # 한글/숫자 토큰
    toks = re.findall(r"[가-힣]{2,}|\d+", text)
    return [t for t in toks if t]

def keyword_overlap_score(query: str, doc: str) -> float:
    q = set(tokenize_ko(query))
    d = tokenize_ko(doc)
    if not q or not d:
        return 0.0
    # 단순 overlap + 빈도 가중(질문에 등장한 단어가 문서에서 많이 나오면 가산)
    overlap = 0.0
    for w in q:
        c = d.count(w)
        if c > 0:
            overlap += 1.0 + min(2.0, 0.3 * c)  # 빈도는 상한을 둠
    return overlap

def rerank_matches_by_keywords(
    query_text_for_keywords: str,
    matches: List[Dict[str, Any]],
    text_getter=lambda m: (m.get("metadata") or {}).get("text") or m.get("metadata", {}).get("content") or "",
) -> List[Dict[str, Any]]:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for m in matches:
        doc_text = str(text_getter(m) or "")
        score = keyword_overlap_score(query_text_for_keywords, doc_text)
        scored.append((score, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored]
