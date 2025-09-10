from __future__ import annotations
from typing import List, Dict
from .kg_client import OxigraphClient

EX = "http://example.org/"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"

def _sparql_escape(uri: str) -> str:
    return f"<{uri}>"

def kg_capability_overlap(client: OxigraphClient, agent_caps: List[str], task_caps: List[str]) -> float:
    '''Compute overlap where a provided capability satisfies a required capability if
    provided rdfs:subClassOf* required (or equal). If ex:subsumes is present (materialized),
    use it for faster checks.
    '''
    if not task_caps:
        return 1.0
    satisfied = 0
    for req in task_caps:
        requ = _sparql_escape(req)
        # First try ex:subsumes
        q_fast = f'''
        PREFIX ex: <{EX}>
        SELECT (COUNT(*) AS ?c) WHERE {{
          VALUES ?p {{ {' '.join(_sparql_escape(p) for p in agent_caps)} }}
          ?p ex:subsumes {requ} .
        }}
        '''
        rows_fast = client.query(q_fast)
        count_fast = int(rows_fast[0]["c"]) if rows_fast and "c" in rows_fast[0] else 0
        if count_fast > 0:
            satisfied += 1
            continue

        # Fallback to rdfs:subClassOf* (property path)
        q = f'''
        PREFIX rdfs: <{RDFS}>
        SELECT (COUNT(*) AS ?c) WHERE {{
          VALUES ?p {{ {' '.join(_sparql_escape(p) for p in agent_caps)} }}
          ?p rdfs:subClassOf* {requ} .
        }}
        '''
        rows = client.query(q)
        count = int(rows[0]["c"]) if rows and "c" in rows[0] else 0
        if count > 0:
            satisfied += 1
    return satisfied / len(task_caps)

def kg_distance(client: OxigraphClient, agent_meta: Dict, requirements: Dict) -> float:
    '''1 - overlap ratio; expects URIs in meta['capabilities_uri'] and requirements['capabilities_uri']'''
    a_caps = agent_meta.get("capabilities_uri") or []
    t_caps = requirements.get("capabilities_uri") or []
    if not t_caps:
        return 0.0
    ratio = kg_capability_overlap(client, a_caps, t_caps)
    return max(0.0, 1.0 - ratio)
