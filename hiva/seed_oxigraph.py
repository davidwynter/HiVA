# Oxigraph seeder & subclass-closure materializer for HiVA.
# - Loads capabilities.ttl / tasks.ttl to Oxigraph.
# - Pre-materializes subclass closure by inserting explicit triples:
#     ?x rdfs:subClassOf* ?y  ==>  INSERT ex:subsumes ?y or reuse rdfs:subClassOf for all implied edges
# - Provides two strategies:
#     A) explicit transitive closure into a dedicated predicate ex:subsumes
#     B) saturation of rdfs:subClassOf with its transitive closure careful: may expand the base ontology

# Usage:
#   python seed_oxigraph.py --query http://localhost:7878/query \
#                           --update http://localhost:7878/update \
#                           --strategy subsumes

import argparse, sys, time
from pathlib import Path
from hiva.knowledge.kg_client import OxigraphClient

EX = "http://example.org/"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"

SUBSUMES = f"{EX}subsumes"

def load_ontologies(client: OxigraphClient, base_dir: Path) -> None:
    cap = (base_dir / "knowledge" / "ontology" / "capabilities.ttl").read_text(encoding="utf-8")
    tasks = (base_dir / "knowledge" / "ontology" / "tasks.ttl").read_text(encoding="utf-8")
    client.load_ttl(cap, base_iri=EX)
    client.load_ttl(tasks, base_iri=EX)

def materialize_subsumes(client: OxigraphClient) -> None:
    # Create ex:subsumes edges for rdfs:subClassOf*
    # Insert all pairs (?x ?y) such that ?x rdfs:subClassOf* ?y
    # We do iterative fixpoint: each INSERT may enable new pairs if ex:subsumes is used again, but here we only depend on rdfs:subClassOf.
    update = f'''
    PREFIX rdfs: <{RDFS}>
    PREFIX ex: <{EX}>
    INSERT {{ ?x ex:subsumes ?y }}
    WHERE  {{ ?x rdfs:subClassOf* ?y }}
    '''
    client.update(update)

def saturate_rdfs_subclassof(client: OxigraphClient, max_iters: int = 5) -> None:
    # Compute transitive closure into rdfs:subClassOf itself via iterative insertion of inferred edges:
    # If ?x rdfs:subClassOf ?y and ?y rdfs:subClassOf ?z then INSERT ?x rdfs:subClassOf ?z
    # Repeat until no new triples are added (bounded by max_iters to avoid infinite loops).
    update_template = f'''
    PREFIX rdfs: <{RDFS}>
    INSERT {{ ?x rdfs:subClassOf ?z }}
    WHERE {{
       ?x rdfs:subClassOf ?y .
       ?y rdfs:subClassOf ?z .
       FILTER NOT EXISTS {{ ?x rdfs:subClassOf ?z }}
    }}
    '''
    for i in range(max_iters):
        client.update(update_template)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="Oxigraph SPARQL query endpoint")
    ap.add_argument("--update", required=True, help="Oxigraph SPARQL update endpoint")
    ap.add_argument("--strategy", choices=["subsumes","saturate"], default="subsumes",
                    help="Materialization strategy: 'subsumes' builds ex:subsumes closure; 'saturate' expands rdfs:subClassOf with closure")
    args = ap.parse_args()

    client = OxigraphClient(query_endpoint=args.query, update_endpoint=args.update)
    base_dir = Path(__file__).resolve().parent.parent / "hiva"
    load_ontologies(client, base_dir)

    if args.strategy == "subsumes":
        materialize_subsumes(client)
        print("Materialized subclass closure into ex:subsumes")
    else:
        saturate_rdfs_subclassof(client)
        print("Saturated rdfs:subClassOf transitive closure")

if __name__ == "__main__":
    main()
