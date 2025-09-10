#!/bin/bash
# setup_hiva.sh - Create HiVA project skeleton with packages and init files

# Base directory
BASE_DIR="hiva"

# Directory list
DIRS=(
  "$BASE_DIR/orchestrator"
  "$BASE_DIR/graph"
  "$BASE_DIR/routing"
  "$BASE_DIR/agents"
  "$BASE_DIR/gradients"
  "$BASE_DIR/evolution"
  "$BASE_DIR/tools"
  "$BASE_DIR/knowledge/ontology"
  "$BASE_DIR/envs"
  "$BASE_DIR/eval"
  "$BASE_DIR/ui"
  "$BASE_DIR/configs"
)

# Files per directory (keyed by directory name)
declare -A FILES
FILES["$BASE_DIR/orchestrator"]="loop.py state.py"
FILES["$BASE_DIR/graph"]="model.py exec_subgraph.py memory.py"
FILES["$BASE_DIR/routing"]="kabb.py distance.py synergy.py"
FILES["$BASE_DIR/agents"]="base.py semantics.py forward_msgs.py backward_msgs.py"
FILES["$BASE_DIR/gradients"]="textgrad_adapter.py parser.py credit_assign.py"
FILES["$BASE_DIR/evolution"]="semantic.py topology.py constraints.py"
FILES["$BASE_DIR/tools"]="registry.py runtime.py evolution.py"
FILES["$BASE_DIR/knowledge"]="kg_client.py matching.py"
FILES["$BASE_DIR/envs"]="base.py programmatic.py qa.py agentic.py math.py"
FILES["$BASE_DIR/eval"]="datasets.py metrics.py runner.py"
FILES["$BASE_DIR/ui"]="inspect.py"
FILES["$BASE_DIR/configs"]="defaults.yaml"

# Create directories and files
for dir in "${DIRS[@]}"; do
  mkdir -p "$dir"
  touch "$dir/__init__.py"
done

# Create files inside each directory
for dir in "${!FILES[@]}"; do
  for f in ${FILES[$dir]}; do
    touch "$dir/$f"
  done
done

# Add __init__.py at base package level
touch "$BASE_DIR/__init__.py"

echo "HiVA project skeleton created under '$BASE_DIR/'"
