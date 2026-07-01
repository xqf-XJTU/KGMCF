# KGMCF：Knowledge-Grounded Multimodal Cognitive Framework for Aerospace Machining

## Introduction

KGMCF is a knowledge-grounded multimodal cognitive framework for aerospace special-shaped feature process planning. The project integrates engineering drawing perception, Aero-MPKG knowledge grounding, multimodal prompt fusion, process-plan generation, symbolic verification, and semantic traceback into a runnable system.The system uses POSIX paths, Linux shell.

## 1. Recommended environment

- Operating system: Ubuntu 22.04 LTS
- Python: 3.10+
- Backend: Flask, py2neo, NumPy, Pillow, ReportLab
- Frontend: HTML, CSS, JavaScript, vis-network
- Browser: Chrome / Edge / Firefox
- Database: Neo4j 5.x
- Multimodal Large Language Model: Qwen2.5-VL-72B

Put the local model or symbolic link under:

```text
models/Qwen2.5-VL-72B-Instruct/
```

Put LoRA adapters under:

```text
adapters/aero_lora/
```
## 2. Hardware Requirements

### Basic prototype mode
For dataset browsing, Neo4j graph visualization, local visual anchoring, KGMCF fallback planning, symbolic verification, and PDF export:
- CPU: 16 cores or above
- RAM: 24 GB minimum, 32 GB recommended
- Disk: 50 GB or above

### Neo4j graph mode
For larger Aero-MMKG visualization and query interaction:
- CPU: 8 cores recommended
- RAM: 16 GB-32 GB recommended
- Disk: SSD recommended

### Qwen2.5-VL-72B runtime mode
For external or local Qwen2.5-VL-72B multimodal inference:
- Recommended: high-memory GPU server
- VRAM: about 144 GB or more for BF16 model loading; lower VRAM may be possible with quantized deployment
- RAM: 128 GB or above recommended
- Disk: 200 GB or above recommended for model weights, adapters, logs, and runtime artifacts

## 3. Setup

Installs Python dependencies, installs Linux CJK fonts for PDF export, and installs optional Node.js tooling for JavaScript syntax checks.

Create the environment:

```bash
python3 -m venv .venv
pip install -r requirements.txt
pip install -e .
```

## 4. Run Linux

```bash
source .venv/bin/activate
```

The script runs:

```bash
python run_pipeline.py
python -m pytest -q
python -m compileall -q src scripts apps experiments tests
node --check apps/prototype/static/js/prototype.js
```


## 5. Start the prototype system

```bash
source .venv/bin/activate
./start_prototype_ubuntu22.sh
```


## 6. Neo4j connection

The project does not hard-code Neo4j credentials. Start Neo4j separately, then enter the Bolt URI, username, and password in the Aero-MMKG Visualization page.

Default URI format:

```text
bolt://localhost:7687
```

The graph API uses Neo4j internal identities:

```text
node.identity
relationship.identity
relationship.start_node.identity
relationship.end_node.identity
```

## 7. Qwen2.5-VL-72B runtime

The model runtime page supports either an OpenAI-compatible local/private endpoint or the local KGMCF fallback planner. Configure:

```text
configs/model_runtime.json
```


## 8. Notes

This Linux package keeps the paper-aligned core prototype workflow:

- Aero-Instruct-5K data management
- CV20 visual prototypes
- Aero-MMKG / Neo4j graph connection
- visual anchoring
- LGSE knowledge retrieval
- multimodal prompt fusion
- Qwen2.5-VL-72B runtime configuration
- KGMCF process planning
- symbolic verification and semantic traceback
- process-card JSON/PDF export
