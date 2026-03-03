# 🧠 Text Analysis Agent — LangGraph + FastAPI + Python

**Author:** Candace Grant · Birds and Roses 
**Core Stack:** LangGraph · FastAPI · Python  

A multi-step AI agent with **conditional routing**, **stateful graph execution**, and **streaming node output** — built using the LangGraph pattern and served as a production API via FastAPI.

---

## Quick Start

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Dashboard:   http://localhost:8000
# Swagger UI:  http://localhost:8000/docs
# Graph Info:  http://localhost:8000/graph/info
```

## LangGraph Architecture

```
START
  │
  ▼
┌─────────────────┐
│ classify_intent  │  ← Router node: determines analysis path
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ extract_features │  ← Always runs: 18 NLP features
└────────┬────────┘
         │
    ┌────┴────┐  Conditional Routing
    │ ROUTER  │  ─────────────────────────────────────────
    └────┬────┘  spam_check   → spam_analysis + sentiment
         │       business     → sentiment + entity_extraction
         │       personal     → sentiment
         │       general      → all three analyzers
    ┌────┴────────────────────────┐
    │  Analysis Nodes (parallel)  │
    │  ┌───────────────────────┐  │
    │  │ spam_analysis         │  │  Weighted scoring + sigmoid
    │  │ sentiment_analysis    │  │  Lexicon-based + intensity
    │  │ entity_extraction     │  │  Pattern-based NER
    │  └───────────────────────┘  │
    └────────────┬────────────────┘
                 │
                 ▼
       ┌─────────────────┐
       │ risk_assessment  │  ← Aggregation node
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │ compile_report   │  ← Terminal node
       └────────┬────────┘
                │
               END
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health + graph node list |
| `GET` | `/graph/info` | Full graph topology, nodes, edges, routes |
| `POST` | `/analyze` | **Full agent pipeline** — all nodes + execution trace |
| `POST` | `/classify` | Quick classification — simplified response |
| `POST` | `/analyze/batch` | Batch analysis (up to 20 messages) |
| `POST` | `/analyze/stream` | **Streaming** — node-by-node output via NDJSON |
| `GET` | `/stats` | Aggregated analytics (intents, risk, latency) |
| `GET` | `/history` | Recent analysis log |

## Example Usage

### Full Analysis
```python
import requests

response = requests.post(
    "http://localhost:8000/analyze",
    json={"text": "URGENT! Click here to claim your $5000 prize NOW!!!"}
)
report = response.json()

print(report["detected_intent"])           # → "spam_check"
print(report["spam_analysis"]["prediction"])  # → "spam"
print(report["risk_assessment"]["risk_level"]) # → "high"
print(report["graph_execution"]["nodes_executed"])
# → ["classify_intent", "extract_features", "spam_analysis",
#     "sentiment_analysis", "risk_assessment", "compile_report"]
```

### Streaming Execution
```python
import requests, json

response = requests.post(
    "http://localhost:8000/analyze/stream",
    json={"text": "Meeting at 3pm to review the budget proposal."},
    stream=True
)
for line in response.iter_lines():
    node = json.loads(line)
    print(f"✓ {node['node']} — {node.get('time_ms', '')}ms")

# ✓ classify_intent — 0.02ms
# ✓ extract_features — 0.05ms
# ✓ sentiment_analysis — 0.03ms
# ✓ entity_extraction — 0.04ms
# ✓ risk_assessment — 0.01ms
# ✓ compile_report — (full report JSON)
```

## Key Interview Talking Points

### LangGraph
- **Stateful graph execution** — typed state flows through nodes, each adding its analysis
- **Conditional routing** — intent classification determines which analysis nodes fire
- **Node-by-node streaming** — transparent agent execution via NDJSON streaming
- **Graph introspection** — `/graph/info` endpoint exposes full topology
- **Why LangGraph over chains?** — Complex workflows need branching, cycles, and state management that linear chains can't handle

### FastAPI
- **Async endpoints** — non-blocking I/O for high throughput
- **Pydantic V2 validation** — type-safe request/response at API boundary
- **StreamingResponse** — real-time node output for agent transparency
- **Auto-generated OpenAPI docs** — Swagger UI + ReDoc from type annotations
- **CORS middleware** — production-ready for frontend integration

### Python
- **NLP feature engineering** — 18 features including vocabulary richness, caps ratio, entity density
- **Regex-based NER** — dates, times, money, emails, URLs, proper nouns
- **Sigmoid normalization** — converting raw scores to calibrated probabilities
- **Clean architecture** — separation of graph logic, node functions, and API layer
