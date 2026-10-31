# Pharmacy AI Agent Evaluation — OpenAI + FastAPI

A local development starter that evaluates an OpenAI tool-using agent against public US pharmacy data. Includes runnable code, Swagger API UI, four sample benchmark cases, tool traces, token counts, latency, and JSON reports.

## Run (Python 3.10+)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: add your own OPENAI_API_KEY; select a model your account supports.
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs to try every endpoint. Do not paste your key into chat or commit .env. OpenAI calls incur API usage charges. No key is bundled.

```bash
curl http://127.0.0.1:8000/agent/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"Find the RxNorm identifier for aspirin and cite the source."}'

curl http://127.0.0.1:8000/evaluate \
  -H 'Content-Type: application/json' -d '{}'

# Or select cases:
curl http://127.0.0.1:8000/evaluate \
  -H 'Content-Type: application/json' -d '{"case_ids":["rxnorm-aspirin"]}'

# Replace REPORT_ID with the id returned by /evaluate:
curl http://127.0.0.1:8000/evaluation/report/REPORT_ID

python -m pytest -q
```

## Architecture and evaluation

Question → OpenAI Responses API → bounded function-calling loop → RxNorm/openFDA → answer and trace → deterministic checks → saved report.

GET /health checks the server. POST /agent/query returns answer, evidence URLs, full tool results, elapsed seconds and token usage. POST /evaluate runs the bundled cases synchronously, saves reports locally, and counts execution errors as failures. GET /evaluation/report/{id} retrieves a saved report. Each case may require expected text, a successful named tool call, and citation of a returned source URL.

These are **heuristic smoke checks**, not factual entailment, clinical safety, prescription validation, or a comprehensive benchmark. A keyword can appear in an incorrect statement; a citation does not establish that the source supports the answer. Review traces manually. Extend data/cases.json with pharmacist-reviewed reference answers, expected tool arguments, negative cases, ambiguous names, prompt injection, service failures, unsupported interaction questions, and personalized-dosing refusal cases before making broader claims. No safety score is claimed here.

Live records can change. Label queries return at most two records and may match multiple products; results are not an exhaustive interaction service. Tool text is bounded and marked when truncated. Upstream failures are visible to the agent. Reports include prompts, answers and retrieved data: use public or synthetic data only. No patient records are needed.

The server is for local development: no authentication, quotas, or production job queue. Keep it bound to localhost. Evaluations can take several minutes and consume multiple API calls. API keys, model access, public API connectivity and live answer quality must be verified in your environment. Automated tests mock external services and do not validate clinical performance.

## Sources

- OpenAI function calling: https://developers.openai.com/api/docs/guides/function-calling
- NLM RxNorm: https://lhncbc.nlm.nih.gov/RxNav/APIs/index.html
- openFDA: https://open.fda.gov/apis/drug/

This product uses publicly available data from the U.S. National Library of Medicine (NLM), National Institutes of Health, Department of Health and Human Services; NLM is not responsible for the product and does not endorse or recommend this or any other product.
