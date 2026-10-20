# Heart Disease Risk Prediction - End-to-End MLOps

Binary classifier (heart disease present / absent) on the UCI Heart Disease (Cleveland) data,
with experiment tracking, CI/CD, Docker, Kubernetes deployment, and monitoring.

## Repository layout

| Path | Purpose |
|---|---|
| `scripts/download_data.py` | Downloads the dataset (UCI first, GitHub mirror fallback) |
| `src/heartml/` | `data.py`, `features.py`, `eda.py`, `train.py`, `api.py`, `config.py` |
| `tests/` | Pytest unit tests (data, features, models, API) |
| `models/` | Packaged pipeline (`model.joblib`) and `model_metadata.json` |
| `reports/` | EDA summary, figures, architecture diagram, final report |
| `screenshots/` | MLflow UI screenshots; add CI, Docker, Kubernetes and Grafana captures here |
| `evidence/` | Real command output: training, pytest, flake8, API calls, request log |
| `Dockerfile`, `docker-compose.yml` | Serving image; API + Prometheus + Grafana stack |
| `k8s/` | Deployment, Service (LoadBalancer), Ingress |
| `monitoring/` | Prometheus config, Grafana datasource and dashboard |
| `.github/workflows/ci.yml` | Lint, test, train, Docker build and smoke test |

## Setup (clean machine)

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python scripts/download_data.py
export PYTHONPATH=src                                  # Windows: set PYTHONPATH=src
```

## Run the pipeline

```bash
python -m heartml.eda          # cleaning + figures in reports/figures
python -m heartml.train        # tuning, CV, MLflow logging, saves models/model.joblib
mlflow ui --backend-store-uri sqlite:///mlflow.db      # http://127.0.0.1:5000
pytest -v                      # unit tests
flake8 src tests scripts       # lint
```

MLflow UI note: if the UI shows "Something went wrong" in a headless or unusual locale
environment, start it with `--gunicorn-opts "--workers 1"`.

## Serve locally

```bash
uvicorn heartml.api:app --port 8000
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,
  "thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

Response: `{"prediction":0,"confidence":0.8146,"probability_disease":0.1854,"model_version":"logistic_regression"}`

PowerShell: `curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" --data "@sample.json"`

Endpoints: `POST /predict`, `GET /health`, `GET /metrics` (Prometheus), `GET /docs` (OpenAPI).

## Docker

```bash
docker build -t heart-api:1.0.0 .
docker run -p 8000:8000 heart-api:1.0.0
```

## Kubernetes (Minikube)
export PYTHONPATH=src            # PowerShell: $env:PYTHONPATH="src"   |  cmd: set PYTHONPATH=src
```bash
minikube start
eval $(minikube docker-env)            # build the image inside Minikube
docker build -t heart-api:1.0.0 .
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
minikube tunnel                        # separate terminal; gives the LoadBalancer an IP
kubectl get svc heart-api              # use EXTERNAL-IP
curl http://<EXTERNAL-IP>/health
# Optional Ingress:  minikube addons enable ingress && kubectl apply -f k8s/ingress.yaml
```
## Access instructions (local testing)

No public URL is deployed. The API runs locally in two ways.

**Docker**
```bash
docker build -t heart-api:1.0.0 .
docker run -p 8000:8000 heart-api:1.0.0
# Swagger UI: http://localhost:8000/docs
```

**Minikube (LoadBalancer)**
```bash
minikube start
minikube docker-env | Invoke-Expression      # PowerShell; Linux/macOS: eval $(minikube docker-env)
docker build -t heart-api:1.0.0 .
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
minikube tunnel                               # keep open in a separate terminal
kubectl get svc heart-api                     # note EXTERNAL-IP
curl http://<EXTERNAL-IP>/health
```
Alternative without a tunnel: `minikube service heart-api --url`.

**Sample request**: use `sample.json` in the repository root.

## Monitoring (Prometheus + Grafana)

```bash
docker compose up --build
# API http://localhost:8000 | Prometheus http://localhost:9090 | Grafana http://localhost:3000 (admin/admin)
```

Grafana auto-loads the "Heart Disease API" dashboard. Structured JSON request logs go to stdout
(`docker logs`); set `LOG_FILE=logs/api.log` for a rotating file.

## Deliverables

| Item | Location |
|---|---|
| Final report | `reports/MLOps_Assignment1_Report_updated.pdf` and `.docx` |
| Screenshots | `screenshots/` |
| Pipeline video | <paste link> |
| CI demo | Branch `lint-fail-demo` fails flake8 on purpose to show the pipeline stops on errors. Do not merge. |
## Data notes

- Source: UCI Heart Disease, Cleveland subset (303 rows, 13 features + target).
- `download_data.py` tries UCI first. If UCI is unreachable it uses a GitHub mirror of the
  cleaned CSV and flips its target so that 1 = disease present (the mirror label is inverted
  relative to UCI; identified from the correlation signs of exang, oldpeak, ca and thalach with the target).
- The target is binarised: 0 = absence, 1 = presence (UCI values 1-4 merged).
- One duplicate row is removed (302 rows used).
