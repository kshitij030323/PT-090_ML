# PathSnap – Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning – PES University, RR Campus
**Team: PT-090**

| Name | Team |
|---|---|
| Kshitij Gupta | PT-090 |
| Maansi S | PT-090 |
| Krish Mathur | PT-090 |
| Shridhar Umesh | PT-090 |

## Pipeline
Raw GPS (.track) + road network (.nodes/.arcs) → candidate generation → feature engineering →
classifier comparison (≥5 models) → Bayesian hyperparameter optimisation → HMM + Viterbi decoding →
FastAPI backend (Laptop 1) → evaluator UI (Laptop 2).

## Repository layout
| Folder | Contents |
|---|---|
| `data/` | Raw & processed data (raw is git-ignored; evaluator labels never stored here) |
| `src/data` | Reading, inspection, cleaning, preprocessing, candidate generation |
| `src/features` | Feature engineering |
| `src/models` | Training/evaluation code for every model, metrics, confusion matrices |
| `src/hmm` | HMM + Viterbi route reconstruction |
| `experiments/` | Logs, Bayesian-optimisation trials, result CSVs, seeds/configs |
| `models/artifacts` | Saved model weights, preprocessors, HMM config |
| `api/` | FastAPI backend (Laptop 1) |
| `ui/` | Evaluator UI (Laptop 2) |
| `visualisations/` | Folium maps and plotting scripts |
| `notebooks/` | EDA and experiment notebooks |
| `report/` | Technical report (PDF) |
| `configs/` | Configuration files |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Place the provided dataset in `data/raw/` (80 train / 10 student-test trips).

## Run (to be completed)
- API (Laptop 1): `uvicorn api.main:app --host 0.0.0.0 --port 8000`
- UI (Laptop 2): _TBD_

## Rules we follow
- Group-aware CV by trip; no trip split across partitions.
- Evaluator-test data/labels never used in training, tuning or the UI; ground truth stays server-side.
- No hardcoded predictions. All results in the report come from code in this repo.

## Status
- [ ] EDA  - [ ] Features  - [ ] 5+ models  - [ ] Bayesian opt  - [ ] HMM/Viterbi  - [ ] API  - [ ] UI  - [ ] Report
