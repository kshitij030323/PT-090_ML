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

## Repository Layout

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

## Run the API and match a trip

Start the API from the project root:

```bash
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/trips/00000076/match` in a browser to get a match
for a student-test trip. Change the trip ID to another ID from
`/trips?partition=student_test`; use `partition=train` only for training trips.
The match endpoint supports GET for direct browser use and POST for clients.
`/docs` remains available for exploring the API.

Place the provided dataset in `dataset/` (80 train / 10 student-test trips) with structure:
```
dataset/
├── train/
│   ├── 00000001/
│   │   ├── 00000001.track
│   │   ├── 00000001.nodes
│   │   ├── 00000001.arcs
│   │   └── 00000001.route
│   └── ...
└── student_test/
    ├── 00000000/
    │   ├── 00000000.track
    │   ├── 00000000.nodes
    │   ├── 00000000.arcs
    │   └── 00000000.route
    └── ...
```

## Person 1 – Dataset & Preprocessing (COMPLETED)

### 1. Data Loading (`src/data/loader.py`)

Loads raw whitespace-separated files without headers:
- **`.track`**: 3 columns → `lon`, `lat`, `t` (timestamp in seconds, nominally 1 Hz)
- **`.nodes`**: 2 columns → `lon`, `lat` (0-based row index = node_id)
- **`.arcs`**: 2 columns → `from_node`, `to_node` (0-based row index = arc_id)
- **`.route`**: 1 column → `arc_id` (ground-truth sequence of road segments)

```python
from src.data import loader as L
track = L.load_track('train', '00000001')  # DataFrame[lon, lat, t]
nodes = L.load_nodes('train', '00000001')  # DataFrame[lon, lat]
arcs  = L.load_arcs('train', '00000001')   # DataFrame[from_node, to_node]
route = L.load_route('train', '00000001')  # ndarray[arc_id]
```

### 2. Exploratory Data Analysis (`src/data/eda.py`, `visualisations/make_eda_plots.py`)

**Run EDA:**
```bash
python -m src.data.eda
python visualisations/make_eda_plots.py
```

**Outputs generated in `experiments/results/`:**
| File | Description |
|---|---|
| `eda_trip_inventory.csv` | Per-trip statistics (90 trips × 33 metrics) |
| `eda_missing_values.csv` | Missing value summary per partition |
| `eda_gps_offset_summary.csv` | GPS-to-route distance percentiles |
| `eda_gps_offset_per_trip.csv` | Per-trip median/P95/max offset |
| `eda_gps_offset_from_route.csv` | All point-level offsets (19K test + 208K train) |

**Key EDA Findings:**
- **80 train trips**, **10 student-test trips** — no evaluator test data used
- **Zero missing values** across all files (`n_missing=0`)
- **Timestamps sorted** (`t_sorted=True` for all trips)
- **No duplicate timestamps** (`n_dup_t=0`)
- **GPS sampling**: ~97% steps exactly 1s apart (`frac_dt_1 ≈ 0.97`)
- **GPS-to-reference-route offset**: Train median 5.6m (P95 139m, max 567m); Test median 5.1m (P95 16.7m, max 105m)
- **Trip duration**: 35s – 57.8h; **Observations**: 250 – 19,313 per trip
- **Road network**: 674 – 123,433 nodes; 1,554 – 259,437 arcs per trip
- **Route length**: 59 – 2,266 arcs per trip
- **Network connectivity**: 1 – 213 connected components; 0 isolated nodes

**Visualisations in `visualisations/`:**
- `eda_distributions.png` — Histograms of observations, time gaps, arcs, components, extent
- `eda_trip_locations.png` — Geographic centroids of all trips
- `eda_gps_offset.png` — GPS noise distribution (clipped at 100m) + per-trip P95
- `eda_trajectory_network_route.png` — Static plot: GPS pts (blue), network (gray), route (green)
- `folium_train_00000061.html` / `folium_student_test_00000076.html` — Interactive maps with layer controls

### 3. Candidate Generation (`src/features/__init__.py` → `generate_candidates()`)

For each GPS point, find nearby road segments (arcs) within a search radius.

**Algorithm:**
1. Project all nodes to local XY (metres) using equirectangular projection centered at trip mean lat/lon
2. Build KD-tree on **arc midpoints** for fast spatial queries
3. For each GPS point: query KD-tree with `search_radius = 2 × radius_m` (catches arcs whose midpoint is far but segment passes close)
4. For each retrieved arc: compute exact point-to-segment distance
5. Keep arcs with `distance ≤ radius_m`, sort by distance, take top `max_candidates`

**Parameters:**
```python
radius_m = 200.0      # Maximum perpendicular distance from GPS to arc segment (metres)
max_candidates = 30   # Maximum candidates per GPS point
search_radius = 400.0 # KD-tree query radius (2× radius_m)
```

**Why 200m radius?** EDA showed GPS offsets up to 567m (train) and 105m (test). 200m captures >98% of true route arcs while limiting candidate count.

**Coverage Results:**
| Partition | Trips | Points with Candidates | Route Arc in Candidates |
|---|---|---|---|
| Train | 80/80 | 99.8% avg | 96% avg |
| Student-test | 10/10 | 95% avg | 93% avg |

Trips with large GPS noise (e.g., 00000033, 00000054) have lower coverage — this is expected and handled in HMM stage.

### 4. Feature Engineering (`src/features/__init__.py` → `extract_features()`)

**15 features per (GPS point, candidate arc) pair:**

| Feature | Description | Type |
|---|---|---|
| `dist_to_arc` | Perpendicular distance from GPS point to arc segment (m) | Continuous |
| `proj_t` | Normalized projection position along arc [0,1] | Continuous |
| `along_dist` | Distance from arc start to projection point (m) | Continuous |
| `arc_length` | Total arc length (m) | Continuous |
| `arc_bearing` | Arc direction (radians, atan2) | Circular |
| `gps_speed` | Instantaneous GPS speed (m/s) | Continuous |
| `gps_bearing` | GPS heading (radians) | Circular |
| `hdg_diff` | Absolute heading difference (radians, wrapped to π) | Continuous |
| `dt` | Time since previous GPS point (s) | Continuous |
| `fr_deg` | Degree of arc's from-node | Integer |
| `to_deg` | Degree of arc's to-node | Integer |
| `connected_to_route` | Binary: arc shares a node with any route arc | Binary |
| `dist_to_nearest_route` | Min distance to any route arc segment (m) | Continuous |
| `obs_idx` | GPS observation index within trip | Integer |
| `arc_id` | Candidate arc identifier | Integer |

**Label:** `is_route` (1 if candidate arc matches ground-truth route at that point, else 0)

**Dataset Statistics:**
| Partition | Rows | Trips | Positive Rate | Features |
|---|---|---|---|---|
| Train | 1,516,836 | 80 | 32.3% | 15 + label + trip_id |
| Student-test | 171,087 | 10 | 37.7% | 15 + label + trip_id |

**Saved to:** `data/processed/train_features.parquet`, `data/processed/test_features.parquet`

### 5. Preprocessing Pipeline (for Model Training)

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

# Load
train = pd.read_parquet('data/processed/train_features.parquet')
test  = pd.read_parquet('data/processed/test_features.parquet')

# Feature columns (exclude metadata)
feature_cols = [c for c in train.columns if c not in 
                ['is_route', 'trip', 'partition', 'obs_idx', 'arc_id']]

X_train = train[feature_cols]
y_train = train['is_route']
groups  = train['trip']  # For group-aware CV

X_test = test[feature_cols]
y_test = test['is_route']

# Scale (fit on train only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# GroupKFold (trip-aware, no leakage)
cv = GroupKFold(n_splits=5)
```

**Circular features** (`arc_bearing`, `gps_bearing`, `hdg_diff`) are kept as radians — tree-based models handle them natively; for linear models, add sin/cos transforms.

## Run Commands

```bash
# 1. Run EDA + generate visualisations
python -m src.data.eda
python visualisations/make_eda_plots.py

# 2. Build feature datasets (already done, outputs in data/processed/)
python -c "from src.features import build_dataset; build_dataset('train'); build_dataset('student_test')"

# 3. Person 2 pipeline: candidates/features -> 5 models -> comparison plots
python -m src.data.build_tables      # candidate + feature table for all 90 trips -> data/processed/
python -m src.data.candidate_sweep   # evidence for the radius/K choice -> experiments/results/
python -m src.models.train_compare   # 5 models, GroupKFold CV, student-test evaluation
python -m src.models.plot_results    # comparison plots -> visualisations/
python -m visualisations.folium_map --partition student_test --trip 00000052 --model LightGBM
```

## Rules We Follow

- Group-aware CV by trip; no trip split across partitions
- Evaluator-test data/labels never used in training, tuning or UI; ground truth stays server-side
- No hardcoded predictions. All results in the report come from code in this repo
- Preprocessing fitted on train only, applied to test


## Status

- [x] **EDA** — Complete (80 training trips and 10 student-test trips)
- [x] **Feature Engineering** — Candidate generation and feature extraction completed
- [x] **Model Comparison** — Five models evaluated using GroupKFold cross-validation
- [x] **LightGBM Selection** — Selected based on model comparison results
- [x] **Bayesian Optimisation** — 25 Optuna trials using group-aware cross-validation
- [x] **HMM / Viterbi** — Implemented transition probabilities and sequence decoding
- [x] **Four-way Evaluation** — Compared tuned/untuned LightGBM with/without HMM
- [ ] **API** — FastAPI backend
- [ ] **UI** — Evaluator interface
- [ ] **Report** — Technical report and final documentation


## Person 2 – Model Experimentation (COMPLETED)

Code: `src/config.py`, `src/data/trip_loader.py`, `src/data/candidates.py`, `src/data/build_tables.py`,
`src/features/features.py`, `src/models/{metrics,train_compare,plot_results}.py`.
Assumptions and decisions: `experiments/NOTES.md`. Results: `experiments/results/`.

Each row is one (GPS observation, candidate arc) pair; the classifier predicts P(candidate is the reference arc).
Candidates: up to K=8 arcs within 100 m, each trip projected into its own UTM zone. Exactly one positive per
observation (the nearest route arc), so observation-level Top-1 accuracy is meaningful.
CV is `GroupKFold(5)` by trip; scalers sit inside the sklearn pipeline, so they are fitted per fold.

| Model | Family | CV macro F1 | Student Top-1 | Student macro F1 | Log loss | Route continuity | Train s | Infer ms/obs |
|---|---|---|---|---|---|---|---|---|
| LogReg | Linear | 0.8788 | 0.8341 | 0.8888 | 0.1511 | 0.9281 | 0.5 | 0.001 |
| KNN | Instance-based | 0.8911 | 0.8369 | 0.8973 | 0.1761 | 0.9377 | 0.02 | 0.73 |
| RandomForest | Ensemble | 0.9070 | 0.8490 | 0.9060 | 0.1159 | 0.9316 | 41.3 | 0.030 |
| **LightGBM** | **Boosting** | **0.9116** | **0.8506** | **0.9087** | **0.1130** | **0.9419** | 3.4 | 0.038 |
| MLP | Neural network | 0.9060 | 0.8491 | 0.9062 | 0.1204 | 0.9408 | 18.4 | 0.004 |

**Selected: LightGBM** — best CV macro F1, student-test Top-1, log loss and route continuity, and cheap to retune.
The margin over Random Forest and the MLP is about 0.005 CV macro F1 against a fold standard deviation of about
0.02, so the win is consistent but not decisive.

Candidate miss rate (the ceiling on Top-1): 0.16% student-test, 2.80% train
(`experiments/results/candidate_miss_rate.csv`); the training trips are far noisier than the student-test trips.


## Person 3 – Optimisation & Route Reconstruction

### 1. Bayesian Hyperparameter Optimisation

Bayesian optimisation was performed on the selected LightGBM model using Optuna.

- Number of trials: 25
- Validation: 5-fold GroupKFold cross-validation
- Grouping: By trip to prevent data leakage
- Optimisation objective: Macro F1-score
- Best cross-validation Macro F1: **0.91289**

The tuned model was compared against the original LightGBM model on the student-test partition.

| Metric | Untuned | Tuned |
|---|---:|---:|
| Student Top-1 Accuracy | 85.14% | 85.20% |
| Macro F1 | 90.77% | 90.92% |
| Log Loss | 0.11403 | 0.11367 |

Best parameters are saved in `models/artifacts/lightgbm_best_params.json`.

### 2. HMM-Based Route Reconstruction

An HMM-based sequence decoding approach was implemented to improve route consistency.

- **Emission probabilities:** Candidate probabilities generated by LightGBM.
- **Transition probabilities:** Based on road-network connectivity and the relationship between network distance and GPS displacement.
- **Viterbi algorithm:** Finds the most likely sequence of candidate road segments across GPS observations.
- **Output:** A reconstructed sequence of road-segment IDs for each trip.

Implementation:
- `src/hmm/transition_probability.py`
- `src/hmm/viterbi.py`

### 3. Four-Way Model Comparison

Four configurations were evaluated on all 10 student-test trips.

| Model | Observation-Weighted Accuracy | Route Continuity |
|---|---:|---:|
| Untuned LightGBM | 85.14% | 94.22% |
| Untuned LightGBM + HMM | 87.86% | 97.83% |
| Tuned LightGBM | 85.20% | 94.24% |
| Tuned LightGBM + HMM | 87.86% | 97.78% |

**Key observations:**

- Bayesian optimisation provided a small improvement over the original LightGBM model.
- HMM-based decoding improved observation-weighted accuracy by approximately 2.7 percentage points.
- Route continuity improved from approximately 94.2% to 97.8%.
- HMM improved results on most trips, although accuracy decreased on some individual trips.

### 4. Result Files

| File | Description |
|---|---|
| `experiments/results/bayesian_trials.csv` | Optuna trial history |
| `experiments/results/tuned_vs_untuned.csv` | Tuned vs untuned comparison |
| `experiments/results/four_model_route_comparison.csv` | Per-trip four-way comparison |
| `experiments/results/four_model_route_summary.csv` | Aggregate evaluation results |

### 5. Evaluation Integrity

- Training and optimisation use the training partition.
- Final comparisons use the student-test partition.
- Evaluator-test data and labels are not used for training, tuning, or evaluation.
