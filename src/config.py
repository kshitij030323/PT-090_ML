"""Central configuration for PathSnap (team PT-090). Change values here, not inside the pipeline code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"          # contains train/ and student_test/ (evaluator data is NEVER stored here)
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "experiments" / "results"
ARTIFACTS_DIR = ROOT / "models" / "artifacts"

SEED = 42

# Candidate generation (metres, in each trip's local UTM projection)
CAND_RADIUS_M = 100.0      # arcs farther than this from a GPS point are not candidates
CAND_K = 8                 # keep at most the K nearest candidate arcs per GPS observation
WINDOW_OFFSETS = (-2, -1, 1, 2)   # neighbouring GPS points used for context features

# Development-table sampling (training trips only): evenly spaced observations per trip.
# Student-test trips always use ALL observations.
TRAIN_MAX_OBS_PER_TRIP = 400
CV_FOLDS = 5
