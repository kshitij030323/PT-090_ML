# Experiment notes (PT-090) - data processing, features and model comparison

Pipeline (run from the repo root, `dataset/train` and `dataset/student_test` must be present):

```
python -m src.data.build_tables      # candidates + features for all 90 trips -> data/processed/
python -m src.data.candidate_sweep   # evidence for the radius/K choice -> experiments/results/
python -m src.models.train_compare   # 5 models, grouped CV, student-test evaluation
python -m src.models.plot_results    # comparison plots -> visualisations/
python -m visualisations.folium_map --partition student_test --trip 00000052 --model LightGBM
```

## Decisions and assumptions (be ready to defend these in the viva)

1. **Per-trip road network.** Every trip ships its own `.nodes`/`.arcs`, so the network is loaded per trip.
2. **Projection.** The trips span many cities (longitude -100 to +137), so degrees cannot be used as distance.
   Each trip is projected into its own UTM zone (`utm_epsg()` in `src/data/trip_loader.py`); all distances are metres.
3. **Candidate generation.** Up to K=8 arcs within 100 m of each GPS point, found with a shapely STRtree.
   Chosen from the measured trade-off in `experiments/results/candidate_sweep.csv` (miss rate vs candidates per
   observation), using training trips only.
4. **Label rule.** `.route` lists the arcs of the route but not which arc each GPS point belongs to. For each
   observation the reference arc is the **route arc nearest to that observation** (within 100 m); a candidate is
   positive only if it is that arc. This gives **exactly one positive per observation** (verified: 50,140
   observations with 1 positive, 874 with 0), so observation-level Top-1 accuracy is meaningful.
   NOTE: labelling a candidate positive whenever the arc appears *anywhere* in the route is NOT equivalent - it
   produces several positives per observation and inflates every metric.
5. **Candidate miss rate** (reference arc absent from the candidate set) is the ceiling on Top-1 accuracy:
   student-test 0.16%, train 2.80% (`experiments/results/candidate_miss_rate.csv`). The train figure is higher
   because the training trips are much noisier - the EDA offset summary gives a 95th percentile of 139 m on train
   against 17 m on student-test. Two training trips (00000054, 00000033) account for most of the misses.
6. **Sampling.** Training trips use 400 evenly spaced observations each (all candidates of an observation are kept
   together); student-test trips use every observation.
7. **Cross-validation.** `GroupKFold(5)` grouped by trip, so no trip is split across folds. Scalers live inside the
   sklearn pipeline, so they are fitted per fold (no preprocessing leakage).
8. **Metrics.** Macro F1 is candidate-level binary (threshold 0.5); `pos_f1`, precision and recall refer to the
   "correct arc" class. Top-1 = argmax candidate per observation, over observations that have at least one
   candidate (student-test: all of them; train: 99.85% of them). Route continuity = fraction of consecutive
   observations whose predicted arcs are equal or directly connected (end node == next start node).
9. **No leakage of evaluation data.** The dataset contains only `train/` and `student_test/`; no evaluator
   partition exists here, and no feature is derived from `.route`. `.route` is used solely to build labels.
10. **Reproducibility.** Seed 42 throughout; GroupKFold is deterministic. A clean-room re-run from a fresh copy of
    these files reproduced every metric in `model_comparison.csv` exactly (max absolute difference 0.0).

## Model selection

LightGBM is selected: best CV macro F1 (0.9116), best student-test Top-1 (0.8506), lowest log loss (0.1130), best
route continuity (0.9419), and it trains in 3.5 s. The margin over Random Forest and the MLP is about 0.005 CV macro
F1 against a fold-to-fold standard deviation of about 0.02, so the win is consistent but not decisive; LightGBM is
also the cheapest of the three to retune and gives well-calibrated probabilities for the HMM stage.

## Known limitations

- Top-1 excludes the 49 training observations that have no candidate at all (0.15%).
- KNN is fitted on a subsample of about 60k rows for tractability; this is recorded in its configuration string.
- The label rule is an approximation of ground truth, since the reference route itself may have limitations.
