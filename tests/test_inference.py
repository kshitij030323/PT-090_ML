import sys
import types

import numpy as np


def test_inference_loader_reads_unlabeled_files_and_uses_training_projection(tmp_path, monkeypatch):
    trip_dir = tmp_path / "student_test" / "trip_a"
    trip_dir.mkdir(parents=True)
    (trip_dir / "trip_a.track").write_text("-122.0 37.0 0\n-121.999 37.001 1\n")
    (trip_dir / "trip_a.nodes").write_text("-122.0 37.0\n-121.999 37.001\n")
    (trip_dir / "trip_a.arcs").write_text("0 1\n")
    assert not (trip_dir / "trip_a.route").exists()

    calls = []

    class FakeTransformer:
        @classmethod
        def from_crs(cls, source, target, always_xy):
            calls.append((source, target, always_xy))
            return cls()

        def transform(self, lon, lat):
            return np.asarray(lon) + 1000, np.asarray(lat) + 2000

    fake_pyproj = types.ModuleType("pyproj")
    fake_pyproj.Transformer = FakeTransformer
    monkeypatch.setitem(sys.modules, "pyproj", fake_pyproj)
    # Ensure trip_loader binds to the fake Transformer for this isolated loader test.
    monkeypatch.delitem(sys.modules, "src.data.trip_loader", raising=False)

    from src.data.inference_loader import load_trip_for_inference
    from src.data.trip_loader import utm_epsg

    trip = load_trip_for_inference(tmp_path, "student_test", "trip_a")

    assert calls == [(4326, utm_epsg(-121.9995, 37.0005), True)]
    assert trip.epsg == calls[0][1]
    np.testing.assert_allclose(trip.x, [878.0, 878.001])
    np.testing.assert_allclose(trip.y, [2037, 2037.001])
    np.testing.assert_array_equal(trip.arcs, [[0, 1]])
    assert trip.route.size == 0
    assert not (trip_dir / "trip_a.route").exists()


def _fake_candidate_module(monkeypatch):
    candidates_module = types.ModuleType("src.data.candidates")

    def arc_endpoints(trip):
        return trip.nodes_xy[trip.arcs[:, 0]], trip.nodes_xy[trip.arcs[:, 1]]

    def point_segment(px, py, ax, ay, bx, by):
        vx, vy = bx - ax, by - ay
        length_sq = vx * vx + vy * vy
        wx, wy = px - ax, py - ay
        fraction = np.clip(np.divide(wx * vx + wy * vy, length_sq,
                                     out=np.zeros_like(np.asarray(wx), dtype=float),
                                     where=length_sq != 0), 0, 1)
        distance = np.hypot(px - (ax + fraction * vx), py - (ay + fraction * vy))
        length = np.sqrt(length_sq)
        lateral = np.divide(vx * wy - vy * wx, length,
                            out=np.zeros_like(np.asarray(wx), dtype=float), where=length != 0)
        return distance, fraction, lateral

    def build_arc_tree(trip):
        a, b = arc_endpoints(trip)
        return object(), a, b

    def generate_candidates(trip, obs_idx, radius, k, tree=None, ends=None):
        assert len(obs_idx) == 3
        return {
            "obs_pos": np.array([0, 0, 2]),
            "arc": np.array([0, 1, 1]),
            "d": np.array([1.0, 2.0, 0.5]),
            "frac": np.array([0.5, 0.5, 0.5]),
            "lat": np.array([0.0, 0.0, 0.0]),
            "rank": np.array([0, 1, 0]),
        }

    candidates_module.arc_endpoints = arc_endpoints
    candidates_module.point_segment = point_segment
    candidates_module.build_arc_tree = build_arc_tree
    candidates_module.generate_candidates = generate_candidates
    monkeypatch.setitem(sys.modules, "src.data.candidates", candidates_module)
    monkeypatch.delitem(sys.modules, "src.features.features", raising=False)


def test_predict_trip_uses_existing_features_selects_best_and_handles_empty_observation(monkeypatch):
    _fake_candidate_module(monkeypatch)
    from src import config as C
    from src.features.features import FEATURE_COLS
    from src.inference import predict_trip

    class Model:
        def predict_proba(self, X):
            assert list(X.columns) == FEATURE_COLS
            assert X.shape == (3, 24)
            return np.array([[0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])

    trip = types.SimpleNamespace(
        trip_id="trip_a",
        x=np.array([0.0, 1.0, 2.0]),
        y=np.array([0.0, 0.0, 0.0]),
        t=np.array([0.0, 1.0, 2.0]),
        nodes_xy=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        arcs=np.array([[0, 1], [1, 2]]),
        lonlat_track=np.array([[-122.0, 37.0], [-121.9, 37.1], [-121.8, 37.2]]),
        lonlat_nodes=np.array([[-122.0, 37.0], [-121.9, 37.1], [-121.8, 37.2]]),
    )

    result = predict_trip(trip, Model())

    assert C.CAND_RADIUS_M == 100.0
    assert C.CAND_K == 8
    assert C.WINDOW_OFFSETS == (-2, -1, 1, 2)
    assert result["predicted_arc_ids"] == [1, None, 1]
    assert result["confidence_scores"] == [0.8, None, 0.9]
    assert result["prediction_method"] == "classifier_only"
    assert result["decoder_status"] == "not_implemented"
    assert result["route_geometry"] == {
        "type": "MultiLineString",
        "coordinates": [
            [[-121.9, 37.1], [-121.8, 37.2]],
            [[-121.9, 37.1], [-121.8, 37.2]],
        ],
    }


def test_match_endpoint_and_startup_load_preserve_existing_routes(monkeypatch):
    import asyncio
    import api.main as api

    loaded = object()
    monkeypatch.setattr(api.joblib, "load", lambda path: loaded)

    async def verify_lifespan_loads_once():
        async with api.lifespan(api.app):
            assert api.app.state.lightgbm_model is loaded

    asyncio.run(verify_lifespan_loads_once())

    monkeypatch.setattr(api, "list_trips", lambda partition: ["trip_a"])
    trip = types.SimpleNamespace(trip_id="trip_a")
    monkeypatch.setattr(api, "load_trip_for_inference", lambda *args: trip)
    expected = {"trip_id": "trip_a", "prediction_method": "classifier_only"}
    monkeypatch.setattr(api, "predict_trip", lambda given_trip, model: expected)

    assert api.match_trip("trip_a") == expected
    assert any(route.path == "/trips/{trip_id}/match" and "POST" in route.methods
               for route in api.app.routes)
    assert any(route.path == "/health" for route in api.app.routes)
    assert any(route.path == "/trips/{trip_id}/info" for route in api.app.routes)
