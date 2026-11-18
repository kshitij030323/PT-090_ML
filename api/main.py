
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
import joblib

from src import config as C
from src.data.loader import (
    list_trips,
    load_track,
    load_nodes,
    load_arcs,
)
from src.data.inference_loader import load_trip_for_inference
from src.inference import predict_trip

def load_lightgbm_model():
    """Load and return the fitted classifier for the API process."""
    return joblib.load(C.ARTIFACTS_DIR / "LightGBM.joblib")


@asynccontextmanager
async def lifespan(application):
    application.state.lightgbm_model = load_lightgbm_model()
    yield


app = FastAPI(
    title="PathSnap API",
    description="GPS Map Matching and Route Reconstruction",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def home():
    return {
        "message": "PathSnap API is running",
        "status": "healthy"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/trips")
def get_trips(partition: str = "student_test"):
    if partition not in ["train", "student_test"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid or unavailable partition"
        )

    trips = list_trips(partition)

    return {
        "partition": partition,
        "total_trips": len(trips),
        "trips": trips
    }


@app.get("/trips/{trip_id}/info")
def get_trip_info(trip_id: str, partition: str = "student_test"):
    if partition not in ["train", "student_test"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid or unavailable partition"
        )

    trips = list_trips(partition)

    if trip_id not in trips:
        raise HTTPException(
            status_code=404,
            detail="Trip not found"
        )

    try:
        track = load_track(partition, trip_id)
        nodes = load_nodes(partition, trip_id)
        arcs = load_arcs(partition, trip_id)

        return {
            "trip_id": trip_id,
            "partition": partition,
            "gps_observations": len(track),
            "road_nodes": len(nodes),
            "road_arcs": len(arcs),
            "gps_coordinates": track[
                ["lon", "lat"]
            ].to_dict(orient="records")
        }

    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load trip data: {error}"
        )


@app.post("/trips/{trip_id}/match")
def match_trip(trip_id: str, partition: str = "student_test"):
    if partition not in ["train", "student_test"]:
        raise HTTPException(status_code=400, detail="Invalid or unavailable partition")

    trips = list_trips(partition)
    if trip_id not in trips:
        raise HTTPException(status_code=404, detail="Trip not found")

    model = getattr(app.state, "lightgbm_model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="LightGBM model is not loaded")

    try:
        trip = load_trip_for_inference(C.DATASET_DIR, partition, trip_id)
        return predict_trip(trip, model)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=500, detail=f"Could not match trip: {error}")
