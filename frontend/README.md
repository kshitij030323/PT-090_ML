# PathSnap evaluator frontend

# PathSnap evaluator frontend

Responsive React + Vite dashboard for reviewing classifier-only map matching
on the `student_test` partition. It reads trip IDs and GPS coordinates from the
FastAPI endpoints, then requests predictions only when **Run map matching** is
pressed. It never accesses reference `.route` files or calculates accuracy.

The dashboard shows the GPS trajectory, predicted road geometry, start and end
points, summary counts, per-observation arc IDs and confidence scores. The
confidence score is a classifier score, not accuracy. HMM/Viterbi decoding is
not implemented in the current API output.

## Start

From the repository root, start the backend in one terminal:

```bash
uvicorn api.main:app --reload --port 8000
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL (normally `http://localhost:5173`). The API must be available
at `http://localhost:8000`; its CORS allowlist includes the two local Vite
origins. To point the frontend at a different API address, set
`VITE_API_BASE_URL` before starting Vite, for example:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

Build for production with `npm run build`.

The build output is written to `frontend/dist/`. For production, configure the
API's CORS allowlist for the deployed frontend origin and serve the build from
a static web server.
