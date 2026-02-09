# yt-thumb-tool

Thumbnail research workspace for YouTube discovery, winners analysis, and reusable visual pattern matching.

## Stack
- Frontend: React + Vite (`frontend/`)
- Backend: FastAPI (`backend/`)
- Deploy: Netlify (frontend) + Render (backend)

## Local Setup
1. Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

2. Frontend
```bash
cd frontend
npm install
npm run dev
```

## Environment Variables

Backend (Render/local):
- `YOUTUBE_API_KEY=<your key>`
- `CORS_ALLOWED_ORIGINS=https://<your-netlify-site>.netlify.app`

Frontend (Netlify/local):
- `VITE_API_BASE_URL=https://<your-render-service>.onrender.com`

## Branching

Main flow:
- `main` = deployable production branch
- `feature/grid-ui`
- `feature/backend-api`
- `feature/pattern-detection`
- `feature/onboarding-flow`
- `feature/spike-pricing`
- `feature/bookmark-patterns`

## Deploy Notes

Netlify:
- Build command: `npm run build`
- Publish dir: `frontend/dist`
- Ensure `VITE_API_BASE_URL` points to Render backend URL.

Render:
- Backend must include Netlify origin in `CORS_ALLOWED_ORIGINS`.
- Redeploy after env var changes.

## Validation

Backend tests:
```bash
python -m pytest backend/tests/test_main.py
```

Frontend production build:
```bash
cd frontend
npm run build
```
