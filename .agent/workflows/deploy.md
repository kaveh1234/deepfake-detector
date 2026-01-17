---
description: how to deploy the application
---

This guide outlines how to deploy the Deepfake Detector as a single unified application (FastAPI serving the React frontend).

## Prerequisites
- Node.js and npm installed
- Python 3.12+ with `uv` installed
- A server with a public IP or a hosting platform (like Render, Railway, or DigitalOcean)

## Deployment Steps

### 1. Build the Frontend
Navigate to the `frontend` directory and build the production assets.
```bash
cd frontend
npm install
npm run build
```
This will create a `dist` folder in `frontend/`.

### 2. Prepare the Backend
Ensure the backend is configured to serve the static files. The `app.py` is already set up to look for `../frontend/dist`.

### 3. Install Dependencies on the Server
On your production server:
```bash
cd backend
uv sync --production
```

### 4. Run with a Production Server
Use `uvicorn` (or `gunicorn` with uvicorn workers for better scaling) to run the application.

**Using Uvicorn:**
```bash
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

### 5. HTTPS and SSL (CRITICAL)
For WebRTC (webcam access) to work, the application **MUST** be served over **HTTPS**.
- Use a reverse proxy like **Nginx** or **Caddy** to handle SSL.
- Ensure your WebSocket configuration in Nginx supports the `Upgrade` header.

#### Example Nginx Config:
```nginx
server {
    listen 443 ssl;
    server_name yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

## Summary for Quick Launch
// turbo
1. Build frontend: `cd frontend && npm run build`
2. Start production server: `cd backend && uv run uvicorn app:app --host 0.0.0.0 --port 8000`
