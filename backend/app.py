"""FastAPI WebSocket server for deepfake detection."""

import asyncio
import json
import base64
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from models import (
    WSMessageType, ControlAction, TrapType, DiscoPhase,
    SessionState
)
from session import SessionManager


# Global session manager
session_manager: Optional[SessionManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global session_manager
    session_manager = SessionManager()
    yield
    if session_manager:
        session_manager.cleanup()


app = FastAPI(
    title="Deepfake KYC Detector",
    description="Real-time deepfake detection via webcam",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)


manager = ConnectionManager()


async def send_response(
    websocket: WebSocket,
    msg_type: WSMessageType,
    data: dict
):
    """Send a response message."""
    await websocket.send_json({
        "type": msg_type.value,
        "data": data
    })


async def handle_control(
    websocket: WebSocket,
    session_id: str,
    action: str,
    data: dict
):
    """Handle control messages from client."""
    global session_manager

    if action == ControlAction.START_SESSION.value:
        session = session_manager.create_session()
        session_id = session.session_id
        await send_response(websocket, WSMessageType.SESSION_UPDATE, {
            "session_id": session_id,
            "state": session.state.value,
            "lives": session.lives,
            "message": "Session started"
        })
        return session_id

    elif action == ControlAction.RESET_SESSION.value:
        session = session_manager.reset_session(session_id)
        await send_response(websocket, WSMessageType.SESSION_UPDATE, {
            "session_id": session_id,
            "state": session.state.value,
            "lives": session.lives,
            "message": "Session reset"
        })

    elif action == ControlAction.START_CALIBRATION.value:
        success = session_manager.start_calibration(session_id)
        if success:
            await send_response(websocket, WSMessageType.SESSION_UPDATE, {
                "session_id": session_id,
                "state": SessionState.CALIBRATING.value,
                "message": "Calibration started - hold still"
            })
        else:
            await send_response(websocket, WSMessageType.ERROR, {
                "message": "Failed to start calibration"
            })

    elif action == ControlAction.RUN_TRAP.value:
        trap_type_str = data.get("trap_type")
        if trap_type_str:
            try:
                trap_type = TrapType(trap_type_str)
                success = session_manager.start_trap(session_id, trap_type)
                if success:
                    await send_response(websocket, WSMessageType.SESSION_UPDATE, {
                        "session_id": session_id,
                        "state": SessionState.RUNNING_TRAP.value,
                        "current_trap": trap_type.value,
                        "message": f"Running {trap_type.value} trap"
                    })
                else:
                    await send_response(websocket, WSMessageType.ERROR, {
                        "message": f"Failed to start {trap_type_str} trap"
                    })
            except ValueError:
                await send_response(websocket, WSMessageType.ERROR, {
                    "message": f"Unknown trap type: {trap_type_str}"
                })

    return session_id


async def handle_frame(
    websocket: WebSocket,
    session_id: str,
    data: dict,
    is_snapshot: bool = False
):
    """Handle incoming video frame."""
    global session_manager

    # Decode base64 frame data
    frame_b64 = data.get("frame")
    if not frame_b64:
        return

    try:
        frame_bytes = base64.b64decode(frame_b64)
    except Exception:
        return

    width = data.get("width", 640)
    height = data.get("height", 480)

    # Check quality gate
    fps = data.get("fps", 30)
    quality_ok, quality_msg = session_manager.check_quality_gate(
        session_id, width, height, fps
    )

    # Process frame
    if is_snapshot:
        metrics = session_manager.process_snapshot(
            session_id, frame_bytes, width, height
        )
    else:
        metrics = session_manager.process_frame(
            session_id, frame_bytes, width, height
        )

    if metrics:
        metrics["quality_ok"] = quality_ok
        metrics["quality_message"] = quality_msg

        await send_response(websocket, WSMessageType.METRICS, metrics)

        # Check for calibration complete
        session = session_manager.get_session(session_id)
        if session and session.state == SessionState.READY and session.calibration:
            await send_response(websocket, WSMessageType.CALIBRATION_COMPLETE, {
                "laplacian_variance": session.calibration.laplacian_variance,
                "nose_face_ratio": session.calibration.nose_face_ratio,
                "edge_count": session.calibration.edge_count,
            })


async def handle_disco_phase(
    websocket: WebSocket,
    session_id: str,
    phase: str
):
    """Handle disco phase change."""
    global session_manager

    try:
        disco_phase = DiscoPhase(phase)
        session_manager.set_disco_phase(session_id, disco_phase)
        await send_response(websocket, WSMessageType.DISCO_PHASE, {
            "phase": phase
        })
    except ValueError:
        pass


async def handle_evaluate_trap(websocket: WebSocket, session_id: str):
    """Evaluate the current trap and send result."""
    global session_manager

    result = session_manager.evaluate_trap(session_id)
    if result:
        await send_response(websocket, WSMessageType.TRAP_RESULT, {
            "trap_type": result.trap_type.value,
            "status": result.status.value,
            "score": result.score,
            "baseline": result.baseline,
            "threshold": result.threshold,
            "penalty": result.penalty,
            "message": result.message,
        })

        # Send session update
        session_info = session_manager.get_session_info(session_id)
        if session_info:
            await send_response(websocket, WSMessageType.SESSION_UPDATE, session_info)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time communication."""
    await websocket.accept()

    session_id = ""

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await send_response(websocket, WSMessageType.ERROR, {
                    "message": "Invalid JSON"
                })
                continue

            msg_type = message.get("type")
            msg_data = message.get("data", {})

            if msg_type == WSMessageType.CONTROL.value:
                action = msg_data.get("action")
                result_session_id = await handle_control(
                    websocket, session_id, action, msg_data
                )
                if result_session_id:
                    session_id = result_session_id

            elif msg_type == WSMessageType.FRAME.value:
                await handle_frame(websocket, session_id, msg_data, is_snapshot=False)

            elif msg_type == WSMessageType.SNAPSHOT.value:
                await handle_frame(websocket, session_id, msg_data, is_snapshot=True)

            elif msg_type == "disco_phase":
                phase = msg_data.get("phase")
                if phase:
                    await handle_disco_phase(websocket, session_id, phase)

            elif msg_type == "evaluate_trap":
                await handle_evaluate_trap(websocket, session_id)

    except WebSocketDisconnect:
        manager.disconnect(session_id)


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "deepfake-detector"}


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get session information."""
    global session_manager

    info = session_manager.get_session_info(session_id)
    if info:
        return info
    return {"error": "Session not found"}


# Serve static files for frontend
import os
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))

if os.path.exists(frontend_path):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_path, "assets")), name="assets")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_path, "index.html"))

    @app.get("/{path:path}")
    async def serve_frontend_path(path: str):
        file_path = os.path.join(frontend_path, path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_path, "index.html"))
