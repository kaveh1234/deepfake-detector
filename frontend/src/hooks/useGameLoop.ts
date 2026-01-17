import { useEffect, useRef, useState } from 'react';
import { useWebSocket } from './useWebSocket';
import { useCamera } from './useCamera';
import { TrapType, TrapStatus, Metrics, SessionState } from '../types';

export function useGameLoop() {
    const ws = useWebSocket();
    const camera = useCamera();

    // Local state for UI that might not come directly from metrics or needs smoother updates
    const [activeTrap, setActiveTrap] = useState<TrapType | null>(null);
    const [trapStatus, setTrapStatus] = useState<TrapStatus>('pending');

    // Derived current trap from session info
    const currentTrap = ws.sessionInfo?.current_trap || null;
    const sessionState = ws.sessionInfo?.state || 'idle';

    // Game Loop for sending frames
    useEffect(() => {
        if (!camera.isStreaming || !ws.connected) return;

        const intervalId = setInterval(() => {
            // Determine if we should send a frame
            // In 'running_trap' or 'calibrating' state, we send frames continuously

            // P0: Snapshot Fix - Use High-Res for analysis steps (traps)
            // If we are in a trap, use captureSnapshot (High Quality, Full Res)
            // Otherwise use captureFrame (Low Quality, Scaled)
            const isHighResNeeded = (sessionState === 'running_trap' || sessionState === 'calibrating');

            const frame = isHighResNeeded ? camera.captureSnapshot() : camera.captureFrame();

            if (frame) {
                // If high res, we are sending the full resolution
                const width = isHighResNeeded ? (camera.resolution?.width || 1280) : 640;
                const height = isHighResNeeded ? (camera.resolution?.height || 720) : 480;

                ws.sendFrame(
                    frame,
                    width,
                    height,
                    camera.fps,
                    isHighResNeeded // Treat as snapshot/high-res frame
                );
            }
        }, 1000 / 15); // Cap at 15 FPS

        return () => clearInterval(intervalId);
    }, [camera, ws, camera.isStreaming, ws.connected, sessionState]);

    const startSession = () => {
        ws.control('start_session');
    };

    const resetSession = () => {
        ws.control('reset_session');
    };

    return {
        ...ws,
        ...camera,
        startSession,
        resetSession,
        activeTrap: currentTrap,
        trapStatus: (ws.trapResult?.status || 'running') as TrapStatus,
    };
}
