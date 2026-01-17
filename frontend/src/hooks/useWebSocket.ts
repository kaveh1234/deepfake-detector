import { useCallback, useEffect, useRef, useState } from 'react';
import type {
    Metrics,
    TrapResult,
    CalibrationData,
    SessionInfo,
    TrapType,
    DiscoPhase,
} from '../types';

interface UseWebSocketReturn {
    connected: boolean;
    sessionId: string | null;
    metrics: Metrics | null;
    trapResult: TrapResult | null;
    calibration: CalibrationData | null;
    sessionInfo: SessionInfo | null;
    error: string | null;
    connect: () => void;
    disconnect: () => void;
    control: (action: string, data?: any) => void;
    sendFrame: (
        frameBase64: string,
        width: number,
        height: number,
        fps: number,
        isSnapshot?: boolean
    ) => void;
    sendDiscoPhase: (phase: DiscoPhase) => void;
    evaluateTrap: () => void;
}

export function useWebSocket(): UseWebSocketReturn {
    const [connected, setConnected] = useState(false);
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [metrics, setMetrics] = useState<Metrics | null>(null);
    const [trapResult, setTrapResult] = useState<TrapResult | null>(null);
    const [calibration, setCalibration] = useState<CalibrationData | null>(null);
    const [sessionInfo, setSessionInfo] = useState<SessionInfo | null>(null);
    const [error, setError] = useState<string | null>(null);

    const socketRef = useRef<WebSocket | null>(null);
    const reconnectTimeoutRef = useRef<number | undefined>(undefined);

    const connect = useCallback(() => {
        if (socketRef.current?.readyState === WebSocket.OPEN) return;

        // Use absolute URL or relative if proxied
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        // const wsUrl = `${protocol}//${window.location.hostname}:8000/ws`;
        // Using Vite proxy
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        const socket = new WebSocket(wsUrl);

        socket.onopen = () => {
            console.log('WebSocket connected');
            setConnected(true);
            setError(null);
        };

        socket.onclose = () => {
            console.log('WebSocket disconnected');
            setConnected(false);
            socketRef.current = null;

            // Auto reconnect after 2s
            reconnectTimeoutRef.current = window.setTimeout(() => {
                connect();
            }, 2000);
        };

        socket.onerror = (event) => {
            console.error('WebSocket error:', event);
            // setError('Connection failed');
        };

        socket.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                const { type, data } = message;

                switch (type) {
                    case 'session_update':
                        // Update full session info
                        if (data.session_id) setSessionId(data.session_id);
                        setSessionInfo((prev) => ({ ...prev, ...data }));
                        break;

                    case 'metrics':
                        setMetrics(data);
                        break;

                    case 'trap_result':
                        setTrapResult(data);
                        // Append to local history if needed
                        break;

                    case 'calibration_complete':
                        setCalibration(data);
                        break;

                    case 'disco_phase':
                        // Ack or trigger UI
                        break;

                    case 'error':
                        setError(data.message);
                        // Clear error after 5s
                        setTimeout(() => setError(null), 5000);
                        break;

                    default:
                        console.warn('Unknown message type:', type);
                }
            } catch (err) {
                console.error('Failed to parse message:', err);
            }
        };

        socketRef.current = socket;
    }, []);

    const disconnect = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
        }
        if (socketRef.current) {
            socketRef.current.close();
            socketRef.current = null;
        }
    }, []);

    const control = useCallback((action: string, data: any = {}) => {
        if (socketRef.current?.readyState === WebSocket.OPEN) {
            socketRef.current.send(
                JSON.stringify({
                    type: 'control',
                    data: { action, ...data },
                })
            );
        }
    }, []);

    const sendFrame = useCallback(
        (
            frameBase64: string,
            width: number,
            height: number,
            fps: number,
            isSnapshot: boolean = false
        ) => {
            if (socketRef.current?.readyState === WebSocket.OPEN) {
                socketRef.current.send(
                    JSON.stringify({
                        type: isSnapshot ? 'snapshot' : 'frame',
                        data: {
                            frame: frameBase64,
                            width,
                            height,
                            fps,
                        },
                    })
                );
            }
        },
        []
    );

    const sendDiscoPhase = useCallback((phase: DiscoPhase) => {
        if (socketRef.current?.readyState === WebSocket.OPEN) {
            socketRef.current.send(
                JSON.stringify({
                    type: 'disco_phase',
                    data: { phase },
                })
            );
        }
    }, []);

    const evaluateTrap = useCallback(() => {
        if (socketRef.current?.readyState === WebSocket.OPEN) {
            socketRef.current.send(
                JSON.stringify({
                    type: 'evaluate_trap',
                    data: {},
                })
            );
        }
    }, []);

    // Connect on mount
    useEffect(() => {
        connect();
        return () => disconnect();
    }, [connect, disconnect]);

    return {
        connected,
        sessionId,
        metrics,
        trapResult,
        calibration,
        sessionInfo,
        error,
        connect,
        disconnect,
        control,
        sendFrame,
        sendDiscoPhase,
        evaluateTrap,
    };
}
