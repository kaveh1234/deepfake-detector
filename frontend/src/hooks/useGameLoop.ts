import { useEffect, useRef, useState, useCallback } from 'react';
import { useWebSocket } from './useWebSocket';
import { useCamera } from './useCamera';
import { TrapType, TrapStatus, DiscoPhase } from '../types';

// Trap sequence: Fisheye -> Squint -> Disco
const TRAP_SEQUENCE: TrapType[] = ['fisheye', 'squint', 'disco'];

// Timing constants (match backend)
const TRAP_DURATION_MS = 3000;  // 3 seconds per trap
const DISCO_PHASE_DURATION_MS = 500;  // 0.5 seconds per phase
const DISCO_PHASES: DiscoPhase[] = ['red', 'green', 'blue'];

export function useGameLoop() {
    const ws = useWebSocket();
    const camera = useCamera();

    // Trap runner state
    const [currentTrapIndex, setCurrentTrapIndex] = useState(-1);
    const [isRunningSequence, setIsRunningSequence] = useState(false);
    const [discoPhaseIndex, setDiscoPhaseIndex] = useState(0);

    // Refs for timers
    const trapTimerRef = useRef<number | null>(null);
    const discoTimerRef = useRef<number | null>(null);

    // Derived state
    const currentTrap = ws.sessionInfo?.current_trap || null;
    const sessionState = ws.sessionInfo?.state || 'idle';

    // Clear all timers
    const clearTimers = useCallback(() => {
        if (trapTimerRef.current) {
            clearTimeout(trapTimerRef.current);
            trapTimerRef.current = null;
        }
        if (discoTimerRef.current) {
            clearInterval(discoTimerRef.current);
            discoTimerRef.current = null;
        }
    }, []);

    // Start the next trap in sequence
    const startNextTrap = useCallback(() => {
        const nextIndex = currentTrapIndex + 1;

        if (nextIndex >= TRAP_SEQUENCE.length) {
            // All traps completed
            setIsRunningSequence(false);
            setCurrentTrapIndex(-1);
            return;
        }

        const trapType = TRAP_SEQUENCE[nextIndex];
        setCurrentTrapIndex(nextIndex);

        // Request backend to start the trap
        ws.control('run_trap', { trap_type: trapType });

        // For disco trap, start phase cycling
        if (trapType === 'disco') {
            setDiscoPhaseIndex(0);
            ws.sendDiscoPhase('red');

            // Cycle through disco phases
            let phaseIdx = 0;
            discoTimerRef.current = window.setInterval(() => {
                phaseIdx++;
                if (phaseIdx < DISCO_PHASES.length) {
                    setDiscoPhaseIndex(phaseIdx);
                    ws.sendDiscoPhase(DISCO_PHASES[phaseIdx]);
                }
            }, DISCO_PHASE_DURATION_MS);
        }

        // Set timer to evaluate trap after duration
        trapTimerRef.current = window.setTimeout(() => {
            // Clear disco timer if running
            if (discoTimerRef.current) {
                clearInterval(discoTimerRef.current);
                discoTimerRef.current = null;
            }

            // Evaluate the trap
            ws.evaluateTrap();
        }, TRAP_DURATION_MS);
    }, [currentTrapIndex, ws, clearTimers]);

    // Watch for calibration complete -> start trap sequence
    useEffect(() => {
        if (sessionState === 'ready' && !isRunningSequence && ws.calibration && currentTrapIndex === -1) {
            // Calibration just completed, start trap sequence
            console.log('Calibration complete, starting trap sequence...');
            setIsRunningSequence(true);
            setCurrentTrapIndex(-1);

            // Small delay before starting first trap
            setTimeout(() => {
                setCurrentTrapIndex(0);
                const trapType = TRAP_SEQUENCE[0];
                ws.control('run_trap', { trap_type: trapType });

                // Set timer to evaluate trap after duration
                trapTimerRef.current = window.setTimeout(() => {
                    ws.evaluateTrap();
                }, TRAP_DURATION_MS);
            }, 500);
        }
    }, [sessionState, isRunningSequence, ws.calibration, currentTrapIndex, ws]);

    // Watch for trap result -> start next trap
    useEffect(() => {
        if (ws.trapResult && isRunningSequence && sessionState === 'ready') {
            // Trap just completed, start next one after a brief delay
            console.log(`Trap ${ws.trapResult.trap_type} completed: ${ws.trapResult.status}`);

            setTimeout(() => {
                startNextTrap();
            }, 500);
        }
    }, [ws.trapResult, isRunningSequence, sessionState, startNextTrap]);

    // Watch for session completion
    useEffect(() => {
        if (sessionState === 'completed') {
            clearTimers();
            setIsRunningSequence(false);
            setCurrentTrapIndex(-1);
        }
    }, [sessionState, clearTimers]);

    // Game Loop for sending frames
    useEffect(() => {
        if (!camera.isStreaming || !ws.connected) return;

        const intervalId = setInterval(() => {
            // Use high-res snapshots during traps/calibration
            const isHighResNeeded = (sessionState === 'running_trap' || sessionState === 'calibrating');
            const frame = isHighResNeeded ? camera.captureSnapshot() : camera.captureFrame();

            if (frame) {
                const width = isHighResNeeded ? (camera.resolution?.width || 1280) : 640;
                const height = isHighResNeeded ? (camera.resolution?.height || 720) : 480;

                ws.sendFrame(
                    frame,
                    width,
                    height,
                    camera.fps,
                    isHighResNeeded
                );
            }
        }, 1000 / 15); // 15 FPS

        return () => clearInterval(intervalId);
    }, [camera, ws, camera.isStreaming, ws.connected, sessionState]);

    // Auto-start session when camera is ready and connected
    useEffect(() => {
        if (ws.connected && camera.isStreaming && !ws.sessionId) {
            console.log("Auto-starting session...");
            ws.control('start_session');
        }
    }, [ws.connected, camera.isStreaming, ws.sessionId, ws]);

    const startSession = useCallback(() => {
        ws.control('start_session');
    }, [ws]);

    const resetSession = useCallback(() => {
        // Clear all local state
        clearTimers();
        setCurrentTrapIndex(-1);
        setIsRunningSequence(false);
        setDiscoPhaseIndex(0);

        // Request backend reset
        ws.control('reset_session');
    }, [ws, clearTimers]);

    // Cleanup on unmount
    useEffect(() => {
        return () => clearTimers();
    }, [clearTimers]);

    return {
        ...ws,
        ...camera,
        startSession,
        resetSession,
        activeTrap: currentTrap,
        trapStatus: (ws.trapResult?.status || 'running') as TrapStatus,
        // Disco state for UI
        discoPhase: DISCO_PHASES[discoPhaseIndex],
        isDiscoActive: currentTrap === 'disco',
        // Sequence progress
        trapSequenceIndex: currentTrapIndex,
        totalTraps: TRAP_SEQUENCE.length,
    };
}
