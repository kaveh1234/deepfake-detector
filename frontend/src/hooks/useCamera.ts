import { useCallback, useEffect, useRef, useState } from 'react';

interface CameraDevice {
    deviceId: string;
    label: string;
}

interface UseCameraReturn {
    videoRef: React.RefObject<HTMLVideoElement | null>;
    canvasRef: React.RefObject<HTMLCanvasElement | null>;
    devices: CameraDevice[];
    selectedDevice: string | null;
    isStreaming: boolean;
    resolution: { width: number; height: number } | null;
    fps: number;
    error: string | null;
    selectDevice: (deviceId: string) => void;
    startCamera: () => Promise<void>;
    stopCamera: () => void;
    captureFrame: () => string | null;
    captureSnapshot: () => string | null;
}

export function useCamera(): UseCameraReturn {
    const videoRef = useRef<HTMLVideoElement | null>(null);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const streamRef = useRef<MediaStream | null>(null);

    const [devices, setDevices] = useState<CameraDevice[]>([]);
    const [selectedDevice, setSelectedDevice] = useState<string | null>(null);
    const [isStreaming, setIsStreaming] = useState(false);
    const [resolution, setResolution] = useState<{ width: number; height: number } | null>(null);
    const [fps, setFps] = useState(0);
    const [error, setError] = useState<string | null>(null);

    const frameCountRef = useRef(0);
    const lastFpsUpdateRef = useRef(Date.now());

    // Enumerate available video devices
    const enumerateDevices = useCallback(async () => {
        try {
            // Request permission first
            await navigator.mediaDevices.getUserMedia({ video: true });

            const allDevices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = allDevices
                .filter((device) => device.kind === 'videoinput')
                .map((device) => ({
                    deviceId: device.deviceId,
                    label: device.label || `Camera ${device.deviceId.slice(0, 8)}`,
                }));

            setDevices(videoDevices);

            // Select first device if none selected
            if (videoDevices.length > 0 && !selectedDevice) {
                setSelectedDevice(videoDevices[0].deviceId);
            }
        } catch (err) {
            console.error('Failed to enumerate devices:', err);
            setError('Failed to access camera devices');
        }
    }, [selectedDevice]);

    useEffect(() => {
        enumerateDevices();
    }, [enumerateDevices]);

    const selectDevice = useCallback((deviceId: string) => {
        setSelectedDevice(deviceId);
        if (isStreaming) {
            // Restart camera with new device
            stopCameraInternal();
            setTimeout(() => startCameraInternal(deviceId), 100);
        }
    }, [isStreaming]);

    const startCameraInternal = async (deviceId: string | null) => {
        try {
            const constraints: MediaStreamConstraints = {
                video: {
                    deviceId: deviceId ? { exact: deviceId } : undefined,
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    frameRate: { ideal: 30 },
                },
            };

            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            streamRef.current = stream;

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                await videoRef.current.play();

                // Get actual resolution
                const track = stream.getVideoTracks()[0];
                const settings = track.getSettings();
                setResolution({
                    width: settings.width || 640,
                    height: settings.height || 480,
                });
            }

            setIsStreaming(true);
            setError(null);

            // Start FPS counter
            frameCountRef.current = 0;
            lastFpsUpdateRef.current = Date.now();
        } catch (err) {
            console.error('Failed to start camera:', err);
            setError('Failed to start camera');
            setIsStreaming(false);
        }
    };

    const stopCameraInternal = () => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach((track) => track.stop());
            streamRef.current = null;
        }
        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }
        setIsStreaming(false);
    };

    const startCamera = useCallback(async () => {
        await startCameraInternal(selectedDevice);
    }, [selectedDevice]);

    const stopCamera = useCallback(() => {
        stopCameraInternal();
    }, []);

    const captureFrame = useCallback((): string | null => {
        if (!videoRef.current || !canvasRef.current || !isStreaming) {
            return null;
        }

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        if (!ctx) return null;

        // Use lower resolution for continuous frames
        canvas.width = 640;
        canvas.height = 480;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Update FPS counter
        frameCountRef.current++;
        const now = Date.now();
        if (now - lastFpsUpdateRef.current >= 1000) {
            setFps(frameCountRef.current);
            frameCountRef.current = 0;
            lastFpsUpdateRef.current = now;
        }

        // Return base64 without data URL prefix
        const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
        return dataUrl.split(',')[1];
    }, [isStreaming]);

    const captureSnapshot = useCallback((): string | null => {
        if (!videoRef.current || !canvasRef.current || !isStreaming) {
            return null;
        }

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        if (!ctx) return null;

        // Use full resolution for snapshots
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);

        // Return high-quality base64
        const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
        return dataUrl.split(',')[1];
    }, [isStreaming]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            stopCameraInternal();
        };
    }, []);

    return {
        videoRef,
        canvasRef,
        devices,
        selectedDevice,
        isStreaming,
        resolution,
        fps,
        error,
        selectDevice,
        startCamera,
        stopCamera,
        captureFrame,
        captureSnapshot,
    };
}
