import { RefObject } from 'react';
import FaceOverlay from '../Feedback/FaceOverlay';
import DiscoFlash from '../Traps/DiscoFlash';
import './Camera.css';

interface VideoFeedProps {
    videoRef: RefObject<HTMLVideoElement | null>;
    canvasRef: RefObject<HTMLCanvasElement | null>;
    isStreaming: boolean;
    stream: MediaStream | null;
    faceDetected?: boolean;
    // Disco specific props
    discoActive?: boolean;
    discoPhase?: 'red' | 'green' | 'blue';
}

export default function VideoFeed({ videoRef, canvasRef, isStreaming, faceDetected = false, discoActive = false, discoPhase = 'red' }: VideoFeedProps) {
    return (
        <div className="video-feed-container">
            {!isStreaming && (
                <div className="video-feed-msg">
                    <p>Camera is off</p>
                </div>
            )}
            <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="video-element" 
            />
            {/* Hidden canvas for capturing frames */}
            <canvas ref={canvasRef} className="hidden-canvas" />
            
            {/* Disco Flash Overlay */}
            <DiscoFlash isActive={discoActive} phase={discoPhase} />

            <div className="overlay-container">
                 <FaceOverlay faceDetected={faceDetected} />
            </div>
        </div>
    );
}
