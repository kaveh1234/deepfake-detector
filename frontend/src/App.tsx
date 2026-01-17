import { useGameLoop } from './hooks/useGameLoop';
import VideoFeed from './components/Camera/VideoFeed';
import CameraSelector from './components/Camera/CameraSelector';
import QualityBanner from './components/Camera/QualityBanner';
import TrapManager from './components/Traps/TrapManager';
import LivesDisplay from './components/Feedback/LivesDisplay';
import VerdictScreen from './components/Feedback/VerdictScreen';
import './App.css';

function App() {
  const game = useGameLoop();

  // Derived state for UI
  const metrics = game.metrics;
  const currentTrap = game.sessionInfo?.current_trap || null;
  const trapStatus = game.trapResult?.status || (currentTrap ? 'running' : 'pending');
  const verdict = game.sessionInfo?.verdict || null;
  
  // Lives
  const lives = game.sessionInfo?.lives ?? 3;

  return (
    <div className="app-container">
      {/* Verdict Overlay */}
      <VerdictScreen 
         verdict={verdict} 
         trapResults={game.sessionInfo?.trap_results || []} 
         onReset={game.resetSession}
      />

      <header className="app-header">
        <h1>Deepfake Detector</h1>
        <div className="header-controls">
           <CameraSelector 
              devices={game.devices}
              selectedDevice={game.selectedDevice}
              onSelectDevice={game.selectDevice}
              isStreaming={game.isStreaming}
           />
           <button 
              className="btn btn-primary"
              onClick={game.isStreaming ? game.stopCamera : game.startCamera}
            >
              {game.isStreaming ? 'Stop Camera' : 'Start Camera'}
           </button>
           
           <button 
              className="btn btn-secondary"
              onClick={game.resetSession}
           >
              Reset
           </button>
           
           <LivesDisplay lives={lives} />
        </div>
      </header>

      <main className="app-main">
        <div className="camera-section">
          <QualityBanner fps={game.fps} resolution={game.resolution} />
          <VideoFeed
             videoRef={game.videoRef}
             canvasRef={game.canvasRef}
             isStreaming={game.isStreaming}
             stream={null}
             faceDetected={metrics?.face_detected}
             discoActive={game.isDiscoActive}
             discoPhase={game.discoPhase}
          />
        </div>

        {/* Trap Runner Section */}
        <div className="traps-section">
           <TrapManager 
              currentTrap={currentTrap}
              trapStatus={trapStatus}
              metrics={metrics}
              onStartTrap={() => {}} 
              sessionState={game.sessionInfo?.state}
           />
        </div>
      </main>
    </div>
  );
}

export default App;
