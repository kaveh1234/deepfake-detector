import './Camera.css';

interface CameraDevice {
  deviceId: string;
  label: string;
}

interface CameraSelectorProps {
  devices: CameraDevice[];
  selectedDevice: string | null;
  onSelectDevice: (deviceId: string) => void;
  isStreaming: boolean;
}

export default function CameraSelector({ devices, selectedDevice, onSelectDevice, isStreaming }: CameraSelectorProps) {
  return (
    <div className="camera-selector">
      <span className="camera-label">Input:</span>
      <select
        value={selectedDevice || ''}
        onChange={(e) => onSelectDevice(e.target.value)}
        disabled={devices.length === 0}
        className="camera-select"
        title={isStreaming ? "Switch camera" : "Select camera"}
      >
        {devices.length === 0 && <option>No devices found</option>}
        {devices.map((device) => (
          <option key={device.deviceId} value={device.deviceId}>
            {device.label}
          </option>
        ))}
      </select>
    </div>
  );
}
