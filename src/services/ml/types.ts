// Initial ML contracts from the project brief. Adjust to match the real pipeline.

export interface SensorWindow {
  timestamp: number;
  emg: number[];
  accelerometer: { x: number[]; y: number[]; z: number[] };
  gyroscope: { x: number[]; y: number[]; z: number[] };
}

export interface ModelMetadata {
  modelId: string;
  version: string;
  status: 'mock' | 'integrated';
}

export interface DetectionResult {
  eventType: 'possible_event' | 'no_event';
  timestamp: number;
  source: 'mock' | 'ml_model';
  modelVersion?: string;
}

/** A structured model output the Insights screen can render. None exist yet. */
export interface ModelInsight {
  id: string;
  label: string;
  value: string;
  timestamp: number;
  modelVersion: string;
}

export interface MlService {
  getMetadata(): ModelMetadata;
  /** Returns human-readable problems; empty array means the window is usable. */
  validateWindow(window: SensorWindow): string[];
  detect(window: SensorWindow): Promise<DetectionResult>;
  getInsights(): ModelInsight[];
}
