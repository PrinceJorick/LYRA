export interface DiscoveredPod {
  id: string;
  name: string;
}

export interface PodReading {
  batteryPercent: number;
  syncedAt: number;
}

/**
 * Contract for talking to the LYRA Pod. Replace mockDeviceService with a BLE-backed
 * implementation once the LYRA BLE protocol is defined; the UI only depends on this interface.
 */
export interface DeviceService {
  scan(): Promise<DiscoveredPod[]>;
  pair(podId: string, onProgress: (percent: number) => void): Promise<PodReading>;
  disconnect(podId: string): Promise<void>;
  /** Battery (MAX17048) and sync info. */
  read(podId: string): Promise<PodReading>;
}
