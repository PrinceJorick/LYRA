import { DEMO_DEVICE_ID } from '../../data/mockDevice';
import type { DeviceService } from './types';

const wait = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

// No Bluetooth involved: everything below is simulated.
export const mockDeviceService: DeviceService = {
  async scan() {
    await wait(2200);
    return [{ id: DEMO_DEVICE_ID, name: 'LYRA Pod' }];
  },
  async pair(_podId, onProgress) {
    for (let p = 0; p <= 100; p += 20) {
      onProgress(p);
      await wait(350);
    }
    return { batteryPercent: 82, syncedAt: Date.now() };
  },
  async disconnect() {
    await wait(300);
  },
  async read() {
    return { batteryPercent: 82, syncedAt: Date.now() };
  },
};
