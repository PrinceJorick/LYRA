import { router } from 'expo-router';
import React, { useState } from 'react';
import { View } from 'react-native';
import { Battery, PodGraphic } from '../components/graphics';
import { HardwareList } from '../components/HardwareList';
import { Button, Card, DemoBadge, Divider, KeyValue, Screen, Txt } from '../components/ui';
import { deviceService } from '../services/device';
import { useApp } from '../state/AppState';
import { space } from '../theme/colors';
import { formatRelative } from '../utils/format';

export default function Device() {
  const { device, setDevice } = useApp();
  const [busy, setBusy] = useState(false);
  const connected = device.connection === 'connected';

  const disconnect = async () => {
    setBusy(true);
    await deviceService.disconnect(device.deviceId);
    setDevice({ ...device, connection: 'disconnected', lastSyncAt: device.lastSyncAt });
    setBusy(false);
  };

  const sync = async () => {
    setBusy(true);
    const r = await deviceService.read(device.deviceId);
    setDevice({ ...device, batteryPercent: r.batteryPercent, lastSyncAt: r.syncedAt });
    setBusy(false);
  };

  return (
    <Screen>
      <View style={{ alignItems: 'center', gap: space.sm }}>
        <DemoBadge />
        <PodGraphic size={180} connected={connected} />
        <Txt v="h2">{device.name}</Txt>
        <Txt muted accessibilityLiveRegion="polite">{connected ? 'Connected (Demo)' : 'Not connected'}</Txt>
      </View>

      <Card style={{ gap: 4 }}>
        <KeyValue label="Connection" value={connected ? 'Connected (simulated)' : 'Disconnected'} />
        <Divider />
        <KeyValue label="Device ID" value={device.deviceId} />
        <Divider />
        <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 6 }}>
          <Txt v="small" muted>Battery</Txt>
          {connected ? <Battery percent={device.batteryPercent} /> : <Txt v="small">Unknown</Txt>}
        </View>
        <Txt v="caption" muted>Mock reading. The MAX17048 fuel gauge is not connected.</Txt>
        <Divider />
        <KeyValue label="Last sync" value={formatRelative(device.lastSyncAt)} />
      </Card>

      {connected ? (
        <View style={{ gap: space.sm }}>
          <Button kind="ghost" label="Sync now (simulated)" disabled={busy} onPress={sync} />
          <Button kind="danger" label="Disconnect Pod" disabled={busy} onPress={disconnect} />
        </View>
      ) : (
        <Button label="Find Pod" onPress={() => router.replace('/pairing')} />
      )}

      <HardwareList />
    </Screen>
  );
}
