import { router } from 'expo-router';
import { Check } from 'lucide-react-native';
import React, { useEffect, useRef, useState } from 'react';
import { ActivityIndicator, View } from 'react-native';
import { PodGraphic } from '../components/graphics';
import { Button, Card, DemoBadge, Notice, Screen, Txt } from '../components/ui';
import { deviceService, type DiscoveredPod } from '../services/device';
import { useApp } from '../state/AppState';
import { space } from '../theme/colors';
import { useTheme } from '../theme/useTheme';

type Phase = 'idle' | 'scanning' | 'found' | 'pairing' | 'done' | 'error';

export default function Pairing() {
  const c = useTheme();
  const { device, setDevice } = useApp();
  const [phase, setPhase] = useState<Phase>('idle');
  const [pod, setPod] = useState<DiscoveredPod | null>(null);
  const [progress, setProgress] = useState(0);
  const alive = useRef(true);
  useEffect(() => () => { alive.current = false; }, []);

  // Already paired (e.g. deep link): nothing to do here.
  useEffect(() => {
    if (device.connection === 'connected' && phase === 'idle') router.replace('/device');
  }, [device.connection, phase]);

  const scan = async () => {
    setPhase('scanning');
    try {
      const found = await deviceService.scan();
      if (!alive.current) return;
      setPod(found[0] ?? null);
      setPhase(found[0] ? 'found' : 'error');
    } catch {
      if (alive.current) setPhase('error');
    }
  };

  const pair = async () => {
    if (!pod) return;
    setPhase('pairing');
    setProgress(0);
    try {
      const reading = await deviceService.pair(pod.id, (p) => alive.current && setProgress(p));
      setDevice({
        ...device,
        connection: 'connected',
        deviceId: pod.id,
        name: pod.name,
        batteryPercent: reading.batteryPercent,
        lastSyncAt: reading.syncedAt,
      });
      if (alive.current) setPhase('done');
    } catch {
      if (alive.current) setPhase('error');
    }
  };

  return (
    <Screen
      footer={
        phase === 'idle' || phase === 'error' ? (
          <Button label={phase === 'error' ? 'Try Again' : 'Find Pod'} onPress={scan} />
        ) : phase === 'found' ? (
          <Button label="Pair" onPress={pair} />
        ) : phase === 'done' ? (
          <Button label="View Pod Details" onPress={() => router.replace('/device')} />
        ) : null
      }
    >
      <View style={{ alignItems: 'center', gap: space.md }}>
        <DemoBadge label="DEMO MODE · SIMULATED PAIRING" />
        <PodGraphic size={200} connected={phase === 'done'} />
        <Txt v="title" style={{ textAlign: 'center' }} accessibilityRole="header">
          {phase === 'done' ? 'Pod connected' : 'Connect your LYRA Pod'}
        </Txt>
        <Txt muted style={{ textAlign: 'center' }}>
          {phase === 'idle' && 'Bring your LYRA Pod close to this phone, then tap Find Pod.'}
          {phase === 'scanning' && 'Looking for nearby pods…'}
          {phase === 'found' && 'A pod is available to pair.'}
          {phase === 'pairing' && 'Pairing…'}
          {phase === 'done' && 'Your pod is connected (Demo).'}
          {phase === 'error' && 'No pod was found. Check that it is nearby and try again.'}
        </Txt>
      </View>

      {phase === 'scanning' ? <ActivityIndicator size="large" color={c.primary} accessibilityLabel="Scanning" /> : null}

      {phase === 'found' || phase === 'pairing' || phase === 'done' ? (
        <Card style={{ gap: space.sm }} accessible accessibilityLabel={`${pod?.name}, ${pod?.id}`}>
          <Txt v="h3">{pod?.name}</Txt>
          <Txt v="small" muted>{pod?.id}</Txt>
          {phase === 'found' ? <Txt v="small" color={c.successText} style={{ fontWeight: '700' }}>Available to pair</Txt> : null}
          {phase === 'pairing' ? (
            <View style={{ gap: 6 }}>
              <View style={{ height: 8, borderRadius: 4, backgroundColor: c.divider, overflow: 'hidden' }}>
                <View style={{ width: `${progress}%`, height: 8, backgroundColor: c.primary }} />
              </View>
              <Txt v="caption" muted accessibilityLiveRegion="polite">Pairing… {progress}%</Txt>
            </View>
          ) : null}
          {phase === 'done' ? (
            <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
              <Check size={16} color={c.successText} />
              <Txt v="small" color={c.successText} style={{ fontWeight: '700' }}>Connected (Demo)</Txt>
            </View>
          ) : null}
        </Card>
      ) : null}

      <Notice>
        This is a simulation. The app does not use Bluetooth yet and no real LYRA Pod is being contacted.
      </Notice>
    </Screen>
  );
}
