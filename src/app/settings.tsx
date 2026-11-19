import { router } from 'expo-router';
import React from 'react';
import { View } from 'react-native';
import { Battery, LyraLogo } from '../components/graphics';
import { HardwareList } from '../components/HardwareList';
import { Card, Chip, DemoBadge, Divider, KeyValue, NavRow, SafetyNotice, Screen, SectionHeader, Txt } from '../components/ui';
import { useApp } from '../state/AppState';
import { space } from '../theme/colors';
import type { ThemeMode } from '../types';

const APP_VERSION = '0.1.0 (Demo)';

const MODES: { value: ThemeMode; label: string }[] = [
  { value: 'light', label: 'Light' },
  { value: 'dark', label: 'Dark' },
  { value: 'system', label: 'System' },
];

export default function Settings() {
  const { device, themeMode, setThemeMode } = useApp();
  const connected = device.connection === 'connected';
  return (
    <Screen>
      <Card style={{ gap: space.md }}>
        <LyraLogo size={44} />
        <Txt v="h3">LYRA Companion</Txt>
        <Txt v="small" muted>Version {APP_VERSION}</Txt>
        <DemoBadge />
      </Card>

      <SectionHeader title="Appearance" />
      <View accessibilityRole="radiogroup" style={{ flexDirection: 'row', gap: space.sm, flexWrap: 'wrap' }}>
        {MODES.map((m) => (
          <Chip key={m.value} role="radio" label={m.label} selected={themeMode === m.value} onPress={() => setThemeMode(m.value)} />
        ))}
      </View>

      <SectionHeader title="Demo and pod" />
      <Card style={{ gap: 4 }}>
        <KeyValue label="Demo mode" value="On: all data is sample or simulated" />
        <Divider />
        <KeyValue label="Pod identifier" value={device.deviceId} />
        <Divider />
        <KeyValue label="Connection" value={connected ? 'Connected (simulated)' : 'Disconnected'} />
        <Divider />
        <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 6 }}>
          <Txt v="small" muted>Battery (mock)</Txt>
          {connected ? <Battery percent={device.batteryPercent} /> : <Txt v="small">Unknown</Txt>}
        </View>
      </Card>

      <HardwareList />

      <Card>
        <NavRow title="About LYRA" onPress={() => router.push('/about')} />
      </Card>
      <SafetyNotice long />
    </Screen>
  );
}
