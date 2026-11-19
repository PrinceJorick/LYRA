import { router } from 'expo-router';
import { Bell, Check, CircleSlash, ClipboardList, Settings } from 'lucide-react-native';
import React, { useMemo } from 'react';
import { Pressable, View } from 'react-native';
import { EventCard } from '../../components/EventCard';
import { Battery, LyraLogo, PodGraphic, Waveform } from '../../components/graphics';
import {
  Button, Card, DemoBadge, NavRow, SafetyNotice, Screen, SectionHeader, Txt,
} from '../../components/ui';
import { emgSample, ILLUSTRATIVE_LABEL, imuSample } from '../../data/mockSignals';
import { sortNewestFirst } from '../../services/events';
import { useApp } from '../../state/AppState';
import { space } from '../../theme/colors';
import { useTheme } from '../../theme/useTheme';
import { formatDate, formatRelative, greeting } from '../../utils/format';

export default function Home() {
  const c = useTheme();
  const { device, events, contextLogs, prefs } = useApp();
  const connected = device.connection === 'connected';
  const recent = useMemo(() => sortNewestFirst(events).slice(0, 2), [events]);
  const emg = useMemo(() => emgSample(5), []);
  const imu = useMemo(() => imuSample(3), []);
  const lastLog = contextLogs[0];

  return (
    <Screen top>
      {/* Header */}
      <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }}>
        <LyraLogo />
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Settings"
          onPress={() => router.push('/settings')}
          style={{ width: 48, height: 48, alignItems: 'center', justifyContent: 'center' }}
        >
          <Settings size={26} color={c.primary} />
        </Pressable>
      </View>

      <View>
        <Txt v="title" accessibilityRole="header">{greeting()}!</Txt>
        <Txt muted>{connected ? 'Your LYRA Pod is connected.' : 'Your LYRA Pod is not connected.'}</Txt>
      </View>

      {/* Pod status */}
      <Card style={{ gap: space.lg }}>
        <View style={{ flexDirection: 'row', alignItems: 'center', gap: space.md }}>
          <View style={{ flex: 1, gap: 8 }}>
            <DemoBadge />
            <Txt v="h2">{device.name}</Txt>
            <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
              {connected ? <Check size={16} color={c.successText} /> : <CircleSlash size={16} color={c.textMuted} />}
              <Txt v="small" color={connected ? c.successText : c.textMuted} style={{ fontWeight: '700' }}>
                {connected ? 'Connected (Demo)' : 'Not connected'}
              </Txt>
            </View>
            {connected ? (
              <>
                <Battery percent={device.batteryPercent} />
                <Txt v="small" muted>Last sync: {formatRelative(device.lastSyncAt)}</Txt>
              </>
            ) : (
              <Txt v="small" muted>Pair your pod to see its status.</Txt>
            )}
          </View>
          <PodGraphic size={128} connected={connected} />
        </View>
        {connected ? (
          <Button kind="ghost" label="View Pod Details →" onPress={() => router.push('/device')} />
        ) : (
          <Button label="Find Pod" onPress={() => router.push('/pairing')} />
        )}
      </Card>

      {/* Signal preview */}
      <View style={{ gap: space.sm }}>
        <SectionHeader title="Signal Preview" />
        <Txt v="caption" muted>{ILLUSTRATIVE_LABEL} · not a real reading</Txt>
      </View>
      <View style={{ flexDirection: 'row', gap: space.md }}>
        <Card style={{ flex: 1, gap: space.sm, padding: space.md }}>
          <Txt v="small" style={{ fontWeight: '700' }}>Muscle Activity</Txt>
          <Txt v="caption" muted>sEMG</Txt>
          <Waveform values={emg} color={c.secondary} />
        </Card>
        <Card style={{ flex: 1, gap: space.sm, padding: space.md }}>
          <Txt v="small" style={{ fontWeight: '700' }}>Motion Activity</Txt>
          <Txt v="caption" muted>IMU</Txt>
          <Waveform values={imu} color={c.primary} />
        </Card>
      </View>

      {/* BEFORE */}
      <SectionHeader stage="BEFORE" title="Prepare" />
      <Card>
        <NavRow
          title="Add Today's Notes"
          subtitle={lastLog ? `Last note: ${formatDate(lastLog.at)}` : 'Record triggers, medication and observations'}
          left={<ClipboardList size={26} color={c.primary} />}
          onPress={() => router.push('/context-log')}
        />
      </Card>

      {/* DURING */}
      <SectionHeader stage="DURING" title="Possible-event alert" />
      <Card style={{ gap: space.md }}>
        <Txt v="small" muted>
          Try the alert flow: a {prefs.countdownSeconds}-second countdown you can cancel. Everything is simulated;
          no message is sent.
        </Txt>
        <Button
          label="Simulate Possible Event Alert"
          icon={<Bell size={20} color={c.onPrimary} />}
          onPress={() => router.push('/alert')}
        />
      </Card>

      {/* AFTER */}
      <SectionHeader stage="AFTER" title="Recent Events" action={{ label: 'See all', onPress: () => router.push('/history') }} />
      {recent.map((e) => <EventCard key={e.id} event={e} />)}

      <SafetyNotice />
    </Screen>
  );
}
