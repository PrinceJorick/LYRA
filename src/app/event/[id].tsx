import { router, useLocalSearchParams } from 'expo-router';
import React from 'react';
import { View } from 'react-native';
import { NotificationStatus } from '../../components/NotificationStatus';
import { OriginBadge, ReviewBadge } from '../../components/EventCard';
import { Waveform } from '../../components/graphics';
import { Button, Card, Chip, KeyValue, Notice, Screen, SectionHeader, Txt } from '../../components/ui';
import { RESPONSIVENESS_OPTIONS } from '../../data/mockReports';
import { emgSample, ILLUSTRATIVE_LABEL, imuSample } from '../../data/mockSignals';
import { EVENT_LABEL, outcomeLabel } from '../../services/events';
import { reportStatusLabel } from '../../services/reports';
import { useApp } from '../../state/AppState';
import { space } from '../../theme/colors';
import { useTheme } from '../../theme/useTheme';
import { formatDateTime } from '../../utils/format';

export default function EventDetails() {
  const c = useTheme();
  const { id } = useLocalSearchParams<{ id: string }>();
  const { events, setReview } = useApp();
  const event = events.find((e) => e.id === id);
  const seed = event?.signalSeed ?? 1;
  const signals = [
    { name: 'sEMG', values: emgSample(seed), color: c.secondary },
    { name: 'Accelerometer', values: imuSample(seed, 0), color: c.primary },
    { name: 'Gyroscope', values: imuSample(seed, 1.4), color: c.primary },
  ];

  if (!event)
    return (
      <Screen>
        <Txt muted>This event could not be found.</Txt>
        <Button label="Back to History" onPress={() => router.replace('/history')} />
      </Screen>
    );

  const o = event.observation;
  return (
    <Screen>
      <View style={{ gap: space.sm }}>
        <Txt v="title" accessibilityRole="header">{EVENT_LABEL}</Txt>
        <Txt>{formatDateTime(event.timestamp, true)}</Txt>
        <Txt v="small" muted>ID: {event.id}</Txt>
        <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 6 }}>
          <ReviewBadge event={event} />
          <OriginBadge event={event} />
        </View>
        <Txt v="small" muted>{outcomeLabel[event.outcome]}</Txt>
      </View>

      <SectionHeader title="Signal Overview" />
      {signals.map((s) => (
        <Card key={s.name} style={{ gap: space.sm, padding: space.md }}>
          <Txt v="small" style={{ fontWeight: '700' }}>{s.name}</Txt>
          <Waveform values={s.values} color={s.color} />
        </Card>
      ))}
      <Txt v="caption" muted>{ILLUSTRATIVE_LABEL}. No real sensor data is attached to this record.</Txt>
      <Notice>This record is illustrative and does not represent a clinically confirmed seizure event.</Notice>

      {event.notification ? (
        <>
          <SectionHeader title="Caregiver notification" />
          <Card style={{ gap: space.sm }}>
            <Txt v="small" muted>Simulated. No message was sent.</Txt>
            <NotificationStatus result={event.notification} />
          </Card>
        </>
      ) : null}

      <SectionHeader title="Mark as" />
      <View accessibilityRole="radiogroup" style={{ flexDirection: 'row', gap: space.sm, flexWrap: 'wrap' }}>
        <Chip role="radio" label="Needs Review" selected={event.review === 'needs_review'} onPress={() => setReview(event.id, 'needs_review')} />
        <Chip role="radio" label="Reviewed" selected={event.review === 'reviewed'} onPress={() => setReview(event.id, 'reviewed')} />
      </View>

      <SectionHeader title="Observations" />
      <Card style={{ gap: space.sm }}>
        {o ? (
          <>
            <Txt v="caption" muted>{o.reporter === 'witness' ? 'Witness-reported' : 'Self-reported'}</Txt>
            <KeyValue label="Duration" value={o.durationApprox || '—'} />
            <KeyValue label="Movements" value={o.movements.join(', ') || '—'} />
            <KeyValue label="Responsiveness" value={RESPONSIVENESS_OPTIONS.find((r) => r.value === o.responsiveness)?.label ?? '—'} />
          </>
        ) : (
          <Txt muted>No observations recorded yet.</Txt>
        )}
        <Txt v="caption" muted>{reportStatusLabel[event.reportStatus]}</Txt>
      </Card>
      <View style={{ gap: space.sm }}>
        <Button kind="secondary" label={o ? 'Edit Observations' : 'Add Observations'} onPress={() => router.push(`/observation/${event.id}`)} />
        <Button label="Export Clinician Summary" onPress={() => router.push(`/summary/${event.id}`)} />
      </View>
    </Screen>
  );
}
