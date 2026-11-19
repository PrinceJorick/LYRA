import { router, useLocalSearchParams } from 'expo-router';
import React, { useState } from 'react';
import { View } from 'react-native';
import { Button, Chip, Field, Notice, Screen, Txt } from '../../components/ui';
import { MOVEMENT_OPTIONS, RESPONSIVENESS_OPTIONS } from '../../data/mockReports';
import { useApp } from '../../state/AppState';
import { space } from '../../theme/colors';
import type { Reporter, Responsiveness } from '../../types';
import { formatDateTime } from '../../utils/format';

export default function ObservationForm() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const { events, saveObservation } = useApp();
  const event = events.find((e) => e.id === id);
  const prev = event?.observation;

  const [reporter, setReporter] = useState<Reporter>(prev?.reporter ?? 'witness');
  const [duration, setDuration] = useState(prev?.durationApprox ?? '');
  const [movements, setMovements] = useState<string[]>(prev?.movements ?? []);
  const [resp, setResp] = useState<Responsiveness>(prev?.responsiveness ?? 'unsure');
  const [recovery, setRecovery] = useState(prev?.recovery ?? '');
  const [notes, setNotes] = useState(prev?.notes ?? '');

  if (!event)
    return (
      <Screen><Txt muted>This event could not be found.</Txt></Screen>
    );

  const toggle = (m: string) => setMovements((cur) => (cur.includes(m) ? cur.filter((x) => x !== m) : [...cur, m]));
  const save = () => {
    saveObservation(event.id, {
      reporter, durationApprox: duration.trim(), movements, responsiveness: resp,
      recovery: recovery.trim(), notes: notes.trim(), recordedAt: Date.now(),
    });
    router.back();
  };

  return (
    <Screen keyboard footer={<Button label="Save Observations" onPress={save} />}>
      <Txt muted>{formatDateTime(event.timestamp, true)} · {event.id}</Txt>
      <Notice>What you enter here is user- or witness-reported. LYRA does not verify it or diagnose from it.</Notice>

      <View style={{ gap: space.sm }}>
        <Txt v="small" style={{ fontWeight: '600' }}>Reported by</Txt>
        <View accessibilityRole="radiogroup" style={{ flexDirection: 'row', gap: space.sm, flexWrap: 'wrap' }}>
          <Chip role="radio" label="Witness" selected={reporter === 'witness'} onPress={() => setReporter('witness')} />
          <Chip role="radio" label="Myself" selected={reporter === 'self'} onPress={() => setReporter('self')} />
        </View>
      </View>

      <Field label="Approximate duration" placeholder="e.g. about 1 minute" value={duration} onChangeText={setDuration} />

      <View style={{ gap: space.sm }}>
        <Txt v="small" style={{ fontWeight: '600' }}>Visible movements</Txt>
        <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: space.sm }}>
          {MOVEMENT_OPTIONS.map((m) => (
            <Chip key={m} role="checkbox" label={m} selected={movements.includes(m)} onPress={() => toggle(m)} />
          ))}
        </View>
      </View>

      <View style={{ gap: space.sm }}>
        <Txt v="small" style={{ fontWeight: '600' }}>Responsiveness</Txt>
        <View accessibilityRole="radiogroup" style={{ flexDirection: 'row', flexWrap: 'wrap', gap: space.sm }}>
          {RESPONSIVENESS_OPTIONS.map((r) => (
            <Chip key={r.value} role="radio" label={r.label} selected={resp === r.value} onPress={() => setResp(r.value)} />
          ))}
        </View>
      </View>

      <Field label="Recovery experience" placeholder="e.g. how long it took to feel back to normal" multiline value={recovery} onChangeText={setRecovery} />
      <Field label="Additional observations" multiline value={notes} onChangeText={setNotes} />
    </Screen>
  );
}
