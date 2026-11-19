import { router } from 'expo-router';
import React, { useState } from 'react';
import { View } from 'react-native';
import { Button, Card, Chip, Field, Notice, Screen, SectionHeader, Txt } from '../components/ui';
import { TRIGGER_OPTIONS } from '../data/mockReports';
import { useApp } from '../state/AppState';
import { space } from '../theme/colors';
import { formatDateTime } from '../utils/format';

export default function ContextLogForm() {
  const { contextLogs, addContextLog } = useApp();
  const [triggers, setTriggers] = useState<string[]>([]);
  const [medicationNote, setMedicationNote] = useState('');
  const [notes, setNotes] = useState('');
  const [error, setError] = useState('');

  const toggle = (t: string) => setTriggers((cur) => (cur.includes(t) ? cur.filter((x) => x !== t) : [...cur, t]));
  const save = () => {
    if (!triggers.length && !medicationNote.trim() && !notes.trim()) {
      setError('Add at least one item before saving.');
      return;
    }
    addContextLog({ triggers, medicationNote: medicationNote.trim(), notes: notes.trim() });
    router.back();
  };

  return (
    <Screen keyboard footer={<Button label="Save Notes" onPress={save} />}>
      <Notice>
        This is a personal record only. LYRA does not give medication advice, dosing instructions or clinical guidance.
      </Notice>

      <View style={{ gap: space.sm }}>
        <Txt v="small" style={{ fontWeight: '600' }}>Possible triggers you noticed today</Txt>
        <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: space.sm }}>
          {TRIGGER_OPTIONS.map((t) => (
            <Chip key={t} role="checkbox" label={t} selected={triggers.includes(t)} onPress={() => toggle(t)} />
          ))}
        </View>
      </View>

      <Field
        label="Medication log (your own entry)"
        placeholder="What you took and when, as you recorded it"
        multiline
        value={medicationNote}
        onChangeText={(t) => { setMedicationNote(t); setError(''); }}
      />
      <Field label="Other observations" multiline value={notes} onChangeText={(t) => { setNotes(t); setError(''); }} error={error} />

      <SectionHeader title="Earlier notes" />
      {contextLogs.map((l) => (
        <Card key={l.id} style={{ gap: 4 }}>
          <Txt v="small" muted>{formatDateTime(l.at, true)}</Txt>
          {l.triggers.length ? <Txt v="small">Triggers: {l.triggers.join(', ')}</Txt> : null}
          {l.medicationNote ? <Txt v="small">Medication: {l.medicationNote}</Txt> : null}
          {l.notes ? <Txt v="small">{l.notes}</Txt> : null}
        </Card>
      ))}
    </Screen>
  );
}
