import { Check, Clock } from 'lucide-react-native';
import React from 'react';
import { View } from 'react-native';
import { useTheme } from '../theme/useTheme';
import type { NotificationResult } from '../types';
import { Txt } from './ui';

function Line({ ok, label }: { ok?: boolean; label: string }) {
  const c = useTheme();
  return (
    <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
      {ok ? <Check size={16} color={c.successText} /> : <Clock size={16} color={c.textMuted} />}
      <Txt v="small" muted={!ok}>{label}</Txt>
    </View>
  );
}

/** Sending, delivery and acknowledgment are separate states (per the proposal). */
export function NotificationStatus({ result }: { result: NotificationResult }) {
  if (result.recipients.length === 0)
    return <Txt muted>No caregiver contacts were saved, so nothing would have been sent.</Txt>;
  return (
    <View style={{ gap: 14 }}>
      {result.recipients.map((r) => (
        <View key={r.caregiverName} style={{ gap: 4 }}>
          <Txt v="small" style={{ fontWeight: '700' }}>{r.caregiverName}</Txt>
          <Line ok label="Message sending: simulated" />
          <Line ok label="Delivery: simulated" />
          <Line label="Caregiver acknowledgment: not simulated" />
        </View>
      ))}
    </View>
  );
}
