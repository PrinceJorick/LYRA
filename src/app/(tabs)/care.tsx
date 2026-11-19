import { router } from 'expo-router';
import { Minus, Phone, Plus, UserPlus } from 'lucide-react-native';
import React from 'react';
import { Pressable, Switch, View } from 'react-native';
import { NotificationStatus } from '../../components/NotificationStatus';
import { Button, Card, Divider, Notice, Screen, SectionHeader, Txt } from '../../components/ui';
import { sortNewestFirst } from '../../services/events';
import { useApp } from '../../state/AppState';
import { space } from '../../theme/colors';
import { useTheme } from '../../theme/useTheme';
import { formatDateTime } from '../../utils/format';

function StepButton({ label, onPress, disabled, children }: { label: string; onPress: () => void; disabled: boolean; children: React.ReactNode }) {
  const c = useTheme();
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={label}
      accessibilityState={{ disabled }}
      disabled={disabled}
      onPress={onPress}
      style={{
        width: 48, height: 48, borderRadius: 24, alignItems: 'center', justifyContent: 'center',
        backgroundColor: c.tint, opacity: disabled ? 0.4 : 1,
      }}
    >
      {children}
    </Pressable>
  );
}

export default function Care() {
  const c = useTheme();
  const { caregivers, prefs, setPrefs, events } = useApp();
  const last = sortNewestFirst(events).find((e) => e.notification);

  return (
    <Screen top title="Care" subtitle="Caregiver contacts and alert preferences.">
      <Notice tone="warning">
        Caregiver notification is experimental. It depends on future integration and connectivity and does not
        guarantee an emergency response. Nothing is sent in this demo.
      </Notice>

      <SectionHeader title="Caregiver contacts" />
      {caregivers.length === 0 ? (
        <Card><Txt muted>No caregiver contacts yet. Add someone to include them in simulated alerts.</Txt></Card>
      ) : (
        caregivers.map((cg) => (
          <Card key={cg.id} style={{ gap: space.md }}>
            <View style={{ gap: 2 }}>
              <Txt v="h3">{cg.name}</Txt>
              <Txt v="small" muted>{cg.relationship || 'No relationship noted'}</Txt>
              <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6, marginTop: 4 }}>
                <Phone size={14} color={c.textMuted} />
                <Txt v="small">{cg.phone}</Txt>
              </View>
            </View>
            <Button kind="ghost" label="Edit contact" onPress={() => router.push({ pathname: '/caregiver', params: { id: cg.id } })} />
          </Card>
        ))
      )}
      <Button
        kind="secondary"
        label="Add Caregiver"
        icon={<UserPlus size={18} color={c.primary} />}
        onPress={() => router.push('/caregiver')}
      />

      <SectionHeader title="Notification preferences" />
      <Card style={{ gap: space.md }}>
        <View style={{ flexDirection: 'row', alignItems: 'center', gap: space.md, minHeight: 48 }}>
          <View style={{ flex: 1 }}>
            <Txt v="h3">Notify when countdown ends</Txt>
            <Txt v="small" muted>Simulated caregiver notification if an alert is not cancelled.</Txt>
          </View>
          <Switch
            accessibilityLabel="Notify caregivers when the countdown ends"
            value={prefs.notifyOnTimeout}
            onValueChange={(v) => setPrefs({ notifyOnTimeout: v })}
            trackColor={{ true: c.primary, false: c.divider }}
          />
        </View>
        <Divider />
        <View style={{ flexDirection: 'row', alignItems: 'center', gap: space.md }}>
          <View style={{ flex: 1 }}>
            <Txt v="h3">Cancel countdown</Txt>
            <Txt v="small" muted>7–10 seconds</Txt>
          </View>
          <StepButton label="Decrease countdown" disabled={prefs.countdownSeconds <= 7} onPress={() => setPrefs({ countdownSeconds: prefs.countdownSeconds - 1 })}>
            <Minus size={20} color={c.primary} />
          </StepButton>
          <Txt v="h2" accessibilityLiveRegion="polite" accessibilityLabel={`${prefs.countdownSeconds} seconds`} style={{ minWidth: 44, textAlign: 'center' }}>
            {prefs.countdownSeconds}s
          </Txt>
          <StepButton label="Increase countdown" disabled={prefs.countdownSeconds >= 10} onPress={() => setPrefs({ countdownSeconds: prefs.countdownSeconds + 1 })}>
            <Plus size={20} color={c.primary} />
          </StepButton>
        </View>
      </Card>

      <SectionHeader title="Last simulated notification" />
      <Card style={{ gap: space.sm }}>
        {last?.notification ? (
          <>
            <Txt v="small" muted>{formatDateTime(last.notification.at, true)} · {last.id}</Txt>
            <NotificationStatus result={last.notification} />
          </>
        ) : (
          <Txt muted>None yet. Run “Simulate Possible Event Alert” on Home and let the countdown finish.</Txt>
        )}
      </Card>
    </Screen>
  );
}
