import { router } from 'expo-router';
import { Check, TriangleAlert, X } from 'lucide-react-native';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { BackHandler, View } from 'react-native';
import { CountdownRing } from '../components/graphics';
import { NotificationStatus } from '../components/NotificationStatus';
import { Button, Card, Notice, Screen, Txt } from '../components/ui';
import { encodePhoneCommand, podLink } from '../services/pod';
import { useApp } from '../state/AppState';
import { space } from '../theme/colors';
import { useTheme } from '../theme/useTheme';
import type { NotificationResult, PossibleEvent } from '../types';

type Phase =
  | { name: 'countdown' }
  | { name: 'cancelled'; event: PossibleEvent; onPod: boolean }
  | { name: 'notified'; event: PossibleEvent; notification?: NotificationResult };

export default function AlertScreen() {
  const c = useTheme();
  const { prefs, podAlert, recordAlert, completeAlert } = useApp();
  // Fixed at mount so changing preferences mid-alert can't alter a running countdown.
  const [total] = useState(prefs.countdownSeconds);
  const [alert] = useState(podAlert); // the pod-reported possible event that opened this screen
  const timestamp = alert?.receivedAt ?? 0;
  const origin = alert?.origin ?? 'simulated';
  const [msLeft, setMsLeft] = useState(total * 1000);
  const [phase, setPhase] = useState<Phase>({ name: 'countdown' });
  const deadline = useRef<number | null>(null); // set when the timer first starts
  const settled = useRef(false);
  // Latest callbacks, so the timer effect never needs to restart.
  const api = useRef({ recordAlert, completeAlert });
  useEffect(() => {
    api.current = { recordAlert, completeAlert };
  });

  const finish = useCallback(async () => {
    if (settled.current) return;
    settled.current = true;
    const event = await api.current.completeAlert(timestamp, origin);
    setPhase({ name: 'notified', event, notification: event.notification });
  }, [timestamp, origin]);

  useEffect(() => {
    if (!alert) router.back(); // opened without a pod event (e.g. a stale link): nothing to show
  }, [alert]);

  // The wearer pressed the pod's cancel button during the countdown.
  useEffect(() => {
    if (phase.name !== 'countdown' || settled.current || !podAlert?.cancelledOnPod || podAlert.eventId !== alert?.eventId) return;
    settled.current = true;
    setPhase({ name: 'cancelled', onPod: true, event: api.current.recordAlert('cancelled', timestamp, origin) });
  }, [podAlert, alert, phase.name, timestamp, origin]);

  useEffect(() => {
    if (phase.name !== 'countdown') return;
    deadline.current ??= Date.now() + total * 1000;
    const t = setInterval(() => {
      const left = deadline.current! - Date.now();
      if (left <= 0) {
        setMsLeft(0);
        clearInterval(t);
        void finish();
      } else setMsLeft(left);
    }, 100);
    return () => clearInterval(t);
  }, [phase.name, finish, total]);

  // Android back must not silently dismiss a live alert; use the Cancel Alert button.
  useEffect(() => {
    if (phase.name !== 'countdown') return;
    const sub = BackHandler.addEventListener('hardwareBackPress', () => true);
    return () => sub.remove();
  }, [phase.name]);

  const cancel = () => {
    if (settled.current) return;
    settled.current = true;
    setPhase({ name: 'cancelled', onPod: false, event: api.current.recordAlert('cancelled', timestamp, origin) });
    if (alert) // tell the pod, so its confirmation resets too (DRAFT phone_command)
      podLink.sendCommand(encodePhoneCommand({ origin, command: 'cancel_event', eventId: alert.eventId }))
        .catch((e: unknown) => console.warn('Pod did not accept the cancel command:', e));
  };

  const seconds = Math.ceil(msLeft / 1000);
  if (!alert) return null;

  if (phase.name === 'countdown') {
    return (
      <Screen
        top
        scroll={false}
        contentStyle={{ alignItems: 'center', justifyContent: 'center' }}
        footer={
          <Button
            label="Cancel Alert"
            icon={<X size={20} color={c.onPrimary} />}
            onPress={cancel}
            accessibilityHint="Stops the countdown. No caregiver notification is simulated."
            style={{ backgroundColor: c.primaryDeep }}
          />
        }
      >
        <View style={{ alignItems: 'center', gap: space.sm }} accessibilityRole="alert">
          <Txt v="title" style={{ textAlign: 'center' }}>Possible convulsive seizure detected</Txt>
          <Txt muted style={{ textAlign: 'center' }}>This is not a diagnosis. Are you okay? Cancel if this was a mistake.</Txt>
        </View>
        <CountdownRing progress={msLeft / (total * 1000)}>
          <View style={{ alignItems: 'center' }} accessible accessibilityLiveRegion="polite" accessibilityLabel={`${seconds} seconds to cancel`}>
            <Txt style={{ fontSize: 76, fontWeight: '800', lineHeight: 86 }}>{seconds}</Txt>
            <Txt v="small" muted>seconds to cancel</Txt>
          </View>
        </CountdownRing>
        <Notice tone="warning">
          {origin === 'simulated' ? 'Simulated pod event (no Bluetooth). ' : ''}
          {prefs.notifyOnTimeout
            ? 'A caregiver notification will be simulated if you do not cancel in time.'
            : 'Caregiver notification is turned off in Care preferences.'}{' '}
          Demo: nothing is actually sent.
        </Notice>
      </Screen>
    );
  }

  if (phase.name === 'cancelled') {
    return (
      <Screen
        top
        footer={
          <>
            <Button label="Back to Home" onPress={() => router.dismissAll()} />
            <Button kind="secondary" label="View Event Details" onPress={() => router.replace(`/event/${phase.event.id}`)} />
          </>
        }
      >
        <View style={{ alignItems: 'center', gap: space.md, paddingTop: space.xxl }}>
          <View style={{ width: 72, height: 72, borderRadius: 36, backgroundColor: c.tint, alignItems: 'center', justifyContent: 'center' }}>
            <Check size={36} color={c.primary} />
          </View>
          <Txt v="title" style={{ textAlign: 'center' }} accessibilityRole="header">
            {phase.onPod ? 'Alert cancelled on the pod' : 'Alert cancelled'}
          </Txt>
          <Txt muted style={{ textAlign: 'center' }}>
            No caregiver notification was simulated. The event was saved to History as “Cancelled by user”.
          </Txt>
          <Txt v="caption" muted>{phase.event.id}</Txt>
        </View>
      </Screen>
    );
  }

  return (
    <Screen
      top
      footer={
        <>
          <Button label="View Event Details" onPress={() => router.replace(`/event/${phase.event.id}`)} />
          <Button kind="secondary" label="Back to Home" onPress={() => router.dismissAll()} />
        </>
      }
    >
      <View style={{ alignItems: 'center', gap: space.md, paddingTop: space.xl }}>
        <View style={{ width: 72, height: 72, borderRadius: 36, backgroundColor: c.warningBg, alignItems: 'center', justifyContent: 'center' }}>
          <TriangleAlert size={34} color={c.warningText} />
        </View>
        <Txt v="title" style={{ textAlign: 'center' }} accessibilityRole="header">
          {phase.notification ? 'Caregiver notification simulated' : 'Countdown finished'}
        </Txt>
        <Txt muted style={{ textAlign: 'center' }}>
          {phase.notification
            ? 'Caregivers would be notified at this point. In this demo, no message was sent.'
            : 'Caregiver notification is turned off, so none was simulated.'}
        </Txt>
      </View>
      {phase.notification ? (
        <Card style={{ gap: space.sm }}>
          <Txt v="h3">Notification status (simulated)</Txt>
          <NotificationStatus result={phase.notification} />
        </Card>
      ) : null}
      <Txt v="caption" muted style={{ textAlign: 'center' }}>
        Saved to History as {phase.event.id} · Needs Review
      </Txt>
    </Screen>
  );
}
