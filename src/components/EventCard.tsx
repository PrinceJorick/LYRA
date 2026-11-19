import { Check, ChevronRight, CircleAlert, X } from 'lucide-react-native';
import { router } from 'expo-router';
import React from 'react';
import { Pressable, View } from 'react-native';
import { EVENT_LABEL, reviewLabel } from '../services/events';
import { space } from '../theme/colors';
import { useTheme } from '../theme/useTheme';
import type { PossibleEvent } from '../types';
import { formatDateTime } from '../utils/format';
import { Badge, Card, Txt } from './ui';

export function ReviewBadge({ event }: { event: PossibleEvent }) {
  const c = useTheme();
  return event.review === 'reviewed' ? (
    <Badge label={reviewLabel.reviewed} tone="success" icon={<Check size={13} color={c.successText} />} />
  ) : (
    <Badge label={reviewLabel.needs_review} tone="warning" icon={<CircleAlert size={13} color={c.warningText} />} />
  );
}

/** Small tag that makes demo/simulated provenance explicit. */
export function OriginBadge({ event }: { event: PossibleEvent }) {
  const c = useTheme();
  if (event.outcome === 'cancelled')
    return <Badge label="Cancelled by user" tone="neutral" icon={<X size={13} color={c.textMuted} />} />;
  return <Badge label={event.origin === 'demo_sample' ? 'Demo data' : 'Simulated'} tone="neutral" />;
}

export function EventCard({ event, flat }: { event: PossibleEvent; flat?: boolean }) {
  const c = useTheme();
  const body = (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={`${EVENT_LABEL}, ${formatDateTime(event.timestamp, true)}, ${reviewLabel[event.review]}`}
      onPress={() => router.push(`/event/${event.id}`)}
      style={{ flexDirection: 'row', alignItems: 'center', gap: space.md, minHeight: 56 }}
    >
      <View
        style={{
          width: 44, height: 44, borderRadius: 22, backgroundColor: c.primary,
          alignItems: 'center', justifyContent: 'center',
        }}
      >
        <Txt v="h3" color={c.onPrimary}>!</Txt>
      </View>
      <View style={{ flex: 1, gap: 4 }}>
        <Txt v="h3">{EVENT_LABEL}</Txt>
        <Txt v="small" muted>{formatDateTime(event.timestamp, true)}</Txt>
        <Txt v="caption" muted>{event.id}</Txt>
        <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginTop: 2 }}>
          <ReviewBadge event={event} />
          <OriginBadge event={event} />
        </View>
      </View>
      <ChevronRight size={20} color={c.textMuted} />
    </Pressable>
  );
  return flat ? body : <Card>{body}</Card>;
}
