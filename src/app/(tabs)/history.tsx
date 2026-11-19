import React, { useMemo, useState } from 'react';
import { Pressable, View } from 'react-native';
import { EventCard } from '../../components/EventCard';
import { Card, Chip, Divider, Notice, Screen, Txt } from '../../components/ui';
import { filterByReview, sortNewestFirst } from '../../services/events';
import { useApp } from '../../state/AppState';
import { space } from '../../theme/colors';
import { useTheme } from '../../theme/useTheme';
import type { ReviewStatus } from '../../types';

type Filter = 'all' | ReviewStatus;
const FILTERS: { value: Filter; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'needs_review', label: 'Needs Review' },
  { value: 'reviewed', label: 'Reviewed' },
];

const EMPTY: Record<Filter, string> = {
  all: 'No events recorded yet.',
  needs_review: 'Nothing needs review.',
  reviewed: 'No reviewed events yet.',
};

export default function History() {
  const c = useTheme();
  const { events, setReview } = useApp();
  const [filter, setFilter] = useState<Filter>('all');
  const list = useMemo(() => sortNewestFirst(filterByReview(events, filter)), [events, filter]);

  return (
    <Screen top title="History" subtitle="Review previously recorded events.">
      <View accessibilityRole="radiogroup" style={{ flexDirection: 'row', flexWrap: 'wrap', gap: space.sm }}>
        {FILTERS.map((f) => (
          <Chip key={f.value} role="radio" label={f.label} selected={filter === f.value} onPress={() => setFilter(f.value)} />
        ))}
      </View>
      <Notice>Records tagged “Demo data” are bundled samples, not real detections.</Notice>

      {list.length === 0 ? (
        <Card style={{ alignItems: 'center', padding: space.xxl }}>
          <Txt muted accessibilityLiveRegion="polite">{EMPTY[filter]}</Txt>
        </Card>
      ) : (
        list.map((e) => (
          <Card key={e.id} style={{ gap: space.sm }}>
            <EventCard event={e} flat />
            <Divider />
            <Pressable
              accessibilityRole="button"
              onPress={() => setReview(e.id, e.review === 'reviewed' ? 'needs_review' : 'reviewed')}
              style={{ minHeight: 44, justifyContent: 'center' }}
            >
              <Txt v="small" color={c.primary} style={{ fontWeight: '700' }}>
                {e.review === 'reviewed' ? 'Mark as Needs Review' : 'Mark as Reviewed'}
              </Txt>
            </Pressable>
          </Card>
        ))
      )}
    </Screen>
  );
}
