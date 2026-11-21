import React, { useMemo } from 'react';
import { View } from 'react-native';
import { Badge, Card, KeyValue, Notice, Screen, SectionHeader, Txt } from '../../components/ui';
import { eventStats } from '../../services/events';
import { mlService } from '../../services/ml';
import { useApp } from '../../state/AppState';
import { space } from '../../theme/colors';
import { useTheme } from '../../theme/useTheme';
import { formatDate } from '../../utils/format';

const tally = (items: string[]) =>
  Object.entries(items.reduce<Record<string, number>>((a, k) => ((a[k] = (a[k] ?? 0) + 1), a), {})).sort(
    (a, b) => b[1] - a[1],
  );

function Stat({ value, label }: { value: number; label: string }) {
  return (
    <View style={{ flex: 1 }} accessible accessibilityLabel={`${label}: ${value}`}>
      <Txt style={{ fontSize: 30, fontWeight: '800' }}>{value}</Txt>
      <Txt v="caption" muted>{label}</Txt>
    </View>
  );
}

export default function Insights() {
  const c = useTheme();
  const { events, contextLogs } = useApp();
  const st = useMemo(() => eventStats(events), [events]);
  const meta = mlService.getMetadata();
  const outputs = mlService.getInsights();
  const withObs = events.filter((e) => e.observation);
  const movements = tally(withObs.flatMap((e) => e.observation!.movements));
  const triggers = tally(contextLogs.flatMap((l) => l.triggers));
  const max = Math.max(1, ...st.perDay.map((d) => d.count));
  const first = st.perDay[0].ts;

  return (
    <Screen top title="Insights" subtitle="Summaries of recorded events and entries.">
      <Notice tone="warning">ML model not integrated. Insights currently use illustrative sample data.</Notice>

      {/* 1. Historical statistics */}
      <SectionHeader title="Event statistics" />
      <Card style={{ gap: space.lg }}>
        <Badge label="Sample + simulated records" tone="neutral" />
        <View style={{ flexDirection: 'row', gap: space.md }}>
          <Stat value={st.total} label="Total events" />
          <Stat value={st.needsReview} label="Needs Review" />
          <Stat value={st.reviewed} label="Reviewed" />
        </View>
        <View style={{ flexDirection: 'row', gap: space.md }}>
          <Stat value={st.last7} label="Last 7 days" />
          <Stat value={st.last14} label="Last 14 days" />
          <Stat value={st.cancelled} label="Cancelled" />
        </View>
      </Card>

      <Card style={{ gap: space.md }}>
        <Txt v="h3">Activity timeline</Txt>
        <Txt v="caption" muted>
          Events per day, 14 days ending {formatDate(st.anchor)} (most recent record).
        </Txt>
        <View
          accessible
          accessibilityLabel={`Bar chart of events per day. ${st.perDay.filter((d) => d.count).map((d) => `${formatDate(d.ts)}: ${d.count}`).join('; ') || 'No events'}`}
          style={{ flexDirection: 'row', alignItems: 'flex-end', gap: 4, height: 90 }}
        >
          {st.perDay.map((d) => (
            <View
              key={d.key}
              style={{
                flex: 1, borderRadius: 4,
                height: d.count ? 12 + (d.count / max) * 78 : 4,
                backgroundColor: d.count ? c.primary : c.divider,
              }}
            />
          ))}
        </View>
        <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
          <Txt v="caption" muted>{formatDate(first)}</Txt>
          <Txt v="caption" muted>{formatDate(st.anchor)}</Txt>
        </View>
      </Card>

      {/* 2. Model outputs */}
      <SectionHeader title="Model output" />
      <Card style={{ gap: space.sm }}>
        <KeyValue label="Model" value={meta.modelId === 'none' ? 'None' : meta.modelId} />
        <KeyValue label="Status" value={meta.status === 'mock' ? 'Mock: ML model not integrated' : 'Integrated'} />
        {outputs.length === 0 ? (
          <Txt v="small" muted>
            No model outputs available. LYRA shows no risk scores, predictions or confidence values until a
            validated model is integrated.
          </Txt>
        ) : (
          outputs.map((o) => <KeyValue key={o.id} label={o.label} value={`${o.value} (v${o.modelVersion})`} />)
        )}
      </Card>

      {/* 3. User-entered */}
      <SectionHeader title="Your entries" />
      <Card style={{ gap: space.md }}>
        <Badge label="User- or witness-reported" tone="neutral" />
        <KeyValue label="Events with observations" value={`${withObs.length} of ${events.length}`} />
        <KeyValue label="Notes logged (Before)" value={String(contextLogs.length)} />
        <View style={{ gap: 4 }}>
          <Txt v="small" style={{ fontWeight: '700' }}>Reported movements</Txt>
          <Txt v="small" muted>{movements.map(([k, n]) => `${k} (${n})`).join(', ') || 'None recorded'}</Txt>
        </View>
        <View style={{ gap: 4 }}>
          <Txt v="small" style={{ fontWeight: '700' }}>Possible triggers noted</Txt>
          <Txt v="small" muted>{triggers.map(([k, n]) => `${k} (${n})`).join(', ') || 'None recorded'}</Txt>
        </View>
      </Card>
    </Screen>
  );
}
