import { router, useLocalSearchParams } from 'expo-router';
import { Share2 } from 'lucide-react-native';
import React, { useMemo, useState } from 'react';
import { Share, View } from 'react-native';
import { Button, Card, DemoBadge, Divider, KeyValue, Notice, Screen, Txt } from '../../components/ui';
import { buildClinicianSummary, reportStatusLabel, summaryToText } from '../../services/reports';
import { useApp } from '../../state/AppState';
import { space } from '../../theme/colors';
import { useTheme } from '../../theme/useTheme';
import { formatDateTime } from '../../utils/format';

export default function SummaryPreview() {
  const c = useTheme();
  const { id } = useLocalSearchParams<{ id: string }>();
  const { events, device, setReportStatus } = useApp();
  const event = events.find((e) => e.id === id);
  const [message, setMessage] = useState('');
  const summary = useMemo(() => (event ? buildClinicianSummary(event, device) : null), [event, device]);

  if (!event || !summary)
    return <Screen><Txt muted>This event could not be found.</Txt></Screen>;

  const share = async () => {
    try {
      const r = await Share.share({ title: summary.title, message: summaryToText(summary) });
      if (r.action === Share.sharedAction) {
        setReportStatus(event.id, 'exported');
        setMessage('Summary shared.');
      }
    } catch {
      // e.g. web browsers without the Web Share API
      setMessage('Sharing is not available on this device. You can still review the summary below.');
    }
  };

  return (
    <Screen
      footer={
        <>
          <Button label="Share Summary" icon={<Share2 size={18} color={c.onPrimary} />} onPress={share} />
          {message ? <Txt v="small" muted style={{ textAlign: 'center' }} accessibilityLiveRegion="polite">{message}</Txt> : null}
        </>
      }
    >
      <Txt v="caption" muted>Preview · {reportStatusLabel[event.reportStatus]}</Txt>
      <Card style={{ gap: space.lg }}>
        <View style={{ gap: 6 }}>
          <Txt v="h2" accessibilityRole="header">{summary.title}</Txt>
          <Txt v="small" muted>Generated {formatDateTime(summary.generatedAt, true)}</Txt>
          {event.origin === 'demo_sample' ? <DemoBadge label="DEMO DATA" /> : null}
        </View>
        {summary.sections.map((s) => (
          <View key={s.title} style={{ gap: 4 }}>
            <Divider />
            <Txt v="label" color={c.secondary} style={{ marginTop: space.sm }}>{s.title}</Txt>
            {s.note ? <Txt v="caption" muted>{s.note}</Txt> : null}
            {s.rows.map((r) => <KeyValue key={r.label} label={r.label} value={r.value} />)}
          </View>
        ))}
      </Card>
      <Notice tone="warning">{summary.disclaimer}</Notice>
      {!event.observation ? (
        <Button kind="secondary" label="Add Observations" onPress={() => router.push(`/observation/${event.id}`)} />
      ) : null}
    </Screen>
  );
}
