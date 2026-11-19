import { router } from 'expo-router';
import { Check, FileText, Pencil } from 'lucide-react-native';
import React, { useMemo } from 'react';
import { View } from 'react-native';
import { Badge, Button, Card, Notice, Screen, Txt } from '../../components/ui';
import { EVENT_LABEL, sortNewestFirst } from '../../services/events';
import { reportStatusLabel } from '../../services/reports';
import { useApp } from '../../state/AppState';
import { space } from '../../theme/colors';
import { useTheme } from '../../theme/useTheme';
import { formatDateTime } from '../../utils/format';

export default function Reports() {
  const c = useTheme();
  const { events } = useApp();
  const list = useMemo(() => sortNewestFirst(events), [events]);

  return (
    <Screen top title="Reports" subtitle="Document an event and prepare a summary for a clinician.">
      <Notice>
        Summaries keep device information separate from what you or a witness report. They are not a diagnosis.
      </Notice>
      {list.map((e) => {
        const hasObs = !!e.observation;
        return (
          <Card key={e.id} style={{ gap: space.md }}>
            <View style={{ gap: 2 }}>
              <Txt v="h3">{EVENT_LABEL}</Txt>
              <Txt v="small" muted>{formatDateTime(e.timestamp, true)}</Txt>
              <Txt v="caption" muted>{e.id}</Txt>
            </View>
            <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 6 }}>
              <Badge
                tone={hasObs ? 'success' : 'warning'}
                label={hasObs ? 'Observation recorded' : 'Observation pending'}
                icon={hasObs ? <Check size={13} color={c.successText} /> : undefined}
              />
              <Badge label={reportStatusLabel[e.reportStatus]} tone={e.reportStatus === 'not_started' ? 'neutral' : 'purple'} />
              {e.origin === 'demo_sample' ? <Badge label="Demo data" tone="neutral" /> : null}
            </View>
            <View style={{ gap: space.sm }}>
              <Button
                kind="secondary"
                label={hasObs ? 'Edit Observations' : 'Add Observations'}
                icon={<Pencil size={18} color={c.primary} />}
                onPress={() => router.push(`/observation/${e.id}`)}
              />
              <Button
                label="Export Clinician Summary"
                icon={<FileText size={18} color={c.onPrimary} />}
                onPress={() => router.push(`/summary/${e.id}`)}
              />
            </View>
          </Card>
        );
      })}
    </Screen>
  );
}
