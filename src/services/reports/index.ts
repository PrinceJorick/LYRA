import { RESPONSIVENESS_OPTIONS } from '../../data/mockReports';
import type { DeviceStatus, PossibleEvent } from '../../types';
import { formatDateTime } from '../../utils/format';
import { EVENT_LABEL, outcomeLabel, reviewLabel } from '../events';

export interface SummarySection {
  title: string;
  note?: string;
  rows: { label: string; value: string }[];
}

export interface ClinicianSummary {
  title: string;
  generatedAt: number;
  sections: SummarySection[];
  disclaimer: string;
}

export const SUMMARY_DISCLAIMER =
  'This record is not a clinical diagnosis. It was produced by an experimental research prototype and contains user- or witness-reported information that has not been clinically verified.';

export const reportStatusLabel = {
  not_started: 'Report not started',
  ready: 'Summary ready',
  exported: 'Summary exported',
} as const;

const NONE = 'Not recorded';

export function buildClinicianSummary(e: PossibleEvent, device: DeviceStatus): ClinicianSummary {
  const o = e.observation;
  return {
    title: 'LYRA Event Summary',
    generatedAt: Date.now(),
    disclaimer: SUMMARY_DISCLAIMER,
    sections: [
      {
        title: 'Event',
        rows: [
          { label: 'Label', value: EVENT_LABEL },
          { label: 'Date and time', value: formatDateTime(e.timestamp, true) },
          { label: 'Event ID', value: e.id },
          { label: 'Review status', value: reviewLabel[e.review] },
        ],
      },
      {
        title: 'Device event information',
        note: 'Automatically collected by the app. No sensor measurements are attached to this demo summary.',
        rows: [
          { label: 'Pod', value: device.deviceId },
          { label: 'Record type', value: e.origin === 'demo_sample' ? 'Demo sample record' : 'Simulated alert' },
          { label: 'Detection source', value: e.detectionSource === 'mock' ? 'Mock (ML model not integrated)' : 'ML model' },
          { label: 'Alert outcome', value: outcomeLabel[e.outcome] },
        ],
      },
      {
        title: o ? `Observations (${o.reporter === 'witness' ? 'witness' : 'self'}-reported)` : 'Observations',
        note: 'Reported by the person or a witness; not measured by LYRA.',
        rows: o
          ? [
              { label: 'Approximate duration', value: o.durationApprox || NONE },
              { label: 'Visible movements', value: o.movements.join(', ') || NONE },
              {
                label: 'Responsiveness',
                value: RESPONSIVENESS_OPTIONS.find((r) => r.value === o.responsiveness)?.label ?? NONE,
              },
              { label: 'Recovery', value: o.recovery || NONE },
              { label: 'Additional notes', value: o.notes || NONE },
            ]
          : [{ label: 'Status', value: 'No observations recorded yet' }],
      },
    ],
  };
}

export function summaryToText(s: ClinicianSummary): string {
  const body = s.sections
    .map((sec) => [sec.title.toUpperCase(), ...sec.rows.map((r) => `${r.label}: ${r.value}`)].join('\n'))
    .join('\n\n');
  return `${s.title}\nGenerated ${formatDateTime(s.generatedAt, true)}\n\n${body}\n\n${s.disclaimer}`;
}
