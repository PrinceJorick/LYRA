import type { EventOutcome, PossibleEvent, ReviewStatus } from '../../types';
import { dayKey } from '../../utils/format';

export const EVENT_LABEL = 'Possible Event'; // never "Seizure Detected/Confirmed"

export const outcomeLabel: Record<EventOutcome, string> = {
  sample_record: 'Demo sample',
  cancelled: 'Cancelled by user',
  notification_simulated: 'Caregiver notification simulated',
};

export const reviewLabel: Record<ReviewStatus, string> = {
  needs_review: 'Needs Review',
  reviewed: 'Reviewed',
};

export const sortNewestFirst = (events: PossibleEvent[]) =>
  [...events].sort((a, b) => b.timestamp - a.timestamp);

export const filterByReview = (events: PossibleEvent[], f: 'all' | ReviewStatus) =>
  f === 'all' ? events : events.filter((e) => e.review === f);

export function nextEventId(events: PossibleEvent[], ts: number): string {
  const prefix = `EVT-${dayKey(ts)}-`;
  const n = events.filter((e) => e.id.startsWith(prefix)).length + 1;
  return `${prefix}${String(n).padStart(3, '0')}`;
}

export function eventStats(events: PossibleEvent[]) {
  const sorted = sortNewestFirst(events);
  const latest = sorted[0]?.timestamp ?? Date.now();
  const DAY = 86_400_000;
  // Windows end at the most recent record so bundled sample dates stay meaningful.
  const inLast = (days: number) => events.filter((e) => latest - e.timestamp < days * DAY).length;
  const perDay = Array.from({ length: 14 }, (_, i) => {
    const ts = latest - (13 - i) * DAY;
    return { key: dayKey(ts), ts, count: events.filter((e) => dayKey(e.timestamp) === dayKey(ts)).length };
  });
  return {
    total: events.length,
    needsReview: events.filter((e) => e.review === 'needs_review').length,
    reviewed: events.filter((e) => e.review === 'reviewed').length,
    cancelled: events.filter((e) => e.outcome === 'cancelled').length,
    last7: inLast(7),
    last14: inLast(14),
    perDay,
    anchor: latest,
  };
}
