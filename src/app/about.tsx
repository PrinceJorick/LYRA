import React from 'react';
import { LyraLogo } from '../components/graphics';
import { Card, SafetyNotice, Screen, SectionHeader, Txt } from '../components/ui';
import { mlService } from '../services/ml';
import { space } from '../theme/colors';

export default function About() {
  const meta = mlService.getMetadata();
  return (
    <Screen>
      <LyraLogo size={56} />
      <Txt v="small" muted>AI-Assisted Epilepsy Monitoring and Caregiver-Support System</Txt>
      <Card style={{ gap: space.md }}>
        <Txt>
          LYRA is an AI-assisted epilepsy monitoring and caregiver-support research project exploring the use of
          surface electromyography and motion signals to identify possible convulsive seizure events.
        </Txt>
      </Card>

      <SectionHeader title="How LYRA is organised" />
      <Card style={{ gap: space.md }}>
        <Txt><Txt style={{ fontWeight: '700' }}>Before: </Txt>record context such as triggers and your own medication notes.</Txt>
        <Txt><Txt style={{ fontWeight: '700' }}>During: </Txt>a possible-event alert with a cancel countdown, then simulated caregiver notification.</Txt>
        <Txt><Txt style={{ fontWeight: '700' }}>After: </Txt>record observations and prepare a summary to discuss with a clinician.</Txt>
      </Card>

      <SectionHeader title="Current status" />
      <Card style={{ gap: space.sm }}>
        <Txt v="small">• Pod pairing, battery and caregiver notification are simulated.</Txt>
        <Txt v="small">• ML model: {meta.status === 'mock' ? 'not integrated' : `${meta.modelId} v${meta.version}`}.</Txt>
        <Txt v="small">• All records are sample or simulated data unless stated otherwise.</Txt>
      </Card>
      <SafetyNotice long />
    </Screen>
  );
}
