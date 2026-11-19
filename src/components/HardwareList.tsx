import React from 'react';
import { View } from 'react-native';
import { CSI_NOTE, HARDWARE_PARTS, HARDWARE_STATUS_NOTE } from '../data/mockDevice';
import { space } from '../theme/colors';
import { useTheme } from '../theme/useTheme';
import { Card, Notice, SectionHeader, Txt } from './ui';

export function HardwareList() {
  const c = useTheme();
  return (
    <>
      <SectionHeader title="Hardware overview" />
      <Card style={{ gap: space.lg }}>
        {HARDWARE_PARTS.map((p) => (
          <View key={p.component} style={{ gap: 2 }}>
            <Txt v="label" color={c.secondary}>{p.component}</Txt>
            <Txt v="small" style={{ fontWeight: '700' }}>{p.part}</Txt>
            <Txt v="caption" muted>{p.purpose}</Txt>
          </View>
        ))}
      </Card>
      <Notice>{HARDWARE_STATUS_NOTE}</Notice>
      <Notice>{CSI_NOTE}</Notice>
    </>
  );
}
