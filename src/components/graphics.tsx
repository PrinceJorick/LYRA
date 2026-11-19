import React, { useId, useState } from 'react';
import { Image, View, type LayoutChangeEvent } from 'react-native';
import Svg, { Circle, Defs, LinearGradient, Path, Rect, Stop } from 'react-native-svg';
import { useTheme } from '../theme/useTheme';
import { Txt } from './ui';

/* ---------- Brand ---------- */
// assets/lyra-mark.png is the supplied side-profile head logo, converted to a transparent
// purple PNG. Replace the file to swap the mark; it is tinted white in dark mode.
const MARK = require('../../assets/lyra-mark.png');

export function LyraLogo({ size = 36, wordmark = true }: { size?: number; wordmark?: boolean }) {
  const c = useTheme();
  return (
    <View
      accessible
      accessibilityRole="image"
      accessibilityLabel="Lyra"
      style={{ flexDirection: 'row', alignItems: 'center', gap: 10 }}
    >
      <Image
        source={MARK}
        style={{ width: size * (396 / 516), height: size, tintColor: c.isDark ? '#FFFFFF' : undefined }}
        resizeMode="contain"
      />
      {wordmark ? (
        <Txt style={{ fontSize: size * 0.8, fontWeight: '800', letterSpacing: 0.5, color: c.isDark ? c.text : c.primaryDeep }}>
          Lyra
        </Txt>
      ) : null}
    </View>
  );
}

/* ---------- Pod ---------- */
// Abstract drawing of the pod from the Blender renders: white rounded dome on an adhesive
// patch, lavender side clips, status LED. No strap, no screen. Replace with a real render later.
export function PodGraphic({ size = 140, connected = true }: { size?: number; connected?: boolean }) {
  const c = useTheme();
  const gid = `dome${useId().replace(/\W/g, '')}`; // unique: several pods can be mounted at once
  return (
    <Svg
      width={size}
      height={size * 0.85}
      viewBox="0 0 200 170"
      accessibilityRole="image"
      accessibilityLabel={`LYRA Pod, ${connected ? 'connected' : 'not connected'}`}
    >
      <Defs>
        <LinearGradient id={gid} x1="0" y1="0" x2="1" y2="1">
          <Stop offset="0" stopColor="#FFFFFF" />
          <Stop offset="1" stopColor="#E9E4F4" />
        </LinearGradient>
      </Defs>
      <Rect x="6" y="8" width="188" height="150" rx="58" fill={c.isDark ? '#3A3050' : '#EFE8DC'} />
      <Path d="M160 140 L186 164 L168 168 L150 148 Z" fill={c.isDark ? '#4A4062' : '#F6F2EA'} />
      <Rect x="42" y="68" width="16" height="38" rx="5" fill="#B7A2E8" />
      <Rect x="142" y="68" width="16" height="38" rx="5" fill="#B7A2E8" />
      <Rect x="54" y="26" width="92" height="116" rx="30" fill={`url(#${gid})`} stroke="#DCD4EC" strokeWidth="1.5" />
      <Circle cx="100" cy="52" r="10" fill={connected ? '#4FD1F0' : '#B9B3C6'} opacity={0.25} />
      <Circle cx="100" cy="52" r="4.5" fill={connected ? '#3CC3E6' : '#A59FB4'} />
    </Svg>
  );
}

/* ---------- Waveform ---------- */
/** Draws `values` (roughly -1..1) as a line. Used only for illustrative sample data. */
export function Waveform({ values, height = 56, color }: { values: number[]; height?: number; color?: string }) {
  const c = useTheme();
  const [w, setW] = useState(0);
  const onLayout = (e: LayoutChangeEvent) => setW(e.nativeEvent.layout.width);
  const mid = height / 2;
  const d = w
    ? values
        .map((v, i) => `${i ? 'L' : 'M'}${((i / (values.length - 1)) * w).toFixed(1)} ${(mid - v * (mid - 3)).toFixed(1)}`)
        .join(' ')
    : '';
  return (
    <View onLayout={onLayout} style={{ height }} accessibilityElementsHidden importantForAccessibility="no-hide-descendants">
      {w ? (
        <Svg width={w} height={height}>
          <Path d={d} stroke={color ?? c.primary} strokeWidth={1.8} fill="none" strokeLinejoin="round" strokeLinecap="round" />
        </Svg>
      ) : null}
    </View>
  );
}

/* ---------- Battery ---------- */
export function Battery({ percent }: { percent: number }) {
  const c = useTheme();
  const low = percent <= 20;
  const fill = low ? c.warning : c.success;
  return (
    <View
      accessible
      accessibilityLabel={`Battery ${percent} percent${low ? ', low' : ''}`}
      style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}
    >
      <View style={{ flexDirection: 'row', alignItems: 'center' }}>
        <View style={{ width: 26, height: 13, borderRadius: 3, borderWidth: 1.5, borderColor: c.textMuted, padding: 1.5 }}>
          <View style={{ width: `${Math.max(4, percent)}%`, flex: 1, borderRadius: 1.5, backgroundColor: fill }} />
        </View>
        <View style={{ width: 2, height: 6, backgroundColor: c.textMuted, borderTopRightRadius: 1, borderBottomRightRadius: 1 }} />
      </View>
      <Txt v="small" style={{ fontWeight: '700' }}>{percent}%{low ? ' · Low' : ''}</Txt>
    </View>
  );
}

/* ---------- Countdown ring ---------- */
export function CountdownRing({
  progress, size = 240, children,
}: { progress: number; size?: number; children?: React.ReactNode }) {
  const c = useTheme();
  const stroke = 14;
  const r = (size - stroke) / 2;
  const circ = 2 * Math.PI * r;
  return (
    <View style={{ width: size, height: size, alignItems: 'center', justifyContent: 'center' }}>
      <Svg width={size} height={size} style={{ position: 'absolute', transform: [{ rotate: '-90deg' }] }}>
        <Circle cx={size / 2} cy={size / 2} r={r} stroke={c.divider} strokeWidth={stroke} fill="none" />
        <Circle
          cx={size / 2} cy={size / 2} r={r} stroke={c.warning} strokeWidth={stroke} fill="none"
          strokeLinecap="round" strokeDasharray={`${circ}`} strokeDashoffset={circ * (1 - Math.max(0, Math.min(1, progress)))}
        />
      </Svg>
      {children}
    </View>
  );
}
