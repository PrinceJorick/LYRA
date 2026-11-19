import { ChevronRight, Info } from 'lucide-react-native';
import React from 'react';
import {
  KeyboardAvoidingView, Platform, Pressable, ScrollView, Text, TextInput, View,
  type StyleProp, type TextInputProps, type TextStyle, type ViewStyle,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { radius, space, TOUCH, type Palette } from '../theme/colors';
import { useStyles, useTheme } from '../theme/useTheme';

/* ---------- Text ---------- */
type Variant = 'title' | 'h2' | 'h3' | 'body' | 'small' | 'caption' | 'label';
const sizes: Record<Variant, TextStyle> = {
  title: { fontSize: 28, fontWeight: '700', lineHeight: 34 },
  h2: { fontSize: 20, fontWeight: '700', lineHeight: 26 },
  h3: { fontSize: 17, fontWeight: '600', lineHeight: 23 },
  body: { fontSize: 16, lineHeight: 23 },
  small: { fontSize: 14, lineHeight: 20 },
  caption: { fontSize: 13, lineHeight: 18 },
  label: { fontSize: 12, fontWeight: '700', letterSpacing: 0.8, textTransform: 'uppercase' },
};

export function Txt({
  v = 'body', muted, color, style, ...rest
}: React.ComponentProps<typeof Text> & { v?: Variant; muted?: boolean; color?: string }) {
  const c = useTheme();
  return <Text {...rest} style={[{ color: color ?? (muted ? c.textMuted : c.text) }, sizes[v], style]} />;
}

/* ---------- Screen ---------- */
export function Screen({
  children, title, subtitle, top, keyboard, footer, scroll = true, contentStyle,
}: {
  children: React.ReactNode;
  title?: string;
  subtitle?: string;
  /** Apply the top safe-area inset (tab screens have no native header). */
  top?: boolean;
  keyboard?: boolean;
  footer?: React.ReactNode;
  scroll?: boolean;
  contentStyle?: StyleProp<ViewStyle>;
}) {
  const c = useTheme();
  const insets = useSafeAreaInsets();
  const body = (
    <>
      {title ? (
        <View style={{ marginBottom: space.lg }}>
          <Txt v="title" accessibilityRole="header">{title}</Txt>
          {subtitle ? <Txt muted style={{ marginTop: 4 }}>{subtitle}</Txt> : null}
        </View>
      ) : null}
      {children}
    </>
  );
  const pad: ViewStyle = { padding: space.lg, paddingTop: (top ? insets.top : 0) + space.lg, gap: space.lg };
  const content = scroll ? (
    <ScrollView
      keyboardShouldPersistTaps="handled"
      keyboardDismissMode="on-drag"
      contentContainerStyle={[pad, { paddingBottom: space.xxl }, contentStyle]}
    >
      {body}
    </ScrollView>
  ) : (
    <View style={[{ flex: 1 }, pad, contentStyle]}>{body}</View>
  );
  return (
    <KeyboardAvoidingView
      style={{ flex: 1, backgroundColor: c.background }}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      enabled={!!keyboard}
    >
      {content}
      {footer ? (
        <View style={{ padding: space.lg, paddingBottom: space.lg + insets.bottom, backgroundColor: c.background, gap: space.sm }}>
          {footer}
        </View>
      ) : null}
    </KeyboardAvoidingView>
  );
}

/* ---------- Card ---------- */
const cardStyles = (c: Palette) => ({
  card: {
    backgroundColor: c.surface,
    borderRadius: radius.lg,
    padding: space.lg,
    shadowColor: c.primaryDeep,
    shadowOpacity: 0.07,
    shadowRadius: 14,
    shadowOffset: { width: 0, height: 4 },
    elevation: 2,
  } as ViewStyle,
});

export function Card({ children, style, ...rest }: React.ComponentProps<typeof View>) {
  const s = useStyles(cardStyles);
  return <View {...rest} style={[s.card, style]}>{children}</View>;
}

/* ---------- Button ---------- */
type BtnKind = 'primary' | 'secondary' | 'ghost' | 'danger';

export function Button({
  label, onPress, kind = 'primary', icon, disabled, style, accessibilityHint,
}: {
  label: string;
  onPress: () => void;
  kind?: BtnKind;
  icon?: React.ReactNode;
  disabled?: boolean;
  style?: StyleProp<ViewStyle>;
  accessibilityHint?: string;
}) {
  const c = useTheme();
  const palette: Record<BtnKind, { bg: string; fg: string; border?: string }> = {
    primary: { bg: c.primary, fg: c.onPrimary },
    secondary: { bg: 'transparent', fg: c.primary, border: c.primary },
    ghost: { bg: c.tint, fg: c.isDark ? c.text : c.primaryDeep },
    danger: { bg: 'transparent', fg: c.danger, border: c.danger },
  };
  const p = palette[kind];
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={label}
      accessibilityHint={accessibilityHint}
      accessibilityState={{ disabled: !!disabled }}
      disabled={disabled}
      onPress={onPress}
      style={({ pressed }) => [
        {
          minHeight: 52, paddingHorizontal: space.xl, borderRadius: radius.pill,
          backgroundColor: p.bg, borderWidth: p.border ? 1.5 : 0, borderColor: p.border,
          alignItems: 'center', justifyContent: 'center', flexDirection: 'row', gap: space.sm,
          opacity: disabled ? 0.45 : pressed ? 0.85 : 1,
        },
        style,
      ]}
    >
      {icon}
      <Text style={{ color: p.fg, fontSize: 16, fontWeight: '600' }}>{label}</Text>
    </Pressable>
  );
}

/* ---------- Chip (filter / multi-select) ---------- */
export function Chip({
  label, selected, onPress, role = 'button',
}: { label: string; selected: boolean; onPress: () => void; role?: 'button' | 'radio' | 'checkbox' }) {
  const c = useTheme();
  return (
    <Pressable
      accessibilityRole={role}
      accessibilityState={role === 'checkbox' ? { checked: selected } : { selected }}
      onPress={onPress}
      style={{
        minHeight: 44, paddingHorizontal: space.lg, borderRadius: radius.pill, justifyContent: 'center',
        backgroundColor: selected ? c.primary : c.surface,
        borderWidth: 1, borderColor: selected ? c.primary : c.divider,
      }}
    >
      <Text style={{ color: selected ? c.onPrimary : c.text, fontWeight: '600', fontSize: 15 }}>
        {selected && role !== 'button' ? '✓ ' : ''}{label}
      </Text>
    </Pressable>
  );
}

/* ---------- Badge: always icon/text, never colour alone ---------- */
export type Tone = 'purple' | 'success' | 'warning' | 'neutral';

export function Badge({ label, tone = 'purple', icon }: { label: string; tone?: Tone; icon?: React.ReactNode }) {
  const c = useTheme();
  const t = {
    purple: { bg: c.tint, fg: c.isDark ? c.text : c.primaryDeep },
    success: { bg: c.successBg, fg: c.successText },
    warning: { bg: c.warningBg, fg: c.warningText },
    neutral: { bg: c.divider, fg: c.textMuted },
  }[tone];
  return (
    <View
      style={{
        flexDirection: 'row', alignItems: 'center', gap: 4, alignSelf: 'flex-start',
        backgroundColor: t.bg, borderRadius: radius.pill, paddingHorizontal: 10, paddingVertical: 4,
      }}
    >
      {icon}
      <Text style={{ color: t.fg, fontSize: 12.5, fontWeight: '700' }}>{label}</Text>
    </View>
  );
}

export const DemoBadge = ({ label = 'DEMO MODE' }: { label?: string }) => <Badge label={label} tone="warning" />;

/* ---------- Section header ---------- */
export function SectionHeader({
  title, stage, action,
}: { title: string; stage?: 'BEFORE' | 'DURING' | 'AFTER'; action?: { label: string; onPress: () => void } }) {
  const c = useTheme();
  return (
    <View style={{ flexDirection: 'row', alignItems: 'flex-end', justifyContent: 'space-between' }}>
      <View style={{ flexShrink: 1 }}>
        {stage ? <Txt v="label" color={c.secondary}>{stage}</Txt> : null}
        <Txt v="h2" accessibilityRole="header">{title}</Txt>
      </View>
      {action ? (
        <Pressable
          onPress={action.onPress}
          accessibilityRole="link"
          style={{ minHeight: 44, justifyContent: 'center', paddingLeft: space.md }}
        >
          <Txt v="small" color={c.primary} style={{ fontWeight: '600' }}>{action.label} →</Txt>
        </Pressable>
      ) : null}
    </View>
  );
}

/* ---------- Pressable row with chevron ---------- */
export function NavRow({
  title, subtitle, onPress, left, right,
}: { title: string; subtitle?: string; onPress: () => void; left?: React.ReactNode; right?: React.ReactNode }) {
  const c = useTheme();
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={subtitle ? `${title}. ${subtitle}` : title}
      onPress={onPress}
      style={{ minHeight: TOUCH + 8, flexDirection: 'row', alignItems: 'center', gap: space.md, paddingVertical: space.sm }}
    >
      {left}
      <View style={{ flex: 1 }}>
        <Txt v="h3">{title}</Txt>
        {subtitle ? <Txt v="small" muted>{subtitle}</Txt> : null}
      </View>
      {right}
      <ChevronRight size={20} color={c.textMuted} />
    </Pressable>
  );
}

export function Divider() {
  const c = useTheme();
  return <View style={{ height: 1, backgroundColor: c.divider }} />;
}

/* ---------- Notices ---------- */
export function Notice({ children, tone = 'purple' }: { children: React.ReactNode; tone?: 'purple' | 'warning' }) {
  const c = useTheme();
  const bg = tone === 'warning' ? c.warningBg : c.tint;
  const fg = tone === 'warning' ? c.warningText : c.primary;
  return (
    <View
      accessible
      style={{ flexDirection: 'row', gap: space.md, backgroundColor: bg, borderRadius: radius.md, padding: space.md }}
    >
      <Info size={20} color={fg} style={{ marginTop: 2 }} />
      <Txt v="small" style={{ flex: 1 }}>{children}</Txt>
    </View>
  );
}

export const SAFETY_SHORT =
  'LYRA is an experimental research project. This application does not provide medical diagnosis or validated emergency monitoring.';
export const SAFETY_LONG =
  'LYRA is an experimental research project. It is not a diagnostic or treatment device. Its notifications and model outputs are not clinically validated and must not replace professional medical evaluation or emergency services.';

export const SafetyNotice = ({ long }: { long?: boolean }) => <Notice>{long ? SAFETY_LONG : SAFETY_SHORT}</Notice>;

/* ---------- Form field ---------- */
export function Field({
  label, hint, error, ...input
}: TextInputProps & { label: string; hint?: string; error?: string }) {
  const c = useTheme();
  return (
    <View style={{ gap: 6 }}>
      <Txt v="small" style={{ fontWeight: '600' }}>{label}</Txt>
      <TextInput
        accessibilityLabel={label}
        accessibilityHint={hint}
        placeholderTextColor={c.textMuted}
        {...input}
        style={[
          {
            minHeight: TOUCH, borderRadius: radius.md, borderWidth: 1.5, paddingHorizontal: space.md,
            paddingVertical: space.md, fontSize: 16, color: c.text, backgroundColor: c.surface,
            borderColor: error ? c.danger : c.divider,
          },
          input.multiline ? { minHeight: 96, textAlignVertical: 'top' } : null,
          input.style,
        ]}
      />
      {error ? <Txt v="caption" color={c.danger} accessibilityLiveRegion="polite">{error}</Txt> : null}
    </View>
  );
}

export function KeyValue({ label, value }: { label: string; value: string }) {
  return (
    <View style={{ flexDirection: 'row', justifyContent: 'space-between', gap: space.lg, paddingVertical: 6 }}>
      <Txt v="small" muted style={{ flexShrink: 0, maxWidth: '45%' }}>{label}</Txt>
      <Txt v="small" style={{ flex: 1, textAlign: 'right', fontWeight: '600' }}>{value}</Txt>
    </View>
  );
}
