// Brand palette from the project brief. Text-use variants (successText / warningText) are
// darkened from the brief's #299D70 / #D99A36 so small text meets WCAG AA on white.

export interface Palette {
  primary: string;
  primaryDeep: string;
  secondary: string;
  tint: string; // light lavender fills
  background: string;
  surface: string;
  text: string;
  textMuted: string;
  divider: string;
  onPrimary: string;
  success: string;
  successText: string;
  successBg: string;
  warning: string;
  warningText: string;
  warningBg: string;
  danger: string;
}

export const light: Palette = {
  primary: '#7138D9',
  primaryDeep: '#51239B',
  secondary: '#9165EA',
  tint: '#EDE5FF',
  background: '#F8F6FC',
  surface: '#FFFFFF',
  text: '#211D2D',
  textMuted: '#6B657A',
  divider: '#EEEAF4',
  onPrimary: '#FFFFFF',
  success: '#299D70',
  successText: '#1B7452',
  successBg: '#E3F5EC',
  warning: '#D99A36',
  warningText: '#8F5A0A',
  warningBg: '#FCF1DC',
  danger: '#B3261E',
};

// Deep-plum dark theme that keeps the violet identity.
export const dark: Palette = {
  primary: '#9165EA',
  primaryDeep: '#7138D9',
  secondary: '#B79BF5',
  tint: '#2E2347',
  background: '#150F20',
  surface: '#21192F',
  text: '#F5F1FC',
  textMuted: '#B7AECA',
  divider: '#352B49',
  onPrimary: '#FFFFFF',
  success: '#4CC38F',
  successText: '#6FD9A8',
  successBg: '#18322A',
  warning: '#E8B04F',
  warningText: '#F0C47A',
  warningBg: '#3A2D14',
  danger: '#F2837A',
};

export const radius = { sm: 10, md: 16, lg: 24, pill: 999 };
export const space = { xs: 4, sm: 8, md: 12, lg: 16, xl: 24, xxl: 32 };
/** Minimum touch target (points). */
export const TOUCH = 48;
