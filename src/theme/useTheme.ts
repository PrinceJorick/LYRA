import { useMemo } from 'react';
import { StyleSheet, useColorScheme, type ImageStyle, type TextStyle, type ViewStyle } from 'react-native';
import { useApp } from '../state/AppState';
import { dark, light, type Palette } from './colors';

export function useTheme(): Palette & { isDark: boolean } {
  const { themeMode } = useApp();
  const system = useColorScheme();
  const isDark = themeMode === 'dark' || (themeMode === 'system' && system === 'dark');
  return useMemo(() => ({ ...(isDark ? dark : light), isDark }), [isDark]);
}

/** Memoised themed stylesheet: `const s = useStyles((c) => ({ box: { backgroundColor: c.surface } }))`. */
export function useStyles<T extends Record<string, ViewStyle | TextStyle | ImageStyle>>(factory: (c: Palette) => T): T {
  const c = useTheme();
  // eslint-disable-next-line react-hooks/exhaustive-deps -- factory is a stable module-level function
  return useMemo(() => StyleSheet.create(factory(c)), [c]);
}
