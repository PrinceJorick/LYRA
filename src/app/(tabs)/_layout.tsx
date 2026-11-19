import { ChartColumn, Clock, FileText, Heart, House } from 'lucide-react-native';
import { Tabs } from 'expo-router';
import React from 'react';
import { Platform, type ColorValue } from 'react-native';
import { useTheme } from '../../theme/useTheme';

const icon = (Icon: typeof House) =>
  function TabIcon({ color, size }: { color: ColorValue; size: number }) {
    return <Icon size={size} color={color as string} />;
  };

const webBar = { height: 68, paddingBottom: 8, paddingTop: 6 };

export default function TabsLayout() {
  const c = useTheme();
  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarActiveTintColor: c.primary,
        tabBarInactiveTintColor: c.textMuted,
        // Web needs an explicit height or the labels clip; native sizes the bar itself.
        tabBarStyle: { backgroundColor: c.surface, borderTopColor: c.divider, ...(Platform.OS === 'web' ? webBar : {}) },
        tabBarLabelStyle: { fontSize: 12, fontWeight: '600' },
        sceneStyle: { backgroundColor: c.background },
      }}
    >
      <Tabs.Screen name="index" options={{ title: 'Home', tabBarIcon: icon(House) }} />
      <Tabs.Screen name="history" options={{ title: 'History', tabBarIcon: icon(Clock) }} />
      <Tabs.Screen name="insights" options={{ title: 'Insights', tabBarIcon: icon(ChartColumn) }} />
      <Tabs.Screen name="reports" options={{ title: 'Reports', tabBarIcon: icon(FileText) }} />
      <Tabs.Screen name="care" options={{ title: 'Care', tabBarIcon: icon(Heart) }} />
    </Tabs>
  );
}
