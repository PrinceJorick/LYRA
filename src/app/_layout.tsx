import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import React from 'react';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { AppProvider } from '../state/AppState';
import { useTheme } from '../theme/useTheme';

function Navigator() {
  const c = useTheme();
  return (
    <>
      <StatusBar style={c.isDark ? 'light' : 'dark'} />
      <Stack
        screenOptions={{
          headerStyle: { backgroundColor: c.background },
          headerTintColor: c.primary,
          headerTitleStyle: { color: c.text, fontWeight: '700' },
          headerShadowVisible: false,
          headerBackButtonDisplayMode: 'minimal',
          contentStyle: { backgroundColor: c.background },
        }}
      >
        <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
        <Stack.Screen name="pairing" options={{ title: 'Pair LYRA Pod' }} />
        <Stack.Screen name="device" options={{ title: 'Pod Details' }} />
        <Stack.Screen name="event/[id]" options={{ title: 'Event Details' }} />
        <Stack.Screen name="observation/[id]" options={{ title: 'Event Observations', presentation: 'modal' }} />
        <Stack.Screen name="summary/[id]" options={{ title: 'Clinician Summary' }} />
        <Stack.Screen name="context-log" options={{ title: "Today's Notes", presentation: 'modal' }} />
        <Stack.Screen name="caregiver" options={{ title: 'Caregiver', presentation: 'modal' }} />
        <Stack.Screen name="settings" options={{ title: 'Settings' }} />
        <Stack.Screen name="about" options={{ title: 'About LYRA' }} />
        {/* No back gesture: the alert is ended with the explicit Cancel Alert button. */}
        <Stack.Screen
          name="alert"
          options={{ headerShown: false, presentation: 'fullScreenModal', gestureEnabled: false, animation: 'fade' }}
        />
      </Stack>
    </>
  );
}

export default function RootLayout() {
  return (
    <SafeAreaProvider>
      <AppProvider>
        <Navigator />
      </AppProvider>
    </SafeAreaProvider>
  );
}
