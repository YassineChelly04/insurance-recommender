/**
 * App.tsx — Root component.
 *
 * Sets up React Navigation v7, initialises the database on startup,
 * performs a security check, and triggers an OTA model update check.
 */

import React, { useEffect } from 'react';
import { Alert, Platform } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import { CameraScreen } from './screens/CameraScreen';
import { HistoryScreen } from './screens/HistoryScreen';
import { openDB } from './services/storage';
import { checkDeviceSecurity } from './services/security';
import { useOTAModelUpdater } from './hooks/useOTAModelUpdater';

const Stack = createNativeStackNavigator();

function AppNavigator() {
  const { checkForUpdate } = useOTAModelUpdater();

  useEffect(() => {
    // 1. Open (and migrate) encrypted database.
    openDB().catch((err) =>
      console.error('Failed to open database:', err),
    );

    // 2. Device security check.
    checkDeviceSecurity().then(({ isCompromised, reasons }) => {
      if (isCompromised && process.env.NODE_ENV === 'production') {
        Alert.alert(
          'Security Warning',
          `This device may be compromised:\n\n${reasons.join('\n')}\n\nSensitive features may be restricted.`,
          [{ text: 'OK' }],
        );
      }
    });

    // 3. OTA model update check (WiFi-only, background).
    checkForUpdate().catch(() => {
      // Non-critical — silently ignore if update check fails.
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <NavigationContainer>
      <Stack.Navigator
        initialRouteName="Camera"
        screenOptions={{
          headerStyle: { backgroundColor: '#1c1c1e' },
          headerTintColor: '#fff',
          headerTitleStyle: { fontWeight: '700' },
        }}
      >
        <Stack.Screen
          name="Camera"
          component={CameraScreen}
          options={{ title: 'Detect', headerShown: false }}
        />
        <Stack.Screen
          name="History"
          component={HistoryScreen}
          options={{ title: 'Scan History' }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

export default function App() {
  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <SafeAreaProvider>
        <AppNavigator />
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
}
