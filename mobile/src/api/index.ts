import Constants from 'expo-constants';
import { Platform } from 'react-native';

import { createClient } from './client';
import type { InspectionDetail } from './types';

export * from './client';
export type * from './types';

/**
 * Where the API lives. EXPO_PUBLIC_API_URL wins (set it for the deployed backend). Otherwise, in development,
 * use port 8000 on the machine running the Expo dev server, so Expo Go on a phone reaches your laptop's API.
 */
function resolveBaseUrl(): string {
  const configured = process.env.EXPO_PUBLIC_API_URL;
  if (configured) return configured;
  const host = Constants.expoConfig?.hostUri?.split(':')[0];
  if (host && Platform.OS !== 'web') return `http://${host}:8000`;
  return 'http://localhost:8000';
}

export const api = createClient({ baseUrl: resolveBaseUrl() });

/** Results just returned by /predict, so the Result screen can show them without a second request. */
export const recentResults = new Map<number, InspectionDetail>();
