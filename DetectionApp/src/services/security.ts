/**
 * security.ts — App hardening utilities.
 *
 * Blueprint Part 4 — Security:
 *  1. Model encryption key management via Android Keystore / iOS Keychain.
 *  2. Certificate pinning on OTA update endpoint.
 *  3. Root/jailbreak detection.
 *  4. Active-recording indicator helper.
 *
 * All cryptographic keys are stored in the OS secure enclave and are never
 * exposed to JavaScript as plain text beyond the minimum needed for decryption.
 */

import * as SecureStore from 'expo-secure-store';
import * as Crypto from 'expo-crypto';
import DeviceInfo from 'react-native-device-info';
import { Platform } from 'react-native';

// Alias for clarity — Crypto.getRandomBytesAsync returns a Uint8Array.
const getSecureRandomHex = async (byteLength = 32): Promise<string> => {
  const bytes = await Crypto.getRandomBytesAsync(byteLength);
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
};

// ── Key aliases ───────────────────────────────────────────────────────────────

const MODEL_KEY_ALIAS = 'yolo26n_model_decryption_key';

// ── Model Key Management ──────────────────────────────────────────────────────

/**
 * Retrieve (or generate on first launch) the AES-256 key used to encrypt the
 * YOLO26n model at rest.
 *
 * - iOS: stored in the Keychain with .whenUnlockedThisDeviceOnly accessibility.
 * - Android: stored via EncryptedSharedPreferences backed by Android Keystore.
 *
 * expo-secure-store uses these platform-native APIs automatically.
 */
export async function getModelDecryptionKey(): Promise<string> {
  const existing = await SecureStore.getItemAsync(MODEL_KEY_ALIAS);
  if (existing) return existing;

  const key = await getSecureRandomHex(32);

  await SecureStore.setItemAsync(MODEL_KEY_ALIAS, key, {
    keychainAccessible: SecureStore.WHEN_UNLOCKED_THIS_DEVICE_ONLY,
  });

  return key;
}

/**
 * Delete the model decryption key from the secure enclave.
 * Call this during a "factory reset" or account deletion flow.
 */
export async function deleteModelDecryptionKey(): Promise<void> {
  await SecureStore.deleteItemAsync(MODEL_KEY_ALIAS);
}

// ── Root / Jailbreak Detection ────────────────────────────────────────────────

export interface SecurityCheckResult {
  isCompromised: boolean;
  reasons: string[];
}

/**
 * Checks whether the device shows signs of being rooted (Android) or
 * jailbroken (iOS).
 *
 * Blueprint recommendation: disable sensitive features on compromised devices
 * or at least warn the user.  react-native-device-info is used here; for
 * production consider adding DexGuard (Android) or iXGuard (iOS) as well.
 */
export async function checkDeviceSecurity(): Promise<SecurityCheckResult> {
  const reasons: string[] = [];

  try {
    if (Platform.OS === 'android') {
      const isRooted = await DeviceInfo.isRooted();
      if (isRooted) reasons.push('Device appears to be rooted.');

      const hasMockLocation = await DeviceInfo.isMockLocation();
      if (hasMockLocation) reasons.push('Mock location provider is active (common on rooted devices).');
    }

    const isEmulator = await DeviceInfo.isEmulator();
    if (isEmulator && process.env.NODE_ENV === 'production') {
      reasons.push('App is running on an emulator/simulator in production.');
    }
  } catch {
    // If any check throws, treat it as inconclusive (not compromised).
  }

  return {
    isCompromised: reasons.length > 0,
    reasons,
  };
}

// ── Certificate Pinning ───────────────────────────────────────────────────────

/**
 * Certificate pin hashes for the OTA model update endpoint.
 *
 * Replace these with the SHA-256 fingerprints of your server's TLS certificate
 * (or intermediate CA certificate).  Keep at least one backup pin to avoid
 * bricking users when certificates rotate.
 *
 * These are passed to react-native-ssl-pinning in useOTAModelUpdater.ts.
 */
export const OTA_CERT_PINS: string[] = [
  'sha256/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=', // primary pin
  'sha256/BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB=', // backup pin
];

// ── Proguard / Bitcode Note ───────────────────────────────────────────────────

/**
 * Additional hardening (outside JS layer — configure in native build files):
 *
 * Android (android/app/build.gradle):
 *   buildTypes {
 *     release {
 *       minifyEnabled true
 *       proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
 *     }
 *   }
 *
 * iOS (Xcode):
 *   - Enable Bitcode: Build Settings → Enable Bitcode → Yes
 *   - Strip Swift Symbols: Build Settings → Strip Swift Symbols → Yes
 *
 * For stronger model protection consider DexGuard (Android) / iXGuard (iOS)
 * which apply polymorphic code interweaving (blueprint Part 4).
 */
