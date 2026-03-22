/**
 * CameraScreen — live camera feed with real-time YOLO26n object detection.
 *
 * Blueprint Part 3 (Architecture):
 *  - Frame processor runs off the JS thread (worklet).
 *  - Active-recording indicator always visible (blueprint Part 4 privacy).
 *  - Haptic feedback on first new detection in a frame (blueprint §UX).
 *  - "Save scan" captures annotated screenshot to Camera Roll.
 */

import React, { useRef, useEffect, useCallback, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  Platform,
} from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
} from 'react-native-vision-camera';
import * as Haptics from 'expo-haptics';
import * as MediaLibrary from 'expo-media-library';
import { captureRef } from 'react-native-view-shot';
import { useNavigation } from '@react-navigation/native';

import { useObjectDetection } from '../hooks/useObjectDetection';
import { BoundingBoxOverlay } from '../components/BoundingBoxOverlay';
import { useDetectionStore } from '../store/detectionStore';
import { openDB, insertSession, insertDetectionRecords } from '../services/storage';

import type { Detection } from '../types';

function generateSessionId(): string {
  return `session_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

export const CameraScreen: React.FC = () => {
  const navigation = useNavigation();
  const device = useCameraDevice('back');
  const { hasPermission, requestPermission } = useCameraPermission();
  const cameraRef = useRef<Camera>(null);
  const overlayRef = useRef<View>(null);

  const { confidenceThreshold } = useDetectionStore();
  const { setLiveDetections, startSession, endSession, currentSessionId } =
    useDetectionStore();

  const sessionId = useRef<string>(generateSessionId());
  const lastDetectedLabels = useRef<Set<string>>(new Set());

  const { modelState, detections, frameProcessor, frameProcessorFps, warmup } =
    useObjectDetection({ confidenceThreshold });

  // ── Permissions ─────────────────────────────────────────────────────────────

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  // ── Session management ───────────────────────────────────────────────────────

  useEffect(() => {
    let dbHandle: Awaited<ReturnType<typeof openDB>> | null = null;

    (async () => {
      dbHandle = await openDB();
      await insertSession(dbHandle, sessionId.current, Date.now());
      startSession(sessionId.current);
      warmup();
    })();

    return () => {
      endSession();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Sync detections to store + DB ─────────────────────────────────────────

  useEffect(() => {
    setLiveDetections(detections);

    (async () => {
      if (detections.length === 0) return;
      const db = await openDB();
      const now = Date.now();

      // Haptic feedback for newly-appearing classes in this frame.
      for (const det of detections) {
        if (!lastDetectedLabels.current.has(det.label)) {
          Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(() => {});
        }
      }

      // Batch all inserts into a single SQLite transaction (avoids N+1 writes).
      await insertDetectionRecords(
        db,
        detections.map((det) => ({
          sessionId: sessionId.current,
          timestamp: now,
          label: det.label,
          classIndex: det.classIndex,
          confidence: det.confidence,
          bboxX: det.boundingBox.x,
          bboxY: det.boundingBox.y,
          bboxWidth: det.boundingBox.width,
          bboxHeight: det.boundingBox.height,
        })),
      );

      lastDetectedLabels.current = new Set(detections.map((d) => d.label));
    })();
  }, [detections, setLiveDetections]);

  // ── Save annotated screenshot ─────────────────────────────────────────────

  const saveScan = useCallback(async () => {
    try {
      const { status } = await MediaLibrary.requestPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission denied', 'Photo library access is required to save scans.');
        return;
      }

      const uri = await captureRef(overlayRef, {
        format: 'png',
        quality: 0.9,
      });

      const asset = await MediaLibrary.createAssetAsync(uri);
      await MediaLibrary.createAlbumAsync('DetectionApp Scans', asset, false);
      Alert.alert('Saved', 'Scan saved to your photo library.');
    } catch (err) {
      Alert.alert('Error', 'Could not save scan.');
    }
  }, []);

  // ── Early returns ────────────────────────────────────────────────────────

  if (!hasPermission) {
    return (
      <View style={styles.center}>
        <Text style={styles.infoText}>Camera permission is required.</Text>
        <TouchableOpacity onPress={requestPermission} style={styles.button}>
          <Text style={styles.buttonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (!device) {
    return (
      <View style={styles.center}>
        <Text style={styles.infoText}>No camera device found.</Text>
      </View>
    );
  }

  return (
    <View style={styles.container} ref={overlayRef} collapsable={false}>
      <Camera
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive
        frameProcessor={frameProcessor}
        frameProcessorFps={frameProcessorFps}
      />

      <BoundingBoxOverlay detections={detections} />

      {/* Active-recording indicator — required by blueprint Part 4 privacy rules */}
      <View style={styles.recordingBadge}>
        <View style={styles.recordingDot} />
        <Text style={styles.recordingText}>LIVE</Text>
      </View>

      {/* Model status */}
      {modelState !== 'loaded' && (
        <View style={styles.statusBanner}>
          <Text style={styles.statusText}>
            {modelState === 'loading' ? 'Loading model…' : 'Model failed to load.'}
          </Text>
        </View>
      )}

      {/* Action buttons */}
      <View style={styles.toolbar}>
        <TouchableOpacity onPress={saveScan} style={styles.button}>
          <Text style={styles.buttonText}>💾 Save Scan</Text>
        </TouchableOpacity>

        <TouchableOpacity
          onPress={() => navigation.navigate('History' as never)}
          style={styles.button}
        >
          <Text style={styles.buttonText}>📋 History</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000',
    padding: 24,
  },
  infoText: { color: '#fff', fontSize: 16, textAlign: 'center', marginBottom: 16 },
  recordingBadge: {
    position: 'absolute',
    top: Platform.OS === 'ios' ? 56 : 16,
    left: 16,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.55)',
    borderRadius: 12,
    paddingHorizontal: 10,
    paddingVertical: 4,
    gap: 6,
  },
  recordingDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#FF3B30',
  },
  recordingText: { color: '#fff', fontSize: 12, fontWeight: '700' },
  statusBanner: {
    position: 'absolute',
    top: Platform.OS === 'ios' ? 100 : 60,
    alignSelf: 'center',
    backgroundColor: 'rgba(0,0,0,0.7)',
    borderRadius: 8,
    paddingHorizontal: 14,
    paddingVertical: 6,
  },
  statusText: { color: '#fff', fontSize: 13 },
  toolbar: {
    position: 'absolute',
    bottom: Platform.OS === 'ios' ? 40 : 24,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 16,
  },
  button: {
    backgroundColor: 'rgba(255,255,255,0.15)',
    borderRadius: 20,
    paddingHorizontal: 18,
    paddingVertical: 10,
  },
  buttonText: { color: '#fff', fontSize: 14, fontWeight: '600' },
});
