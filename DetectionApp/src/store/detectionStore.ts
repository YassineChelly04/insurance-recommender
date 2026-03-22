/**
 * detectionStore.ts — Lightweight global state with Zustand.
 *
 * Blueprint Part 2 (Tech Stack): Zustand is recommended for MVP because it is
 * lightweight and avoids the boilerplate of Redux.
 */

import { create } from 'zustand';
import type { Detection, DetectionRecord, OTAUpdateProgress } from '../types';

interface DetectionState {
  // Live camera detections (current frame)
  liveDetections: Detection[];
  setLiveDetections: (detections: Detection[]) => void;

  // Active scanning session
  currentSessionId: string | null;
  startSession: (sessionId: string) => void;
  endSession: () => void;

  // History loaded from SQLite
  history: DetectionRecord[];
  setHistory: (records: DetectionRecord[]) => void;
  appendHistory: (record: DetectionRecord) => void;

  // OTA update state
  otaUpdate: OTAUpdateProgress;
  setOtaUpdate: (update: OTAUpdateProgress) => void;

  // User settings
  confidenceThreshold: number;
  setConfidenceThreshold: (value: number) => void;
}

export const useDetectionStore = create<DetectionState>((set) => ({
  liveDetections: [],
  setLiveDetections: (liveDetections) => set({ liveDetections }),

  currentSessionId: null,
  startSession: (sessionId) => set({ currentSessionId: sessionId }),
  endSession: () => set({ currentSessionId: null }),

  history: [],
  setHistory: (history) => set({ history }),
  appendHistory: (record) =>
    set((state) => ({ history: [record, ...state.history] })),

  otaUpdate: { status: 'idle', progress: 0 },
  setOtaUpdate: (otaUpdate) => set({ otaUpdate }),

  confidenceThreshold: 0.5,
  setConfidenceThreshold: (confidenceThreshold) =>
    set({ confidenceThreshold }),
}));
