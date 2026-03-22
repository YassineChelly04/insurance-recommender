/**
 * Core domain types for the DetectionApp.
 *
 * DetectionRecord mirrors the encrypted SQLite schema defined in
 * src/services/storage.ts.
 */

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Detection {
  classIndex: number;
  label: string;
  confidence: number;
  boundingBox: BoundingBox;
}

export interface DetectionRecord {
  id?: number;
  sessionId: string;
  timestamp: number;
  label: string;
  classIndex: number;
  confidence: number;
  bboxX: number;
  bboxY: number;
  bboxWidth: number;
  bboxHeight: number;
  imagePath?: string;
}

export interface DetectionSession {
  sessionId: string;
  startedAt: number;
  detections: Detection[];
}

export type ModelState = 'loading' | 'loaded' | 'error';

export interface ModelInfo {
  version: string;
  checksum: string;
  size: number;
  url: string;
}

export interface OTAUpdateProgress {
  status: 'idle' | 'checking' | 'downloading' | 'verifying' | 'applying' | 'done' | 'error';
  progress: number;
  error?: string;
}
