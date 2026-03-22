/**
 * BoundingBoxOverlay — draws semi-transparent bounding boxes and labels over
 * the camera feed using React Native's absolute-positioned View elements.
 *
 * Blueprint Part 7 (UX): shows label + confidence on each detection; tap for
 * full metadata (progressive disclosure).
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Dimensions,
} from 'react-native';
import type { Detection } from '../types';

interface BoundingBoxOverlayProps {
  detections: Detection[];
  /** Width of the camera preview area in logical pixels. */
  previewWidth?: number;
  /** Height of the camera preview area in logical pixels. */
  previewHeight?: number;
}

const { width: SCREEN_W, height: SCREEN_H } = Dimensions.get('window');

const COLORS = [
  '#FF3B30', '#FF9500', '#FFCC00', '#34C759',
  '#5AC8FA', '#007AFF', '#5856D6', '#FF2D55',
];

function labelColor(classIndex: number): string {
  return COLORS[classIndex % COLORS.length];
}

function confidenceColor(confidence: number): string {
  if (confidence >= 0.8) return '#34C759';
  if (confidence >= 0.6) return '#FF9500';
  return '#FF3B30';
}

export const BoundingBoxOverlay: React.FC<BoundingBoxOverlayProps> = ({
  detections,
  previewWidth = SCREEN_W,
  previewHeight = SCREEN_H,
}) => {
  const [expanded, setExpanded] = useState<number | null>(null);

  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="box-none">
      {detections.map((det, idx) => {
        const { boundingBox: bb, label, confidence, classIndex } = det;
        const left   = bb.x      * previewWidth;
        const top    = bb.y      * previewHeight;
        const width  = bb.width  * previewWidth;
        const height = bb.height * previewHeight;
        const color  = labelColor(classIndex);
        const isOpen = expanded === idx;

        return (
          <TouchableOpacity
            key={idx}
            activeOpacity={0.8}
            onPress={() => setExpanded(isOpen ? null : idx)}
            style={[styles.box, { left, top, width, height, borderColor: color }]}
          >
            {/* Primary label chip — always visible */}
            <View style={[styles.labelChip, { backgroundColor: color }]}>
              <Text style={styles.labelText}>{label}</Text>
              <Text
                style={[
                  styles.confidenceText,
                  { color: confidenceColor(confidence) },
                ]}
              >
                {(confidence * 100).toFixed(0)}%
              </Text>
            </View>

            {/* Expanded metadata — progressive disclosure (blueprint §UX) */}
            {isOpen && (
              <View style={styles.metadataCard}>
                <Text style={styles.metaText}>Class: {classIndex}</Text>
                <Text style={styles.metaText}>
                  BBox: ({bb.x.toFixed(3)}, {bb.y.toFixed(3)})
                </Text>
                <Text style={styles.metaText}>
                  Size: {bb.width.toFixed(3)} × {bb.height.toFixed(3)}
                </Text>
                <Text style={styles.metaText}>
                  Conf: {(confidence * 100).toFixed(1)}%
                </Text>
              </View>
            )}
          </TouchableOpacity>
        );
      })}
    </View>
  );
};

const styles = StyleSheet.create({
  box: {
    position: 'absolute',
    borderWidth: 2,
    borderRadius: 4,
  },
  labelChip: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
    alignSelf: 'flex-start',
    gap: 4,
  },
  labelText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '700',
  },
  confidenceText: {
    fontSize: 11,
    fontWeight: '600',
  },
  metadataCard: {
    backgroundColor: 'rgba(0,0,0,0.75)',
    borderRadius: 6,
    padding: 8,
    marginTop: 4,
  },
  metaText: {
    color: '#fff',
    fontSize: 11,
  },
});
