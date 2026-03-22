/**
 * DetectionResultCard — shows a summary of a single DetectionRecord from
 * the scan history.
 */

import React from 'react';
import { View, Text, Image, StyleSheet } from 'react-native';
import type { DetectionRecord } from '../types';

interface DetectionResultCardProps {
  record: DetectionRecord;
}

export const DetectionResultCard: React.FC<DetectionResultCardProps> = ({
  record,
}) => {
  const date = new Date(record.timestamp).toLocaleString();
  const confidence = (record.confidence * 100).toFixed(1);

  return (
    <View style={styles.card}>
      {record.imagePath ? (
        <Image source={{ uri: record.imagePath }} style={styles.thumbnail} />
      ) : (
        <View style={[styles.thumbnail, styles.placeholder]}>
          <Text style={styles.placeholderText}>No image</Text>
        </View>
      )}
      <View style={styles.info}>
        <Text style={styles.label}>{record.label}</Text>
        <Text style={styles.confidence}>{confidence}% confidence</Text>
        <Text style={styles.meta}>{date}</Text>
        <Text style={styles.meta}>Session: {record.sessionId.slice(0, 8)}…</Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  card: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    borderRadius: 10,
    marginHorizontal: 16,
    marginVertical: 6,
    overflow: 'hidden',
    elevation: 2,
    shadowColor: '#000',
    shadowOpacity: 0.08,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 2 },
  },
  thumbnail: {
    width: 80,
    height: 80,
  },
  placeholder: {
    backgroundColor: '#e5e5ea',
    justifyContent: 'center',
    alignItems: 'center',
  },
  placeholderText: {
    fontSize: 10,
    color: '#8e8e93',
  },
  info: {
    flex: 1,
    padding: 10,
    justifyContent: 'center',
  },
  label: {
    fontSize: 15,
    fontWeight: '700',
    color: '#1c1c1e',
  },
  confidence: {
    fontSize: 13,
    color: '#34C759',
    marginTop: 2,
  },
  meta: {
    fontSize: 11,
    color: '#8e8e93',
    marginTop: 2,
  },
});
