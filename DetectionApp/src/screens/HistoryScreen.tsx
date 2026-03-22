/**
 * HistoryScreen — displays paginated scan history from encrypted SQLite, with
 * JSON export capability.
 *
 * Blueprint Part 5: export detection records as JSON via the native share sheet.
 */

import React, { useEffect, useState, useCallback } from 'react';
import {
  View,
  FlatList,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
} from 'react-native';
import * as FileSystem from 'expo-file-system';
import * as Sharing from 'expo-sharing';

import { DetectionResultCard } from '../components/DetectionResultCard';
import {
  openDB,
  getAllDetections,
  deleteDetectionsBySession,
  getStorageStats,
} from '../services/storage';
import type { DetectionRecord } from '../types';

const PAGE_SIZE = 50;

export const HistoryScreen: React.FC = () => {
  const [records, setRecords] = useState<DetectionRecord[]>([]);
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [stats, setStats] = useState({ totalRecords: 0, totalSessions: 0 });
  const [loading, setLoading] = useState(false);

  const loadPage = useCallback(
    async (pageOffset: number, reset = false) => {
      if (loading) return;
      setLoading(true);
      try {
        const db = await openDB();
        const page = await getAllDetections(db, PAGE_SIZE, pageOffset);
        const s = await getStorageStats(db);
        setStats(s);
        setRecords((prev) => (reset ? page : [...prev, ...page]));
        setHasMore(page.length === PAGE_SIZE);
        setOffset(pageOffset + PAGE_SIZE);
      } finally {
        setLoading(false);
      }
    },
    [loading],
  );

  useEffect(() => {
    loadPage(0, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const exportHistory = useCallback(async () => {
    try {
      const db = await openDB();
      const all = await getAllDetections(db, 10_000, 0);
      const json = JSON.stringify(all, null, 2);
      const path = `${FileSystem.documentDirectory}scan_export_${Date.now()}.json`;
      await FileSystem.writeAsStringAsync(path, json);
      await Sharing.shareAsync(path);
    } catch (err) {
      Alert.alert('Export failed', String(err));
    }
  }, []);

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.statsText}>
          {stats.totalRecords} records · {stats.totalSessions} sessions
        </Text>
        <TouchableOpacity onPress={exportHistory} style={styles.exportButton}>
          <Text style={styles.exportText}>Export JSON</Text>
        </TouchableOpacity>
      </View>

      <FlatList
        data={records}
        keyExtractor={(item, i) => `${item.id ?? i}`}
        renderItem={({ item }) => <DetectionResultCard record={item} />}
        onEndReached={() => hasMore && loadPage(offset)}
        onEndReachedThreshold={0.4}
        contentContainerStyle={styles.list}
        ListEmptyComponent={
          <Text style={styles.emptyText}>No scans yet. Open the camera to start detecting!</Text>
        }
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f2f2f7' },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: '#fff',
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#c6c6c8',
  },
  statsText: { fontSize: 13, color: '#3c3c43' },
  exportButton: {
    backgroundColor: '#007AFF',
    borderRadius: 14,
    paddingHorizontal: 14,
    paddingVertical: 6,
  },
  exportText: { color: '#fff', fontSize: 13, fontWeight: '600' },
  list: { paddingVertical: 8 },
  emptyText: {
    textAlign: 'center',
    marginTop: 60,
    fontSize: 15,
    color: '#8e8e93',
    paddingHorizontal: 32,
  },
});
