/**
 * storage.ts — Encrypted SQLite persistence layer using expo-sqlite.
 *
 * Blueprint Part 4 (Security):
 *  - All detection metadata is stored with SQLCipher encryption.
 *  - Never use plain AsyncStorage for sensitive detection data.
 *
 * Blueprint Part 5 (Saving Output):
 *  - DetectionRecord stores class, confidence, bounding box, timestamp,
 *    session ID, and optional image path.
 *
 * Migration strategy:
 *  - SCHEMA_VERSION constant controls migrations.
 *  - applyMigrations() is idempotent and must be called once after openDB().
 */

import * as SQLite from 'expo-sqlite';
import * as SecureStore from 'expo-secure-store';
import * as Crypto from 'expo-crypto';

import type { DetectionRecord } from '../types';

const DB_NAME = 'detections.db';
const DB_KEY_ALIAS = 'detection_db_encryption_key';

// ── Key Management ────────────────────────────────────────────────────────────

/**
 * Generate a cryptographically secure 32-byte random hex string.
 * Uses expo-crypto's getRandomBytesAsync which delegates to the OS CSPRNG
 * (SecRandomCopyBytes on iOS, /dev/urandom on Android).
 */
async function generateSecureKey(): Promise<string> {
  const bytes = await Crypto.getRandomBytesAsync(32);
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

let _db: SQLite.SQLiteDatabase | null = null;

// ── Key Management ────────────────────────────────────────────────────────────

/**
 * Retrieve or create the database encryption key stored in the OS secure
 * enclave (iOS Keychain / Android Keystore via expo-secure-store).
 */
async function getOrCreateDbKey(): Promise<string> {
  const existing = await SecureStore.getItemAsync(DB_KEY_ALIAS);
  if (existing) return existing;

  const newKey = await generateSecureKey();
  await SecureStore.setItemAsync(DB_KEY_ALIAS, newKey, {
    keychainAccessible: SecureStore.WHEN_UNLOCKED_THIS_DEVICE_ONLY,
  });
  return newKey;
}

// ── Database Initialisation ───────────────────────────────────────────────────

export async function openDB(): Promise<SQLite.SQLiteDatabase> {
  if (_db) return _db;

  // expo-sqlite v15+ accepts a passphrase option that enables SQLCipher
  // transparent encryption.
  const passphrase = await getOrCreateDbKey();
  _db = await SQLite.openDatabaseAsync(DB_NAME, { passphrase });
  await applyMigrations(_db);
  return _db;
}

// ── Migrations ────────────────────────────────────────────────────────────────

/**
 * Apply all schema migrations in order.
 * Each migration is guarded by a version check so re-runs are safe.
 */
export async function applyMigrations(
  db: SQLite.SQLiteDatabase,
): Promise<void> {
  // schema_version table tracks applied migrations.
  await db.execAsync(`
    CREATE TABLE IF NOT EXISTS schema_version (
      version   INTEGER PRIMARY KEY,
      applied_at INTEGER NOT NULL
    );
  `);

  const row = await db.getFirstAsync<{ version: number }>(
    'SELECT MAX(version) AS version FROM schema_version;',
  );
  const currentVersion = row?.version ?? 0;

  if (currentVersion < 1) {
    await migration_v1(db);
    await db.runAsync(
      'INSERT INTO schema_version (version, applied_at) VALUES (?, ?);',
      [1, Date.now()],
    );
  }

  // Future migrations:
  // if (currentVersion < 2) { await migration_v2(db); ... }
}

/**
 * v1 — Initial schema.
 *
 * detection_records: one row per detected object per frame.
 * sessions: groups records belonging to the same scan session.
 */
async function migration_v1(db: SQLite.SQLiteDatabase): Promise<void> {
  await db.execAsync(`
    CREATE TABLE IF NOT EXISTS sessions (
      session_id  TEXT    PRIMARY KEY,
      started_at  INTEGER NOT NULL,
      ended_at    INTEGER
    );

    CREATE TABLE IF NOT EXISTS detection_records (
      id           INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id   TEXT    NOT NULL REFERENCES sessions(session_id),
      timestamp    INTEGER NOT NULL,
      label        TEXT    NOT NULL,
      class_index  INTEGER NOT NULL,
      confidence   REAL    NOT NULL,
      bbox_x       REAL    NOT NULL,
      bbox_y       REAL    NOT NULL,
      bbox_width   REAL    NOT NULL,
      bbox_height  REAL    NOT NULL,
      image_path   TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_records_session
      ON detection_records(session_id);

    CREATE INDEX IF NOT EXISTS idx_records_timestamp
      ON detection_records(timestamp);

    CREATE INDEX IF NOT EXISTS idx_records_label
      ON detection_records(label);
  `);
}

// ── CRUD helpers ──────────────────────────────────────────────────────────────

export async function insertSession(
  db: SQLite.SQLiteDatabase,
  sessionId: string,
  startedAt: number,
): Promise<void> {
  await db.runAsync(
    'INSERT OR IGNORE INTO sessions (session_id, started_at) VALUES (?, ?);',
    [sessionId, startedAt],
  );
}

export async function closeSession(
  db: SQLite.SQLiteDatabase,
  sessionId: string,
  endedAt: number,
): Promise<void> {
  await db.runAsync(
    'UPDATE sessions SET ended_at = ? WHERE session_id = ?;',
    [endedAt, sessionId],
  );
}

export async function insertDetectionRecord(
  db: SQLite.SQLiteDatabase,
  record: Omit<DetectionRecord, 'id'>,
): Promise<number> {
  const result = await db.runAsync(
    `INSERT INTO detection_records
       (session_id, timestamp, label, class_index, confidence,
        bbox_x, bbox_y, bbox_width, bbox_height, image_path)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);`,
    [
      record.sessionId,
      record.timestamp,
      record.label,
      record.classIndex,
      record.confidence,
      record.bboxX,
      record.bboxY,
      record.bboxWidth,
      record.bboxHeight,
      record.imagePath ?? null,
    ],
  );
  return result.lastInsertRowId;
}

/**
 * Batch-insert multiple detection records in a single transaction.
 * Prefer this over multiple calls to insertDetectionRecord to avoid N+1 writes.
 */
export async function insertDetectionRecords(
  db: SQLite.SQLiteDatabase,
  records: Omit<DetectionRecord, 'id'>[],
): Promise<void> {
  if (records.length === 0) return;
  await db.withTransactionAsync(async () => {
    for (const record of records) {
      await db.runAsync(
        `INSERT INTO detection_records
           (session_id, timestamp, label, class_index, confidence,
            bbox_x, bbox_y, bbox_width, bbox_height, image_path)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);`,
        [
          record.sessionId,
          record.timestamp,
          record.label,
          record.classIndex,
          record.confidence,
          record.bboxX,
          record.bboxY,
          record.bboxWidth,
          record.bboxHeight,
          record.imagePath ?? null,
        ],
      );
    }
  });
}

export async function getDetectionsBySession(
  db: SQLite.SQLiteDatabase,
  sessionId: string,
): Promise<DetectionRecord[]> {
  return db.getAllAsync<DetectionRecord>(
    `SELECT id, session_id AS sessionId, timestamp, label,
            class_index AS classIndex, confidence,
            bbox_x AS bboxX, bbox_y AS bboxY,
            bbox_width AS bboxWidth, bbox_height AS bboxHeight,
            image_path AS imagePath
     FROM detection_records
     WHERE session_id = ?
     ORDER BY timestamp ASC;`,
    [sessionId],
  );
}

export async function getAllDetections(
  db: SQLite.SQLiteDatabase,
  limit = 500,
  offset = 0,
): Promise<DetectionRecord[]> {
  return db.getAllAsync<DetectionRecord>(
    `SELECT id, session_id AS sessionId, timestamp, label,
            class_index AS classIndex, confidence,
            bbox_x AS bboxX, bbox_y AS bboxY,
            bbox_width AS bboxWidth, bbox_height AS bboxHeight,
            image_path AS imagePath
     FROM detection_records
     ORDER BY timestamp DESC
     LIMIT ? OFFSET ?;`,
    [limit, offset],
  );
}

export async function deleteDetectionsBySession(
  db: SQLite.SQLiteDatabase,
  sessionId: string,
): Promise<void> {
  await db.runAsync(
    'DELETE FROM detection_records WHERE session_id = ?;',
    [sessionId],
  );
  await db.runAsync('DELETE FROM sessions WHERE session_id = ?;', [sessionId]);
}

export async function getStorageStats(
  db: SQLite.SQLiteDatabase,
): Promise<{ totalRecords: number; totalSessions: number }> {
  const records = await db.getFirstAsync<{ count: number }>(
    'SELECT COUNT(*) AS count FROM detection_records;',
  );
  const sessions = await db.getFirstAsync<{ count: number }>(
    'SELECT COUNT(*) AS count FROM sessions;',
  );
  return {
    totalRecords: records?.count ?? 0,
    totalSessions: sessions?.count ?? 0,
  };
}
