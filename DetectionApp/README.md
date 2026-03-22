# DetectionApp — On-Device Real-World Object Detection

> **MVP, offline-first, fully on-device AI** · React Native (Expo) · YOLO26n · TFLite + CoreML

---

## What's in this directory

| Path | Description |
|---|---|
| `package.json` | All library versions recommended in the blueprint |
| `app.json` | Expo configuration (permissions, plugins, extra env) |
| `src/App.tsx` | Root component — navigation, DB init, security check, OTA |
| `src/hooks/useObjectDetection.ts` | **Part 2** — Inference pipeline hook |
| `src/hooks/useOTAModelUpdater.ts` | **Part 6** — OTA model updater |
| `src/services/storage.ts` | **Part 4** — Encrypted SQLite schema + CRUD |
| `src/services/security.ts` | **Part 5** — Keystore/Keychain, root detection, cert pins |
| `src/store/detectionStore.ts` | Zustand global state |
| `src/screens/CameraScreen.tsx` | Live camera + bounding box overlay |
| `src/screens/HistoryScreen.tsx` | Paginated history + JSON export |
| `src/components/BoundingBoxOverlay.tsx` | Bounding box + label rendering |
| `src/components/DetectionResultCard.tsx` | History list item |
| `src/types/index.ts` | Shared TypeScript types |
| `scripts/export_yolo26n.py` | **Part 3** — YOLO26n → TFLite INT8 + CoreML + ONNX |

---

## Quick start

```bash
cd DetectionApp
npm install
npx expo start
```

### Prepare the model

```bash
pip install ultralytics pillow numpy
python scripts/export_yolo26n.py --output-dir ./models

# Copy the TFLite model into the assets folder
cp models/yolo26n_int8.tflite assets/yolo26n.tflite
```

---

## Part 1 — Project setup

All library versions are pinned in `package.json`.  Key packages:

| Library | Version | Role |
|---|---|---|
| `expo` | `~52.0.0` | Managed workflow, cross-platform |
| `react-native-vision-camera` | `^4.6.0` | Camera + frame processors |
| `react-native-fast-tflite` | `^1.3.0` | JSI TFLite inference |
| `vision-camera-resize-plugin` | `^3.1.0` | GPU-accelerated frame resize |
| `expo-sqlite` | `~15.1.0` | SQLCipher-backed encrypted DB |
| `expo-secure-store` | `~14.0.0` | Keystore / Keychain |
| `react-native-ssl-pinning` | `^1.6.0` | Certificate pinning |
| `react-native-device-info` | `^11.1.0` | Root / jailbreak detection |
| `zustand` | `^5.0.0` | Lightweight state management |

---

## Part 2 — Inference pipeline

See `src/hooks/useObjectDetection.ts`.

```typescript
import { useObjectDetection, FRAME_PROCESSOR_FPS } from './hooks/useObjectDetection';

const { modelState, detections, frameProcessor, warmup } = useObjectDetection({
  confidenceThreshold: 0.5,   // expose as user setting
});
```

**Key design decisions:**
- Inference runs inside a VisionCamera **worklet** — completely off the JS thread.
- `resize()` from `vision-camera-resize-plugin` down-samples each frame to `640×640`
  on the GPU before feeding it to the model.
- YOLO26n is NMS-free: outputs are `[boxes, scores, classes]` — no post-processing needed.
- Frames are throttled to `FRAME_PROCESSOR_FPS` (7 fps) to save battery.
- `warmup()` must be called once after model load to avoid a slow first inference.

---

## Part 3 — Model export

```bash
python scripts/export_yolo26n.py [--output-dir ./models] [--skip-coreml]
```

The script:
1. Downloads `yolo26n.pt` from the Ultralytics CDN (cached locally).
2. Exports INT8-quantised **TFLite** for Android (`yolo26n_int8.tflite`, ~4–6 MB).
3. Exports INT8 **CoreML** for iOS (`yolo26n_int8.mlpackage`, ~4–6 MB).
4. Exports FP16 **ONNX** for debugging (`yolo26n_fp16.onnx`, ~10 MB).
5. Runs a verification inference on each exported format.

> **Note:** CoreML export requires macOS with Xcode installed.  Use `--skip-coreml`
> on Linux/Windows CI runners.

---

## Part 4 — Storage layer

See `src/services/storage.ts`.

### Schema (v1)

```sql
CREATE TABLE sessions (
  session_id  TEXT    PRIMARY KEY,
  started_at  INTEGER NOT NULL,
  ended_at    INTEGER
);

CREATE TABLE detection_records (
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
```

**Encryption:** `expo-sqlite` v15 accepts a `passphrase` option that enables SQLCipher
transparent AES-256 encryption.  The passphrase is stored in the OS secure enclave via
`expo-secure-store` (iOS Keychain / Android Keystore).

**Migrations:** `applyMigrations()` is idempotent and tracked in a `schema_version` table.
Add new migration functions (`migration_v2`, etc.) as the schema evolves.

---

## Part 5 — Security

See `src/services/security.ts`.

### Android Keystore + iOS Keychain

```typescript
import { getModelDecryptionKey } from './services/security';

const key = await getModelDecryptionKey();
// key is stored in Keychain (iOS) / EncryptedSharedPreferences (Android)
// and is never written to disk in plain text.
```

`expo-secure-store` uses `SecItemAdd` / Keychain Services on iOS and
`EncryptedSharedPreferences` backed by Android Keystore on Android.

### Certificate pinning

The OTA endpoint is pinned via `react-native-ssl-pinning` in `useOTAModelUpdater.ts`.
Update `OTA_CERT_PINS` in `security.ts` when your TLS certificate rotates.
Always include at least **one backup pin** to avoid locking users out.

### Root / jailbreak detection

```typescript
import { checkDeviceSecurity } from './services/security';

const { isCompromised, reasons } = await checkDeviceSecurity();
if (isCompromised) { /* warn user or restrict features */ }
```

### Additional hardening (native build files)

| Platform | Action |
|---|---|
| Android | Enable `minifyEnabled true` + ProGuard in `build.gradle` |
| Android | Root detection via `DeviceInfo.isRooted()` |
| iOS | Enable Bitcode stripping in Xcode |
| iOS | Jailbreak detection via DeviceInfo native checks |
| Both | Consider DexGuard (Android) / iXGuard (iOS) for model IP protection |

---

## Part 6 — OTA model updater

See `src/hooks/useOTAModelUpdater.ts`.

```typescript
const { update, checkForUpdate, activeModelPath } = useOTAModelUpdater();
```

**Update flow:**
1. Check `/model-version` endpoint (certificate-pinned).
2. Skip if not on WiFi.
3. Download encrypted `.tflite` to temp cache dir with progress callbacks.
4. Verify SHA-256 checksum.
5. Atomically move file to `documentDirectory` (replaces old model).
6. Delete temp file on failure.

**Environment variables** (set in `.env` or `app.json` → `extra`):

```
MODEL_UPDATE_BASE_URL=https://your-model-server.com
MODEL_UPDATE_CERT_HASH=sha256/<base64-fingerprint>=
```

---

## Part 7 — What to fine-tune first

### Recommended fine-tuning strategy for people, vehicles, and places

#### 1. Start with COCO pre-trained weights (already done)
YOLO26n ships pre-trained on COCO-80 which already covers:
- **People:** `person` class — strong baseline
- **Vehicles:** `car`, `bus`, `truck`, `motorcycle`, `bicycle`, `airplane`, `boat`
- **Places:** partially covered via context objects (benches, traffic lights, etc.)

For many real-world use cases, the COCO weights are sufficient without fine-tuning.

#### 2. Fine-tune on domain-specific data first

**Priority 1 — People detection in your specific environment**
- Dataset: [CrowdHuman](https://www.crowdhuman.org/) (~15K images, dense pedestrian scenes)
- Augmentation: heavy occlusion, varied lighting, crowd scenarios
- Why: COCO under-represents crowded or low-light conditions

**Priority 2 — Vehicles**
- Dataset: [UA-DETRAC](http://detrac-db.rit.albany.edu/) (traffic surveillance) or
  [Open Images v7](https://storage.googleapis.com/openimages/web/index.html) (vehicle subset)
- Filter for: cars, trucks, motorcycles, bicycles in the environments you target

**Priority 3 — Places / landmarks**
- Dataset: [Places365](http://places2.csail.mit.edu/) for scene classification (use as
  MobileViT input after YOLO detection)
- For landmark recognition: [Google Landmarks Dataset v2](https://github.com/cvdfoundation/google-landmark)

#### 3. What specific aspects to fine-tune

| Aspect | What to adjust | Why |
|---|---|---|
| **Head layers only** (transfer learning) | Freeze backbone, train detection head | Fastest convergence, smallest dataset needed (~1K images) |
| **Full fine-tune** | Unfreeze all layers after head converges | Better accuracy for very different domains |
| **Confidence calibration** | Lower threshold to 0.3 for recall-critical scenarios | COCO-trained model may be conservative in your environment |
| **Anchor-free head** | YOLO26n is already anchor-free — no anchor re-tuning needed | Simplifies domain adaptation |
| **Input resolution** | Try `imgsz=320` for faster mobile inference, `imgsz=640` for accuracy | Trade-off on device |

#### 4. Fine-tuning command

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.train(
    data="your_dataset.yaml",   # YOLO format dataset config
    epochs=50,
    imgsz=640,
    batch=16,
    lr0=0.001,
    freeze=10,                  # freeze first 10 layers (backbone)
    augment=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
)
```

#### 5. Dataset format

```yaml
# your_dataset.yaml
path: /path/to/dataset
train: images/train
val: images/val

nc: 3
names: ['person', 'vehicle', 'place']
```

#### 6. Evaluation metrics to watch

- **mAP@0.5** — primary accuracy metric
- **Inference latency on device** — benchmark on actual mobile hardware, not desktop
- **False positive rate** — critical for user trust; tune confidence threshold per class

---

## Blueprint change notes (as of March 2026)

The following items in the original blueprint may have evolved:

| Item | Status | Note |
|---|---|---|
| `YOLO26n` | ✅ Released Jan 2026 | Available via `ultralytics` package |
| `react-native-fast-tflite` | ✅ Current | JSI architecture stable in RN 0.76+ |
| `react-native-vision-camera` v4 | ✅ Current | Frame processor API stable |
| `expo-sqlite` v15 passphrase support | ✅ Current | SQLCipher integration available |
| `expo-secure-store` Keychain support | ✅ Current | Stable on both platforms |
| MobileViT weights (`apple/mobilevit-xx-small`) | ✅ Available on HuggingFace | For classification after detection |
| MobileNetV4 via `timm` | ✅ Available | For custom backbone fine-tuning |

---

## Storage estimates (blueprint §Part 5)

| Item | Size per unit | 1,000 scans |
|---|---|---|
| Annotated PNG image | ~200–500 KB | ~350 MB |
| SQLite detection record | ~500 bytes | ~500 KB |
| Session metadata | ~100 bytes | ~100 KB |
| **Total** | | **~350 MB** |

Add a storage management screen (show used space, allow bulk delete by session).
