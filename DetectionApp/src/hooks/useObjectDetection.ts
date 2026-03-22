/**
 * useObjectDetection — React hook that wraps react-native-fast-tflite and
 * react-native-vision-camera for real-time, on-device YOLO26n inference.
 *
 * Architecture notes (from blueprint Part 3):
 *  - Inference runs off the JS thread inside a VisionCamera worklet.
 *  - Frames are throttled to 5–10 fps to preserve battery.
 *  - YOLO26n eliminates NMS post-processing — outputs are boxes, scores,
 *    classes directly.
 *  - Confidence threshold defaults to 0.5; expose as a user-tunable setting.
 */

import { useEffect, useRef, useCallback, useState } from 'react';
import {
  useTensorflowModel,
  type TensorflowModel,
} from 'react-native-fast-tflite';
import { useFrameProcessor } from 'react-native-vision-camera';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { runOnJS } from 'react-native-reanimated';

import type { Detection, BoundingBox, ModelState } from '../types';

/** COCO-80 label set — replace with your custom labels if fine-tuned. */
const COCO_LABELS: string[] = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
  'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
  'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
  'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
  'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
  'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
  'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
  'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
  'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
  'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
  'hair drier', 'toothbrush',
];

/** Input resolution expected by YOLO26n. */
const MODEL_INPUT_SIZE = 640;

/** Default confidence threshold — expose in app settings to let users adjust. */
const DEFAULT_CONFIDENCE_THRESHOLD = 0.5;

/** Default frame rate for the frame processor (fps). Tunable per device. */
const DEFAULT_FRAME_PROCESSOR_FPS = 7;

export interface UseObjectDetectionOptions {
  confidenceThreshold?: number;
  labels?: string[];
  /** Frames per second to process. Lower values save battery. Default: 7. */
  frameProcessorFps?: number;
}

export interface UseObjectDetectionResult {
  modelState: ModelState;
  detections: Detection[];
  frameProcessor: ReturnType<typeof useFrameProcessor>;
  frameProcessorFps: number;
  warmup: () => void;
}

/**
 * Parse raw YOLO26n tensor outputs into structured Detection objects.
 *
 * YOLO26n (NMS-free) output layout:
 *   outputs[0] — flat bounding boxes  [numDetections × 4] (cx, cy, w, h normalised)
 *   outputs[1] — confidence scores    [numDetections]
 *   outputs[2] — class indices        [numDetections]
 *
 * All coordinates are normalised to [0, 1].
 */
function parseOutputs(
  boxes: Float32Array,
  scores: Float32Array,
  classes: Float32Array,
  confidenceThreshold: number,
  labels: string[],
): Detection[] {
  const numDetections = scores.length;
  const results: Detection[] = [];

  for (let i = 0; i < numDetections; i++) {
    const confidence = scores[i];
    if (confidence < confidenceThreshold) continue;

    const classIndex = Math.round(classes[i]);
    const label = labels[classIndex] ?? `class_${classIndex}`;

    // YOLO outputs are centre-x, centre-y, width, height (normalised).
    const cx = boxes[i * 4];
    const cy = boxes[i * 4 + 1];
    const w  = boxes[i * 4 + 2];
    const h  = boxes[i * 4 + 3];

    const boundingBox: BoundingBox = {
      x: cx - w / 2,
      y: cy - h / 2,
      width: w,
      height: h,
    };

    results.push({ classIndex, label, confidence, boundingBox });
  }

  return results;
}

export function useObjectDetection(
  options: UseObjectDetectionOptions = {},
): UseObjectDetectionResult {
  const {
    confidenceThreshold = DEFAULT_CONFIDENCE_THRESHOLD,
    labels = COCO_LABELS,
    frameProcessorFps = DEFAULT_FRAME_PROCESSOR_FPS,
  } = options;

  const [modelState, setModelState] = useState<ModelState>('loading');
  const [detections, setDetections] = useState<Detection[]>([]);

  // Load the bundled YOLO26n TFLite model from the assets folder.
  const tfliteModel = useTensorflowModel(
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    require('../../assets/yolo26n.tflite'),
  );

  const model: TensorflowModel | undefined =
    tfliteModel.state === 'loaded' ? tfliteModel.model : undefined;

  useEffect(() => {
    if (tfliteModel.state === 'loaded') setModelState('loaded');
    if (tfliteModel.state === 'error')  setModelState('error');
  }, [tfliteModel.state]);

  /** Warm up the model with a blank pass so the first real inference is fast. */
  const warmup = useCallback(() => {
    if (!model) return;
    const blank = new Uint8Array(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3);
    model.runSync([blank]);
  }, [model]);

  const { resize } = useResizePlugin();

  /** Ref guards the JS-thread callback from being recreated on each render. */
  const detectionsRef = useRef(detections);
  const onDetections = useCallback((next: Detection[]) => {
    detectionsRef.current = next;
    setDetections(next);
  }, []);

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      if (!model) return;

      // Down-sample the raw camera frame to MODEL_INPUT_SIZE × MODEL_INPUT_SIZE.
      const resized = resize(frame, {
        scale: { width: MODEL_INPUT_SIZE, height: MODEL_INPUT_SIZE },
        pixelFormat: 'rgb',
        dataType: 'uint8',
      });

      // YOLO26n is NMS-free: it returns boxes, scores, and classes directly.
      const outputs = model.runSync([resized]);
      const boxes   = outputs[0] as Float32Array;
      const scores  = outputs[1] as Float32Array;
      const classes = outputs[2] as Float32Array;

      const parsed = parseOutputs(
        boxes,
        scores,
        classes,
        confidenceThreshold,
        labels,
      );

      // Bridge back to the JS thread to update React state.
      runOnJS(onDetections)(parsed);
    },
    [model, confidenceThreshold, labels, resize, onDetections],
  );

  return { modelState, detections, frameProcessor, frameProcessorFps, warmup };
}

/** Default COCO label list — exported for use in other parts of the app. */
export { COCO_LABELS, DEFAULT_FRAME_PROCESSOR_FPS };
