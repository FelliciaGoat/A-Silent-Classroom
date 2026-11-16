"""
main_pipeline.py - Silent Classroom: one frame-by-frame loop.

Combines
  * AttentionTracker  - MediaPipe FaceMesh -> face boxes + head-pose attention
  * GestureDetector   - YOLO pose (raised arms) + MediaPipe hands (index finger / open palm)
  * EmotionAnalyzer   - DeepFace facial expression (confusion heuristic)

Speed: FaceMesh runs every frame (cheap); pose/hand detection runs every
`gesture_every` frames and DeepFace every `emotion_every` frames. Results in
between come from a cache, so the annotated output never stalls.

Usage
  python main_pipeline.py --source 0                       # webcam
  python main_pipeline.py --source class.mp4 --output out.mp4
  python main_pipeline.py --image photo.jpg --output out.jpg
"""
import argparse
import os
import time
import urllib.request
from collections import Counter

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from ultralytics import YOLO
from deepface import DeepFace

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_MODEL_PATH = "hand_landmarker.task"
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_MODEL_PATH = "face_landmarker.task"
POSE_WEIGHTS = "yolo11s-pose.pt"

GESTURE_INDEX = "index_finger"
GESTURE_PALM = "open_palm"
GESTURE_OTHER = "other"

GREEN, ORANGE, RED, YELLOW, WHITE = (0, 200, 0), (0, 165, 255), (0, 0, 255), (0, 255, 255), (255, 255, 255)


def _clip_box(x1, y1, x2, y2, w, h):
    return max(0, int(x1)), max(0, int(y1)), min(w - 1, int(x2)), min(h - 1, int(y2))


def _center(box):
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _inside(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _label(img, text, x, y, color, scale=0.55):
    """Text with a dark backing so it stays readable on any background."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    y = max(th + 4, y)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 3), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


# --------------------------------------------------------------------------
class AttentionTracker:
    """Face boxes and a head-pose attention estimate from FaceMesh landmarks."""

    def __init__(self, max_faces=10, yaw_range=(0.30, 0.70), pitch_range=(0.40, 0.75),
                 face_model=FACE_MODEL_PATH):
        if not os.path.exists(face_model):
            urllib.request.urlretrieve(FACE_MODEL_URL, face_model)
        self.face_mesh = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=face_model),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=max_faces,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
            )
        )
        self.yaw_range = yaw_range
        self.pitch_range = pitch_range

    def process(self, rgb):
        result = self.face_mesh.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                data=np.ascontiguousarray(rgb)))
        if not result.face_landmarks:
            return []
        h, w = rgb.shape[:2]
        faces = []
        for pts in result.face_landmarks:
            xs = [p.x for p in pts]
            ys = [p.y for p in pts]
            box = _clip_box(min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h, w, h)
            # nose position between the cheeks (yaw) and forehead/chin (pitch)
            yaw = (pts[1].x - pts[234].x) / max(pts[454].x - pts[234].x, 1e-6)
            pitch = (pts[1].y - pts[10].y) / max(pts[152].y - pts[10].y, 1e-6)
            attentive = (self.yaw_range[0] <= yaw <= self.yaw_range[1]
                         and self.pitch_range[0] <= pitch <= self.pitch_range[1])
            faces.append({"box": box, "yaw": float(yaw), "pitch": float(pitch), "attentive": bool(attentive)})
        return faces

    def close(self):
        self.face_mesh.close()


# --------------------------------------------------------------------------
class GestureDetector:
    """Raised arms (YOLO pose) and hand gestures (MediaPipe hands)."""

    def __init__(self, pose_weights=POSE_WEIGHTS, hand_model=HAND_MODEL_PATH,
                 person_conf=0.40, imgsz=640, max_hands=30):
        if not os.path.exists(hand_model):
            urllib.request.urlretrieve(HAND_MODEL_URL, hand_model)
        self.pose = YOLO(pose_weights)
        self.person_conf = person_conf
        self.imgsz = imgsz
        self.hands = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=hand_model),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=max_hands,
                min_hand_detection_confidence=0.35,
                min_hand_presence_confidence=0.35,
                min_tracking_confidence=0.35,
            )
        )

    @staticmethod
    def classify_hand(lm):
        index_up = lm[8].y < lm[6].y
        middle_up = lm[12].y < lm[10].y
        ring_up = lm[16].y < lm[14].y
        pinky_up = lm[20].y < lm[18].y
        if index_up and not (middle_up or ring_up or pinky_up):
            return GESTURE_INDEX
        if index_up and middle_up and ring_up and pinky_up:
            return GESTURE_PALM
        return GESTURE_OTHER

    def detect(self, bgr, rgb):
        h, w = bgr.shape[:2]
        people = []
        res = self.pose.predict(bgr, imgsz=self.imgsz, conf=self.person_conf, iou=0.45,
                                classes=[0], max_det=100, verbose=False)[0]
        if res.boxes is not None and res.keypoints is not None and len(res.boxes):
            boxes = res.boxes.xyxy.cpu().numpy()
            kps = res.keypoints.xy.cpu().numpy()
            kconf = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else None
            for i, (box, kp) in enumerate(zip(boxes, kps)):
                ok = lambda j: kconf is None or kconf[i][j] >= 0.20
                # COCO: 5/6 shoulders, 9/10 wrists - wrist above shoulder = raised arm
                raised = ((ok(5) and ok(9) and kp[9][1] < kp[5][1] - 10)
                          or (ok(6) and ok(10) and kp[10][1] < kp[6][1] - 10))
                people.append({"box": _clip_box(*box, w, h), "raised": bool(raised), "hand": None})

        hands = []
        out = self.hands.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        for lm in out.hand_landmarks or []:
            gesture = self.classify_hand(lm)
            centre = (lm[9].x * w, lm[9].y * h)
            hands.append({"gesture": gesture, "centre": centre, "points": [(p.x * w, p.y * h) for p in lm]})
            for person in people:  # attach hand to the person whose box contains it
                if _inside(centre, person["box"]) and gesture in (GESTURE_INDEX, GESTURE_PALM):
                    person["hand"] = gesture
                    break
        return {"people": people, "hands": hands}

    def close(self):
        self.hands.close()


# --------------------------------------------------------------------------
class EmotionAnalyzer:
    """DeepFace emotion on face crops; confusion = surprise + fear + sad."""

    CONFUSION_EMOTIONS = ("surprise", "fear", "sad")

    def __init__(self, confusion_threshold=0.45, pad=0.15, min_face=40):
        self.confusion_threshold = confusion_threshold
        self.pad = pad
        self.min_face = min_face
        try:  # warm-up so the first real frame is not slow
            DeepFace.analyze(np.zeros((48, 48, 3), np.uint8), actions=["emotion"],
                             detector_backend="skip", enforce_detection=False, silent=True)
        except Exception:
            pass

    def analyze(self, bgr, faces):
        h, w = bgr.shape[:2]
        out = []
        for f in faces:
            x1, y1, x2, y2 = f["box"]
            px, py = int((x2 - x1) * self.pad), int((y2 - y1) * self.pad)
            x1, y1, x2, y2 = _clip_box(x1 - px, y1 - py, x2 + px, y2 + py, w, h)
            if x2 - x1 < self.min_face or y2 - y1 < self.min_face:
                continue
            try:
                r = DeepFace.analyze(bgr[y1:y2, x1:x2], actions=["emotion"],
                                     detector_backend="skip", enforce_detection=False, silent=True)
                r = r[0] if isinstance(r, list) else r
            except Exception:
                continue
            emo = {k: float(v) for k, v in r["emotion"].items()}
            confusion = min(1.0, sum(emo.get(k, 0.0) for k in self.CONFUSION_EMOTIONS) / 100.0)
            out.append({"center": _center(f["box"]), "width": f["box"][2] - f["box"][0],
                        "emotion": r["dominant_emotion"], "scores": emo,
                        "confusion": confusion, "confused": confusion > self.confusion_threshold})
        return out


# --------------------------------------------------------------------------
class ClassroomPipeline:
    def __init__(self, gesture_every=2, emotion_every=10, max_faces=10, draw_mesh=False):
        self.gesture_every = max(1, gesture_every)
        self.emotion_every = max(1, emotion_every)
        self.draw_mesh = draw_mesh
        self.attention = AttentionTracker(max_faces=max_faces)
        self.gestures = GestureDetector()
        self.emotions = EmotionAnalyzer()
        self.frame_idx = 0
        self._gesture_cache = {"people": [], "hands": []}
        self._emotion_cache = []
        self._fps = 0.0

    @staticmethod
    def _match_emotion(face, cache):
        c = _center(face["box"])
        best, best_d = None, None
        for e in cache:
            d = np.hypot(c[0] - e["center"][0], c[1] - e["center"][1])
            if best_d is None or d < best_d:
                best, best_d = e, d
        w = face["box"][2] - face["box"][0]
        return best if best is not None and best_d < max(w, 1) else None

    def process_frame(self, bgr, force=False):
        """Return (annotated_bgr, report). force=True runs every model (single images / API calls)."""
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        faces = self.attention.process(rgb)

        if force or self.frame_idx % self.gesture_every == 0:
            self._gesture_cache = self.gestures.detect(bgr, rgb)
        if faces and (force or self.frame_idx % self.emotion_every == 0):
            self._emotion_cache = self.emotions.analyze(bgr, faces)
        elif not faces:
            self._emotion_cache = []

        people, hands = self._gesture_cache["people"], self._gesture_cache["hands"]
        out = bgr.copy()
        report = self._annotate(out, faces, people, hands)

        dt = time.perf_counter() - t0
        self._fps = 0.9 * self._fps + 0.1 / max(dt, 1e-6) if self._fps else 1.0 / max(dt, 1e-6)
        report["fps"] = round(self._fps, 1)
        _label(out, f"FPS {self._fps:.1f}", 10, out.shape[0] - 10, WHITE, 0.55)
        self.frame_idx += 1
        return out, report

    def _annotate(self, out, faces, people, hands):
        for person in people:  # body boxes
            x1, y1, x2, y2 = person["box"]
            color = GREEN if person["raised"] or person["hand"] == GESTURE_INDEX else (200, 150, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            if person["raised"]:
                _label(out, "HAND RAISED", x1, y1 - 6, GREEN)
        for hnd in hands:  # hand landmarks
            for px, py in hnd["points"]:
                cv2.circle(out, (int(px), int(py)), 2, YELLOW, -1)

        faces_info, alerts = [], 0
        for face in faces:
            x1, y1, x2, y2 = face["box"]
            emo = self._match_emotion(face, self._emotion_cache)
            person = next((p for p in people if _inside(_center(face["box"]), p["box"])), None)
            reasons = []
            if person and person["raised"]:
                reasons.append("hand raised")
            if person and person["hand"] == GESTURE_INDEX:
                reasons.append("index finger")
            if emo and emo["confused"]:
                reasons.append("confused face")
            doubt = bool(reasons)
            alerts += doubt

            color = RED if doubt else (GREEN if face["attentive"] else ORANGE)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            tag = (emo["emotion"].upper() if emo else "...") + ("" if face["attentive"] else " | LOOKING AWAY")
            _label(out, tag, x1, y2 + 18, color)
            if doubt:
                _label(out, "DOUBT: " + ", ".join(reasons), x1, y1 - 6, RED)
            faces_info.append({"box": list(face["box"]), "attentive": face["attentive"],
                               "emotion": emo["emotion"] if emo else None,
                               "confusion": round(emo["confusion"], 3) if emo else None,
                               "confused": bool(emo and emo["confused"]),
                               "doubt": doubt, "reasons": reasons})

        # raised arms whose face is not visible still count as a doubt
        alerts += sum(1 for p in people if p["raised"] and not any(
            _inside(_center(f["box"]), p["box"]) for f in faces))

        counts = Counter(h["gesture"] for h in hands)
        if alerts:
            cv2.rectangle(out, (0, 0), (out.shape[1], 34), RED, -1)
            cv2.putText(out, f"DOUBT ALERT: {alerts}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
        y = 52 if alerts else 22
        for line in (f"People {len(people)}  Faces {len(faces)}  Attentive {sum(f['attentive'] for f in faces)}",
                     f"Raised {sum(p['raised'] for p in people)}  Index {counts[GESTURE_INDEX]}  Palm {counts[GESTURE_PALM]}"):
            _label(out, line, 10, y, WHITE, 0.55)
            y += 24

        return {
            "people": len(people), "faces": len(faces),
            "attentive_faces": sum(f["attentive"] for f in faces),
            "raised_hands": sum(p["raised"] for p in people),
            "index_finger": counts[GESTURE_INDEX], "open_palm": counts[GESTURE_PALM],
            "confused_faces": sum(1 for f in faces_info if f["confused"]),
            "doubt_alerts": alerts, "faces_detail": faces_info,
        }

    def close(self):
        self.attention.close()
        self.gestures.close()


# --------------------------------------------------------------------------
def run_stream(pipeline, source, output=None, show=False, max_frames=None):
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) if output else None

    n, alerts, t0 = 0, 0, time.time()
    try:
        while max_frames is None or n < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            annotated, report = pipeline.process_frame(frame)
            alerts += report["doubt_alerts"] > 0
            if writer:
                writer.write(annotated)
            if show:
                cv2.imshow("Silent Classroom", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            n += 1
    finally:
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()
    elapsed = max(time.time() - t0, 1e-6)
    print(f"{n} frames in {elapsed:.1f}s ({n / elapsed:.1f} FPS), frames with a doubt alert: {alerts}")


def main():
    ap = argparse.ArgumentParser(description="Silent Classroom pipeline")
    ap.add_argument("--source", default="0", help="webcam index or video path")
    ap.add_argument("--image", help="analyze a single image instead")
    ap.add_argument("--output", help="write annotated video/image here")
    ap.add_argument("--emotion-every", type=int, default=10)
    ap.add_argument("--gesture-every", type=int, default=2)
    ap.add_argument("--show", action="store_true", help="live preview window (q to quit)")
    args = ap.parse_args()

    pipe = ClassroomPipeline(gesture_every=args.gesture_every, emotion_every=args.emotion_every)
    try:
        if args.image:
            img = cv2.imread(args.image)
            if img is None:
                raise RuntimeError(f"Could not read image: {args.image}")
            annotated, report = pipe.process_frame(img, force=True)
            print({k: v for k, v in report.items() if k != "faces_detail"})
            if args.output:
                cv2.imwrite(args.output, annotated)
        else:
            run_stream(pipe, args.source, args.output, args.show)
    finally:
        pipe.close()


if __name__ == "__main__":
    main()
