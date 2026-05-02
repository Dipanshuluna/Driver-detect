"""
Utility functions for Driver Drowsiness Detection.

EAR formula:
    EAR = (distance(p2, p6) + distance(p3, p5)) / (2 * distance(p1, p4))

MAR follows the same idea for the mouth. Open-mouth/yawn frames produce a
higher vertical-to-horizontal mouth ratio.
"""

import csv
import math
import time
import wave
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import distance


EAR_THRESHOLD = 0.25
EAR_SMOOTHING_WINDOW = 5
DROWSY_FRAME_LIMIT = 20

MAR_THRESHOLD = 0.65
YAWN_FRAME_LIMIT = 15

HEAD_TILT_THRESHOLD_DEGREES = 15.0
HEAD_TILT_FRAME_LIMIT = 15

LOW_LIGHT_THRESHOLD = 85

# MediaPipe Face Mesh landmark indices around each eye.
# The six points are ordered to match the standard EAR equation.
EYE_LANDMARKS = {
    "left": [33, 160, 158, 133, 153, 144],
    "right": [362, 385, 387, 263, 373, 380],
}

# Six mouth points: left corner, upper inner lip points, right corner,
# lower inner lip points. The ordering supports the MAR equation below.
MOUTH_LANDMARKS = [61, 81, 13, 291, 312, 14]

# A compact pair of stable facial landmarks for roll/head tilt estimation.
HEAD_TILT_LANDMARKS = {
    "left_eye_outer": 33,
    "right_eye_outer": 263,
}

NORMAL_COLOR = (0, 255, 0)
WARNING_COLOR = (0, 180, 255)
DROWSY_COLOR = (0, 0, 255)
INFO_COLOR = (255, 255, 0)
POINT_COLOR = (255, 0, 255)
TEXT_SHADOW_COLOR = (0, 0, 0)


class MovingAverage:
    """Small fixed-window moving average for stabilizing noisy signals."""

    def __init__(self, window_size: int):
        self.values = deque(maxlen=window_size)

    def update(self, value: float) -> float:
        self.values.append(value)
        return self.average

    @property
    def average(self) -> float:
        if not self.values:
            return 0.0
        return float(sum(self.values) / len(self.values))

    def clear(self) -> None:
        self.values.clear()


class FPSCounter:
    """Tracks smoothed frames-per-second for display."""

    def __init__(self, smoothing: float = 0.9):
        self.smoothing = smoothing
        self.previous_time = time.perf_counter()
        self.fps = 0.0

    def update(self) -> float:
        current_time = time.perf_counter()
        elapsed = current_time - self.previous_time
        self.previous_time = current_time

        if elapsed <= 0:
            return self.fps

        instant_fps = 1.0 / elapsed
        if self.fps == 0:
            self.fps = instant_fps
        else:
            self.fps = (self.smoothing * self.fps) + ((1 - self.smoothing) * instant_fps)

        return self.fps


class EventLogger:
    """Append drowsiness-related events to a CSV file."""

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.last_event_times = {}
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def _ensure_header(self) -> None:
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            return

        with self.csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "timestamp",
                    "event_type",
                    "ear",
                    "mar",
                    "head_tilt_degrees",
                    "attention_score",
                    "details",
                ]
            )

    def log_event(
        self,
        event_type: str,
        ear: float,
        mar: float,
        head_tilt: float,
        attention_score: int,
        details: str = "",
        cooldown_seconds: float = 3.0,
    ) -> None:
        now = time.time()
        last_time = self.last_event_times.get(event_type, 0)
        if now - last_time < cooldown_seconds:
            return

        self.last_event_times[event_type] = now
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))

        with self.csv_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    timestamp,
                    event_type,
                    f"{ear:.4f}",
                    f"{mar:.4f}",
                    f"{head_tilt:.2f}",
                    attention_score,
                    details,
                ]
            )


def eye_aspect_ratio(eye_points: np.ndarray) -> float:
    """Calculate Eye Aspect Ratio from six eye landmark points."""
    vertical_1 = distance.euclidean(eye_points[1], eye_points[5])
    vertical_2 = distance.euclidean(eye_points[2], eye_points[4])
    horizontal = distance.euclidean(eye_points[0], eye_points[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def mean_eye_aspect_ratio(left_eye: np.ndarray, right_eye: np.ndarray) -> float:
    """Return the average EAR across both eyes."""
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    return (left_ear + right_ear) / 2.0


def mouth_aspect_ratio(mouth_points: np.ndarray) -> float:
    """Calculate Mouth Aspect Ratio from six mouth landmarks."""
    vertical_1 = distance.euclidean(mouth_points[1], mouth_points[5])
    vertical_2 = distance.euclidean(mouth_points[2], mouth_points[4])
    horizontal = distance.euclidean(mouth_points[0], mouth_points[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def calculate_head_tilt(landmark_points: list) -> float:
    """Estimate head roll angle in degrees from outer eye landmarks."""
    left_eye = landmark_points[HEAD_TILT_LANDMARKS["left_eye_outer"]]
    right_eye = landmark_points[HEAD_TILT_LANDMARKS["right_eye_outer"]]
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    return math.degrees(math.atan2(dy, dx))


def calculate_attention_score(
    ear: float,
    mar: float,
    head_tilt_degrees: float,
    drowsy_counter: int,
    yawn_counter: int,
    tilt_counter: int,
) -> int:
    """
    Estimate attention from 0 to 100.

    The score combines closed-eye duration, yawning, head tilt, and how far the
    live metrics are past their thresholds.
    """
    score = 100.0

    score -= min(45.0, (drowsy_counter / DROWSY_FRAME_LIMIT) * 45.0)
    score -= min(25.0, (yawn_counter / YAWN_FRAME_LIMIT) * 25.0)
    score -= min(20.0, (tilt_counter / HEAD_TILT_FRAME_LIMIT) * 20.0)

    if ear < EAR_THRESHOLD:
        score -= min(15.0, (EAR_THRESHOLD - ear) * 100.0)
    if mar > MAR_THRESHOLD:
        score -= min(10.0, (mar - MAR_THRESHOLD) * 35.0)

    excess_tilt = max(0.0, abs(head_tilt_degrees) - HEAD_TILT_THRESHOLD_DEGREES)
    score -= min(10.0, excess_tilt * 0.5)

    return int(max(0, min(100, round(score))))


def enhance_low_light(frame):
    """
    Improve frame contrast in dim scenes using CLAHE on the luminance channel.

    This helps MediaPipe receive a clearer face image without changing the
    original BGR frame layout expected by OpenCV.
    """
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
    average_brightness = float(np.mean(y_channel))

    if average_brightness >= LOW_LIGHT_THRESHOLD:
        return frame, False, average_brightness

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_y = clahe.apply(y_channel)
    enhanced = cv2.merge((enhanced_y, cr_channel, cb_channel))
    enhanced_frame = cv2.cvtColor(enhanced, cv2.COLOR_YCrCb2BGR)
    return enhanced_frame, True, average_brightness


def get_landmark_points(face_landmarks, frame_width: int, frame_height: int) -> list:
    """Convert normalized MediaPipe landmarks to pixel coordinates."""
    points = []

    for landmark in face_landmarks.landmark:
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        points.append((x, y))

    return points


def get_points(landmark_points: list, indices: list) -> np.ndarray:
    """Extract landmark coordinates by index."""
    return np.array([landmark_points[index] for index in indices], dtype=np.float32)


def get_eye_points(landmark_points: list, eye_indices: list) -> np.ndarray:
    """Extract eye landmark coordinates by index."""
    return get_points(landmark_points, eye_indices)


def get_mouth_points(landmark_points: list) -> np.ndarray:
    """Extract mouth landmark coordinates."""
    return get_points(landmark_points, MOUTH_LANDMARKS)


def draw_landmarks(frame, points: np.ndarray, color=POINT_COLOR) -> None:
    """Draw small circles on selected landmarks."""
    for point in points.astype(int):
        cv2.circle(frame, tuple(point), 2, color, -1)


def draw_eye_landmarks(frame, left_eye: np.ndarray, right_eye: np.ndarray) -> None:
    """Draw small circles on detected eye landmarks."""
    draw_landmarks(frame, np.vstack((left_eye, right_eye)))


def draw_overlay_line(
    frame,
    text: str,
    position: tuple,
    color=INFO_COLOR,
    scale: float = 0.65,
    thickness: int = 2,
) -> None:
    """Draw readable text with a shadow for varied lighting conditions."""
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        TEXT_SHADOW_COLOR,
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def ensure_alarm_file(path: Path, duration_seconds: float = 0.8) -> None:
    """
    Create a simple WAV alarm if it does not exist.

    This keeps the project self-contained while main.py handles playback.
    """
    if path.exists():
        return

    sample_rate = 44100
    amplitude = 16000
    frames = []
    total_samples = int(sample_rate * duration_seconds)

    for sample_index in range(total_samples):
        t = sample_index / sample_rate
        # Alternating tones make the generated alarm easier to notice.
        frequency = 880 if int(t * 8) % 2 == 0 else 660
        value = int(amplitude * math.sin(2 * math.pi * frequency * t))
        frames.append(value)

    audio = np.array(frames, dtype=np.int16)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())
