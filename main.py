"""
Advanced Driver Drowsiness Detection System

Install dependencies:
    pip install opencv-python mediapipe scipy numpy pygame

Run:
    python main.py

Features:
    - Webcam capture with OpenCV
    - MediaPipe Face Mesh landmarks
    - Smoothed Eye Aspect Ratio (EAR)
    - Yawn detection with Mouth Aspect Ratio (MAR)
    - Head tilt detection
    - FPS counter
    - Low-light contrast enhancement
    - CSV event logging
    - Attention score from 0 to 100

Press 'q' to exit.
"""

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import pygame

from utils import (
    DROWSY_COLOR,
    DROWSY_FRAME_LIMIT,
    EAR_SMOOTHING_WINDOW,
    EAR_THRESHOLD,
    EYE_LANDMARKS,
    FPSCounter,
    EventLogger,
    HEAD_TILT_FRAME_LIMIT,
    HEAD_TILT_THRESHOLD_DEGREES,
    INFO_COLOR,
    MAR_THRESHOLD,
    NORMAL_COLOR,
    WARNING_COLOR,
    YAWN_FRAME_LIMIT,
    MovingAverage,
    calculate_attention_score,
    calculate_head_tilt,
    draw_eye_landmarks,
    draw_landmarks,
    draw_overlay_line,
    enhance_low_light,
    ensure_alarm_file,
    get_eye_points,
    get_landmark_points,
    get_mouth_points,
    mean_eye_aspect_ratio,
    mouth_aspect_ratio,
)


BASE_DIR = Path(__file__).resolve().parent
ALARM_FILE = BASE_DIR / "alarm.wav"
EVENT_LOG_FILE = BASE_DIR / "drowsiness_events.csv"


@dataclass
class DetectionState:
    """Mutable state kept across video frames."""

    drowsy_counter: int = 0
    yawn_counter: int = 0
    tilt_counter: int = 0
    alarm_playing: bool = False


def play_alarm(alarm_path: Path) -> None:
    """Play the alarm sound once in a background thread."""
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()

        if pygame.mixer.music.get_busy():
            return

        pygame.mixer.music.load(str(alarm_path))
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    except Exception as exc:
        # Keep the video loop alive even if the local audio device is unavailable.
        print(f"Could not play alarm sound: {exc}")


def start_alarm_if_needed(alarm_path: Path, state: DetectionState) -> None:
    """Start a non-blocking alarm if one is not already playing."""
    if state.alarm_playing:
        return

    state.alarm_playing = True

    def alarm_worker() -> None:
        play_alarm(alarm_path)
        state.alarm_playing = False

    threading.Thread(target=alarm_worker, daemon=True).start()


def update_detection_counters(
    state: DetectionState,
    ear: float,
    mar: float,
    head_tilt: float,
) -> None:
    """Update consecutive-frame counters for each fatigue signal."""
    if ear < EAR_THRESHOLD:
        state.drowsy_counter += 1
    else:
        state.drowsy_counter = 0

    if mar > MAR_THRESHOLD:
        state.yawn_counter += 1
    else:
        state.yawn_counter = 0

    if abs(head_tilt) > HEAD_TILT_THRESHOLD_DEGREES:
        state.tilt_counter += 1
    else:
        state.tilt_counter = 0


def build_status_messages(state: DetectionState) -> tuple:
    """Return status text, color, and triggered event names."""
    events = []

    if state.drowsy_counter >= DROWSY_FRAME_LIMIT:
        events.append("DROWSY")
    if state.yawn_counter >= YAWN_FRAME_LIMIT:
        events.append("YAWNING")
    if state.tilt_counter >= HEAD_TILT_FRAME_LIMIT:
        events.append("HEAD TILT")

    if "DROWSY" in events:
        return "DROWSY!", DROWSY_COLOR, events
    if events:
        return " / ".join(events), WARNING_COLOR, events
    return "Awake", NORMAL_COLOR, events


def log_triggered_events(
    logger: EventLogger,
    events: list,
    ear: float,
    mar: float,
    head_tilt: float,
    attention_score: int,
) -> None:
    """Write detected attention events to CSV with a per-event cooldown."""
    for event in events:
        logger.log_event(
            event_type=event,
            ear=ear,
            mar=mar,
            head_tilt=head_tilt,
            attention_score=attention_score,
            details="Consecutive frame threshold reached",
        )


def draw_dashboard(
    frame,
    status_text: str,
    status_color: tuple,
    ear: Optional[float],
    mar: Optional[float],
    head_tilt: Optional[float],
    attention_score: Optional[int],
    fps: float,
    low_light_enabled: bool,
    brightness: float,
) -> None:
    """Draw all on-screen telemetry in one compact overlay."""
    draw_overlay_line(frame, f"FPS: {fps:.1f}", (30, 30), INFO_COLOR)

    if ear is None:
        draw_overlay_line(frame, "EAR: --", (30, 60), INFO_COLOR)
        draw_overlay_line(frame, "MAR: --", (30, 90), INFO_COLOR)
        draw_overlay_line(frame, "Head tilt: --", (30, 120), INFO_COLOR)
        draw_overlay_line(frame, "Attention: --", (30, 150), INFO_COLOR)
    else:
        draw_overlay_line(frame, f"EAR: {ear:.2f}", (30, 60), INFO_COLOR)
        draw_overlay_line(frame, f"MAR: {mar:.2f}", (30, 90), INFO_COLOR)
        draw_overlay_line(frame, f"Head tilt: {head_tilt:.1f} deg", (30, 120), INFO_COLOR)
        draw_overlay_line(frame, f"Attention: {attention_score}/100", (30, 150), INFO_COLOR)

    light_status = "Low light enhanced" if low_light_enabled else "Light OK"
    draw_overlay_line(
        frame,
        f"{light_status} ({brightness:.0f})",
        (30, 180),
        WARNING_COLOR if low_light_enabled else NORMAL_COLOR,
    )
    draw_overlay_line(frame, status_text, (30, 220), status_color, scale=0.9, thickness=2)

    if status_text == "DROWSY!":
        draw_overlay_line(frame, "DROWSY!", (30, 285), DROWSY_COLOR, scale=1.8, thickness=4)


def process_frame(
    frame,
    face_mesh,
    state: DetectionState,
    ear_average: MovingAverage,
    fps_counter: FPSCounter,
    logger: EventLogger,
):
    """
    Detect face landmarks, calculate fatigue metrics, draw overlay, and log events.

    EAR is smoothed with a moving average before applying the drowsiness
    threshold, which reduces false alarms from single-frame landmark jitter.
    """
    fps = fps_counter.update()
    frame = cv2.flip(frame, 1)
    display_frame, low_light_enabled, brightness = enhance_low_light(frame)

    height, width = display_frame.shape[:2]
    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = face_mesh.process(rgb_frame)
    rgb_frame.flags.writeable = True

    status_text = "No face detected"
    status_color = WARNING_COLOR
    ear = None
    mar = None
    head_tilt = None
    attention_score = None

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmark_points = get_landmark_points(face_landmarks, width, height)

        left_eye = get_eye_points(landmark_points, EYE_LANDMARKS["left"])
        right_eye = get_eye_points(landmark_points, EYE_LANDMARKS["right"])
        mouth_points = get_mouth_points(landmark_points)

        raw_ear = mean_eye_aspect_ratio(left_eye, right_eye)
        ear = ear_average.update(raw_ear)
        mar = mouth_aspect_ratio(mouth_points)
        head_tilt = calculate_head_tilt(landmark_points)

        update_detection_counters(state, ear, mar, head_tilt)
        attention_score = calculate_attention_score(
            ear,
            mar,
            head_tilt,
            state.drowsy_counter,
            state.yawn_counter,
            state.tilt_counter,
        )

        status_text, status_color, events = build_status_messages(state)
        log_triggered_events(logger, events, ear, mar, head_tilt, attention_score)

        if state.drowsy_counter >= DROWSY_FRAME_LIMIT:
            start_alarm_if_needed(ALARM_FILE, state)

        draw_eye_landmarks(display_frame, left_eye, right_eye)
        draw_landmarks(display_frame, mouth_points)
    else:
        state.drowsy_counter = 0
        state.yawn_counter = 0
        state.tilt_counter = 0
        ear_average.clear()

    draw_dashboard(
        display_frame,
        status_text,
        status_color,
        ear,
        mar,
        head_tilt,
        attention_score,
        fps,
        low_light_enabled,
        brightness,
    )

    return display_frame


def run_detection() -> None:
    """Open the webcam and run the drowsiness detection loop."""
    ensure_alarm_file(ALARM_FILE)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Check camera permissions or device index.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mp_face_mesh = mp.solutions.face_mesh
    state = DetectionState()
    ear_average = MovingAverage(EAR_SMOOTHING_WINDOW)
    fps_counter = FPSCounter()
    logger = EventLogger(EVENT_LOG_FILE)

    try:
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Error: Could not read frame from webcam.")
                    break

                processed_frame = process_frame(
                    frame,
                    face_mesh,
                    state,
                    ear_average,
                    fps_counter,
                    logger,
                )

                cv2.imshow("Advanced Driver Drowsiness Detection", processed_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_detection()
