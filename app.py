"""
Streamlit Driver Drowsiness Detection App

Install:
    pip install -r requirements.txt

Run:
    streamlit run app.py

The webcam is captured in the browser with streamlit-webrtc, while MediaPipe
Face Mesh and the drowsiness metrics run on the Python server.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import av
import cv2
import mediapipe as mp
import pandas as pd
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer

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
    get_eye_points,
    get_landmark_points,
    get_mouth_points,
    mean_eye_aspect_ratio,
    mouth_aspect_ratio,
)


BASE_DIR = Path(__file__).resolve().parent
EVENT_LOG_FILE = BASE_DIR / "drowsiness_events.csv"
MAX_PROCESSING_WIDTH = 640
MAX_CHART_POINTS = 120


@dataclass
class StreamMetrics:
    """Latest values displayed by the Streamlit dashboard."""

    status: str = "Stopped"
    ear: Optional[float] = None
    mar: Optional[float] = None
    head_tilt: Optional[float] = None
    attention_score: Optional[int] = None
    fps: float = 0.0
    brightness: float = 0.0
    low_light_enabled: bool = False
    ear_history: deque = field(default_factory=lambda: deque(maxlen=MAX_CHART_POINTS))
    alerts: deque = field(default_factory=lambda: deque(maxlen=50))


@dataclass
class DetectionCounters:
    """Consecutive-frame counters for fatigue signals."""

    drowsy: int = 0
    yawn: int = 0
    tilt: int = 0


class DrowsinessVideoProcessor(VideoProcessorBase):
    """Real-time video processor used by streamlit-webrtc."""

    def __init__(self):
        self.lock = threading.Lock()
        self.metrics = StreamMetrics(status="Starting")
        self.counters = DetectionCounters()
        self.ear_average = MovingAverage(EAR_SMOOTHING_WINDOW)
        self.fps_counter = FPSCounter()
        self.logger = EventLogger(EVENT_LOG_FILE)
        self.last_alert_times = {}
        self.frame_number = 0
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        processed = self.process_image(image)
        return av.VideoFrame.from_ndarray(processed, format="bgr24")

    def process_image(self, frame):
        """Process one webcam frame and return the annotated frame."""
        self.frame_number += 1
        frame = cv2.flip(frame, 1)
        frame = resize_for_realtime(frame, MAX_PROCESSING_WIDTH)
        display_frame, low_light_enabled, brightness = enhance_low_light(frame)

        fps = self.fps_counter.update()
        height, width = display_frame.shape[:2]

        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        status = "No face detected"
        status_color = WARNING_COLOR
        ear = None
        mar = None
        head_tilt = None
        attention_score = None
        events = []

        if results.multi_face_landmarks:
            landmark_points = get_landmark_points(
                results.multi_face_landmarks[0], width, height
            )
            left_eye = get_eye_points(landmark_points, EYE_LANDMARKS["left"])
            right_eye = get_eye_points(landmark_points, EYE_LANDMARKS["right"])
            mouth_points = get_mouth_points(landmark_points)

            raw_ear = mean_eye_aspect_ratio(left_eye, right_eye)
            ear = self.ear_average.update(raw_ear)
            mar = mouth_aspect_ratio(mouth_points)
            head_tilt = calculate_head_tilt(landmark_points)

            self.update_counters(ear, mar, head_tilt)
            attention_score = calculate_attention_score(
                ear,
                mar,
                head_tilt,
                self.counters.drowsy,
                self.counters.yawn,
                self.counters.tilt,
            )
            status, status_color, events = self.build_status()

            draw_eye_landmarks(display_frame, left_eye, right_eye)
            draw_landmarks(display_frame, mouth_points)
            self.record_events(events, ear, mar, head_tilt, attention_score)
        else:
            self.reset_counters()
            self.ear_average.clear()

        draw_video_overlay(
            display_frame,
            status,
            status_color,
            ear,
            mar,
            head_tilt,
            attention_score,
            fps,
            low_light_enabled,
            brightness,
        )
        self.update_metrics(
            status,
            ear,
            mar,
            head_tilt,
            attention_score,
            fps,
            low_light_enabled,
            brightness,
        )

        return display_frame

    def update_counters(self, ear: float, mar: float, head_tilt: float) -> None:
        self.counters.drowsy = self.counters.drowsy + 1 if ear < EAR_THRESHOLD else 0
        self.counters.yawn = self.counters.yawn + 1 if mar > MAR_THRESHOLD else 0
        self.counters.tilt = (
            self.counters.tilt + 1
            if abs(head_tilt) > HEAD_TILT_THRESHOLD_DEGREES
            else 0
        )

    def reset_counters(self) -> None:
        self.counters.drowsy = 0
        self.counters.yawn = 0
        self.counters.tilt = 0

    def build_status(self) -> tuple:
        events = []
        if self.counters.drowsy >= DROWSY_FRAME_LIMIT:
            events.append("DROWSY")
        if self.counters.yawn >= YAWN_FRAME_LIMIT:
            events.append("YAWNING")
        if self.counters.tilt >= HEAD_TILT_FRAME_LIMIT:
            events.append("HEAD TILT")

        if "DROWSY" in events:
            return "DROWSY!", DROWSY_COLOR, events
        if events:
            return " / ".join(events), WARNING_COLOR, events
        return "Awake", NORMAL_COLOR, events

    def record_events(
        self,
        events: list,
        ear: float,
        mar: float,
        head_tilt: float,
        attention_score: int,
    ) -> None:
        now = time.time()
        for event in events:
            last_time = self.last_alert_times.get(event, 0)
            if now - last_time < 3.0:
                continue

            self.last_alert_times[event] = now
            timestamp = time.strftime("%H:%M:%S", time.localtime(now))
            alert = {
                "time": timestamp,
                "event": event,
                "ear": round(ear, 3),
                "mar": round(mar, 3),
                "tilt": round(head_tilt, 1),
                "attention": attention_score,
            }

            self.logger.log_event(
                event_type=event,
                ear=ear,
                mar=mar,
                head_tilt=head_tilt,
                attention_score=attention_score,
                details="Streamlit dashboard event",
                cooldown_seconds=0,
            )
            with self.lock:
                self.metrics.alerts.appendleft(alert)

    def update_metrics(
        self,
        status: str,
        ear: Optional[float],
        mar: Optional[float],
        head_tilt: Optional[float],
        attention_score: Optional[int],
        fps: float,
        low_light_enabled: bool,
        brightness: float,
    ) -> None:
        with self.lock:
            self.metrics.status = status
            self.metrics.ear = ear
            self.metrics.mar = mar
            self.metrics.head_tilt = head_tilt
            self.metrics.attention_score = attention_score
            self.metrics.fps = fps
            self.metrics.low_light_enabled = low_light_enabled
            self.metrics.brightness = brightness
            if ear is not None:
                self.metrics.ear_history.append(
                    {
                        "frame": self.frame_number,
                        "EAR": ear,
                        "threshold": EAR_THRESHOLD,
                    }
                )

    def get_snapshot(self) -> StreamMetrics:
        with self.lock:
            return StreamMetrics(
                status=self.metrics.status,
                ear=self.metrics.ear,
                mar=self.metrics.mar,
                head_tilt=self.metrics.head_tilt,
                attention_score=self.metrics.attention_score,
                fps=self.metrics.fps,
                brightness=self.metrics.brightness,
                low_light_enabled=self.metrics.low_light_enabled,
                ear_history=deque(self.metrics.ear_history, maxlen=MAX_CHART_POINTS),
                alerts=deque(self.metrics.alerts, maxlen=50),
            )


def resize_for_realtime(frame, max_width: int):
    """Resize wide frames to reduce per-frame MediaPipe cost."""
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame

    scale = max_width / width
    new_height = int(height * scale)
    return cv2.resize(frame, (max_width, new_height), interpolation=cv2.INTER_AREA)


def draw_video_overlay(
    frame,
    status: str,
    status_color: tuple,
    ear: Optional[float],
    mar: Optional[float],
    head_tilt: Optional[float],
    attention_score: Optional[int],
    fps: float,
    low_light_enabled: bool,
    brightness: float,
) -> None:
    """Draw compact telemetry directly on the browser video feed."""
    draw_overlay_line(frame, f"FPS: {fps:.1f}", (18, 28), INFO_COLOR, scale=0.55)
    draw_overlay_line(
        frame,
        f"EAR: {ear:.2f}" if ear is not None else "EAR: --",
        (18, 54),
        INFO_COLOR,
        scale=0.55,
    )
    draw_overlay_line(
        frame,
        f"MAR: {mar:.2f}" if mar is not None else "MAR: --",
        (18, 80),
        INFO_COLOR,
        scale=0.55,
    )
    draw_overlay_line(
        frame,
        f"Tilt: {head_tilt:.1f} deg" if head_tilt is not None else "Tilt: --",
        (18, 106),
        INFO_COLOR,
        scale=0.55,
    )
    draw_overlay_line(
        frame,
        f"Attention: {attention_score}/100" if attention_score is not None else "Attention: --",
        (18, 132),
        INFO_COLOR,
        scale=0.55,
    )
    light_text = "Low light enhanced" if low_light_enabled else "Light OK"
    light_color = WARNING_COLOR if low_light_enabled else NORMAL_COLOR
    draw_overlay_line(frame, f"{light_text} ({brightness:.0f})", (18, 158), light_color, scale=0.55)
    draw_overlay_line(frame, status, (18, 196), status_color, scale=0.85, thickness=2)


def metric_text(value, precision: int = 2, suffix: str = "") -> str:
    if value is None:
        return "--"
    return f"{value:.{precision}f}{suffix}"


def render_dashboard(snapshot: StreamMetrics) -> None:
    """Render dashboard metrics, EAR chart, and alert table."""
    metric_cols = st.columns(5)
    metric_cols[0].metric("Status", snapshot.status)
    metric_cols[1].metric("Attention", "--" if snapshot.attention_score is None else snapshot.attention_score)
    metric_cols[2].metric("EAR", metric_text(snapshot.ear))
    metric_cols[3].metric("MAR", metric_text(snapshot.mar))
    metric_cols[4].metric("FPS", metric_text(snapshot.fps, precision=1))

    st.caption(
        f"Head tilt: {metric_text(snapshot.head_tilt, precision=1, suffix=' deg')} | "
        f"Brightness: {metric_text(snapshot.brightness, precision=0)} | "
        f"{'Low-light enhancement active' if snapshot.low_light_enabled else 'Lighting normal'}"
    )

    chart_df = pd.DataFrame(list(snapshot.ear_history))
    if chart_df.empty:
        st.info("Start detection to populate the real-time EAR graph.")
    else:
        st.line_chart(chart_df, x="frame", y=["EAR", "threshold"], height=260)

    alerts_df = pd.DataFrame(list(snapshot.alerts))
    if alerts_df.empty:
        st.info("No alerts logged in this session.")
    else:
        st.dataframe(alerts_df, use_container_width=True, hide_index=True)

    if EVENT_LOG_FILE.exists():
        st.download_button(
            "Download CSV logs",
            data=EVENT_LOG_FILE.read_bytes(),
            file_name="drowsiness_events.csv",
            mime="text/csv",
        )


def configure_page() -> None:
    st.set_page_config(
        page_title="Driver Drowsiness Detection",
        layout="wide",
    )
    st.title("Driver Drowsiness Detection Dashboard")
    st.caption("Real-time EAR, yawn, head tilt, low-light, and alert monitoring.")


def main() -> None:
    configure_page()

    if "detection_enabled" not in st.session_state:
        st.session_state.detection_enabled = False

    with st.sidebar:
        st.header("Controls")
        start_col, stop_col = st.columns(2)
        if start_col.button("Start", use_container_width=True):
            st.session_state.detection_enabled = True
        if stop_col.button("Stop", use_container_width=True):
            st.session_state.detection_enabled = False

        detection_enabled = st.session_state.detection_enabled
        st.success("Detection running") if detection_enabled else st.info("Detection stopped")
        st.caption("Allow camera access in the browser when prompted.")
        st.divider()
        st.write("Thresholds")
        st.write(f"EAR < `{EAR_THRESHOLD}` for `{DROWSY_FRAME_LIMIT}` frames")
        st.write(f"MAR > `{MAR_THRESHOLD}` for `{YAWN_FRAME_LIMIT}` frames")
        st.write(
            f"Head tilt > `{HEAD_TILT_THRESHOLD_DEGREES}` deg for "
            f"`{HEAD_TILT_FRAME_LIMIT}` frames"
        )

    video_col, dashboard_col = st.columns([1.25, 1])

    rtc_config = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    with video_col:
        st.subheader("Live Camera")
        ctx = webrtc_streamer(
            key="drowsiness-detector",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            video_processor_factory=DrowsinessVideoProcessor,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 24, "max": 30},
                },
                "audio": False,
            },
            async_processing=True,
            desired_playing=detection_enabled,
        )

    with dashboard_col:
        st.subheader("Detection Dashboard")
        dashboard_placeholder = st.empty()

    if ctx.video_processor:
        snapshot = ctx.video_processor.get_snapshot()
    else:
        snapshot = StreamMetrics(status="Stopped")

    with dashboard_placeholder.container():
        render_dashboard(snapshot)

    if detection_enabled:
        time.sleep(0.5)
        st.rerun()


if __name__ == "__main__":
    main()
