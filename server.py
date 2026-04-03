"""
FastAPI backend for Just Dance.
Replaces game.py + video_player.py (pygame).

Run with:
    uvicorn server:app --host 0.0.0.0 --port 8000
or:
    python server.py
"""

import asyncio
import base64
import cv2
import math
import os
import time
import threading
import json
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from posecompare import PoseComparator
from shared import get_huge_shit, add_huge_shit, set_detection_events, get_detection_events


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SONG_DIR = Path("./song")

# ---------------------------------------------------------------------------
# Global state  (webcam is NOT opened here — only inside run_game_loop)
# ---------------------------------------------------------------------------
_connected: list[WebSocket] = []
_game_running = False
_main_loop: asyncio.AbstractEventLoop | None = None
_pose_comparator = PoseComparator()

# COCO skeleton connections (index pairs into the 17-keypoint array)
_SKELETON = [
    (5, 7), (7, 9),    # left arm
    (6, 8), (8, 10),   # right arm
    (5, 6),            # shoulders
    (11, 13), (13, 15),# left leg
    (12, 14), (14, 16),# right leg
    (11, 12),          # hips
    (5, 11), (6, 12),  # torso sides
    (0, 5), (0, 6),    # neck → shoulders
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_songs() -> list[str]:
    if not SONG_DIR.exists():
        return []
    return [d.name for d in SONG_DIR.iterdir() if d.is_dir()]


def get_tempo(song_name: str) -> int | None:
    metafile = SONG_DIR / song_name / f"{song_name}.meta"
    if not metafile.exists():
        return None
    try:
        return int(metafile.read_text().strip())
    except ValueError:
        return None


def extract_audio(song_name: str) -> Path | None:
    wav = SONG_DIR / song_name / f"{song_name}.wav"
    mp4 = SONG_DIR / song_name / f"{song_name}.mp4"
    if wav.exists():
        return wav
    if not mp4.exists():
        return None
    try:
        from moviepy import VideoFileClip
        clip = VideoFileClip(str(mp4))
        clip.audio.write_audiofile(str(wav))
        return wav
    except Exception as e:
        print(f"Audio extraction failed: {e}")
        return None


def ensure_poses_loaded(song_name: str):
    """Re-extract poses from the saved video if not already in memory."""
    try:
        get_huge_shit(song_name)
        return
    except KeyError:
        pass

    video_path = SONG_DIR / song_name / f"{song_name}.mp4"
    meta_path  = SONG_DIR / song_name / f"{song_name}.meta"
    if not video_path.exists() or not meta_path.exists():
        return

    print(f"Loading poses for '{song_name}' from disk...")
    model = _pose_comparator.foregroundPersons.yolo_pose_model

    bpm       = int(meta_path.read_text().strip())
    sample_ms = int(1000 / (bpm / 60))

    cap         = cv2.VideoCapture(str(video_path))
    fps         = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_ms = int(frame_count / fps * 1000)

    rows: list[list] = [[i, None] for i in range(0, duration_ms, sample_ms)]
    csv_idx      = 0
    total_frames = 0

    while cap.isOpened() and csv_idx < len(rows) - 1:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        ts = int(1000 * total_frames / fps)
        if ts >= rows[csv_idx][0]:
            csv_idx += 1
            cv2.imwrite("frame.jpg", frame)
            results = model("frame.jpg")
            kp = results[0].keypoints
            if kp is not None and len(kp.xy) > 0:
                rows[csv_idx][1] = kp.xy[0].cpu().numpy()

    cap.release()
    add_huge_shit(song_name, rows)
    print(f"Loaded {csv_idx} pose keyframes for '{song_name}'")


def _draw_silhouette(frame: np.ndarray, people: list) -> str | None:
    """
    Draw a glowing cyan skeleton silhouette on a black canvas.
    Returns base64-encoded JPEG, or None if no people detected.
    The frontend uses mix-blend-mode:screen so black → transparent.
    """
    if not people:
        return None

    h, w   = frame.shape[:2]
    out_w, out_h = 320, 240
    sx, sy = out_w / w, out_h / h

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    for (_tid, keypoints) in people:
        kps = (keypoints * np.array([sx, sy])).astype(int)

        # Draw limb lines
        for a, b in _SKELETON:
            if a >= len(kps) or b >= len(kps):
                continue
            pa, pb = tuple(kps[a]), tuple(kps[b])
            if (pa[0] == 0 and pa[1] == 0) or (pb[0] == 0 and pb[1] == 0):
                continue
            cv2.line(canvas, pa, pb, (180, 240, 255), 9)

        # Draw joint dots
        for kp in kps:
            if kp[0] != 0 or kp[1] != 0:
                cv2.circle(canvas, tuple(kp), 6, (255, 255, 255), -1)

    # Glow: blur and add back on top
    if np.any(canvas > 0):
        glow   = cv2.GaussianBlur(canvas, (23, 23), 0)
        canvas = cv2.addWeighted(canvas, 1.0, glow, 1.3, 0)

    _, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf).decode()


async def broadcast(message: dict):
    data = json.dumps(message)
    dead = []
    for client in list(_connected):
        try:
            await client.send_text(data)
        except Exception:
            dead.append(client)
    for c in dead:
        if c in _connected:
            _connected.remove(c)


# ---------------------------------------------------------------------------
# Game loop (runs in a background thread)
# ---------------------------------------------------------------------------

def run_game_loop(song_name: str):
    global _game_running

    tempo = get_tempo(song_name)
    if tempo is None:
        _game_running = False
        return

    # Open webcam HERE so it's released when the game ends
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Error: could not open webcam.")
        _game_running = False
        return

    video_path          = SONG_DIR / song_name / f"{song_name}.mp4"
    cap                 = cv2.VideoCapture(str(video_path))
    timestamps_and_poses= get_huge_shit(song_name)
    begin_song_time_ms  = int(timestamps_and_poses[0][0])

    video_offset = 0.20
    real_start   = time.time()
    prev_beat    = 0

    shared: dict = {
        "latest_scores": {},
        "effect_start":  None,
        "face_events":   [],
        "cumulative":    {},
        "silhouette":    None,
    }
    lock = threading.Lock()

    def process_beat(beat_idx: int, elapsed_s: float):
        if beat_idx >= len(timestamps_and_poses):
            return
        current_pose = timestamps_and_poses[beat_idx][1]

        ret, wc_frame = webcam.read()
        if not ret:
            return

        # Full analysis: depth filter → YOLO tracking → compare poses
        people = _pose_comparator.analyze_image(wc_frame)

        def to_rating(sim: float) -> str:
            if sim < 0.45: return "GREAT"
            if sim < 0.8:  return "OK"
            return "BAD"

        ratings = {}
        if people and current_pose is not None:
            for track_id, keypoints in people:
                sim = _pose_comparator.compare_poses(current_pose, keypoints)
                ratings[str(track_id)] = to_rating(sim)

        silhouette = _draw_silhouette(wc_frame, people)

        score_map = {"GREAT": 10, "OK": 5, "BAD": 0}
        with lock:
            shared["latest_scores"] = ratings
            shared["effect_start"]  = time.time()
            shared["silhouette"]    = silhouette
            shared["face_events"].append((elapsed_s * 1000, ratings))
            for pid, rating in ratings.items():
                shared["cumulative"][pid] = shared["cumulative"].get(pid, 0) + score_map[rating]
        print(f"Beat {beat_idx}: {ratings}")

    try:
        while cap.isOpened() and _game_running:
            elapsed_s = time.time() - real_start - video_offset
            elapsed_s = max(0.001, elapsed_s)
            cap.set(cv2.CAP_PROP_POS_MSEC, elapsed_s * 1000)
            ret, frame = cap.read()
            if not ret:
                break

            _, buf      = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 55])
            frame_b64   = base64.b64encode(buf).decode()

            beats_per_second = tempo / 60.0
            current_beat = max(
                0,
                math.floor((elapsed_s - begin_song_time_ms / 1000.0) * beats_per_second),
            )

            if current_beat > prev_beat:
                prev_beat = current_beat
                threading.Thread(
                    target=process_beat,
                    args=(current_beat, elapsed_s),
                    daemon=True,
                ).start()

            with lock:
                scores_snap    = dict(shared["latest_scores"])
                effect_start   = shared["effect_start"]
                cumulative_snap= dict(shared["cumulative"])
                silhouette_b64 = shared["silhouette"]

            effect_ms = (time.time() - effect_start) * 1000 if effect_start else None

            msg = {
                "type":       "frame",
                "frame":      frame_b64,
                "scores":     scores_snap,
                "cumulative": cumulative_snap,
                "beat":       current_beat,
                "effect_ms":  effect_ms,
                "silhouette": silhouette_b64,
            }

            if _main_loop and not _main_loop.is_closed():
                asyncio.run_coroutine_threadsafe(broadcast(msg), _main_loop)

            time.sleep(1 / 30)
    finally:
        cap.release()
        webcam.release()          # <-- webcam shut down here, always
        print("Webcam released.")

    set_detection_events(shared["face_events"])
    _game_running = False
    if _main_loop and not _main_loop.is_closed():
        asyncio.run_coroutine_threadsafe(broadcast({"type": "stopped"}), _main_loop)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    global _main_loop
    _main_loop = asyncio.get_event_loop()


@app.get("/songs")
def get_songs():
    return {"songs": list_songs()}


@app.post("/upload")
async def upload_song(
    background_tasks: BackgroundTasks,
    bpm: int = Form(...),
    start_beat: int = Form(0),
    song_name: str = Form(...),
    url: str = Form(...),
):
    from start_button_logic import upload_video
    background_tasks.add_task(upload_video, bpm, start_beat, song_name, url)
    return {"status": "uploading", "song": song_name}


@app.post("/start/{song_name}")
def start_game(song_name: str):
    global _game_running
    if _game_running:
        return {"error": "Game already running"}
    if song_name not in list_songs():
        return {"error": "Song not found"}
    ensure_poses_loaded(song_name)
    try:
        get_huge_shit(song_name)
    except KeyError:
        return {"error": "Could not load pose data for this song"}
    _game_running = True
    threading.Thread(target=run_game_loop, args=(song_name,), daemon=True).start()
    return {"status": "started"}


@app.post("/stop")
def stop_game():
    global _game_running
    _game_running = False
    return {"status": "stopped"}


@app.get("/song/{song_name}/audio")
def get_audio(song_name: str):
    wav = extract_audio(song_name)
    if wav and wav.exists():
        return FileResponse(str(wav), media_type="audio/wav")
    return {"error": "Audio not available"}


@app.get("/stats")
def get_stats():
    events = get_detection_events()
    if not events:
        return {"players": {}}

    score_map = {"GREAT": 10, "OK": 5, "BAD": 0}
    players: dict[str, dict] = {}

    for _ts, ratings in events:
        if not isinstance(ratings, dict):
            continue
        for pid, rating in ratings.items():
            if pid not in players:
                players[pid] = {"GREAT": 0, "OK": 0, "BAD": 0, "score": 0}
            players[pid][rating] = players[pid].get(rating, 0) + 1
            players[pid]["score"] += score_map.get(rating, 0)

    return {"players": players}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _connected.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        if ws in _connected:
            _connected.remove(ws)


_frontend_dist = Path("frontend/dist")
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
