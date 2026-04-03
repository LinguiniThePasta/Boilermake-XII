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

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from posecompare import PoseComparator
from shared import get_huge_shit, add_huge_shit, set_detection_events


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SONG_DIR = Path("./song")

# --- Global state ---
_connected: list[WebSocket] = []
_game_running = False
_main_loop: asyncio.AbstractEventLoop | None = None
_webcam = cv2.VideoCapture(0)
_pose_comparator = PoseComparator()


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
        return  # already loaded
    except KeyError:
        pass

    video_path = SONG_DIR / song_name / f"{song_name}.mp4"
    meta_path = SONG_DIR / song_name / f"{song_name}.meta"
    if not video_path.exists() or not meta_path.exists():
        return

    print(f"Loading poses for '{song_name}' from disk...")
    from ultralytics import YOLO
    model = YOLO("yolo11n-pose.pt")

    bpm = int(meta_path.read_text().strip())
    sample_ms = int(1000 / (bpm / 60))  # ms per beat

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_ms = int(frame_count / fps * 1000)

    rows: list[list] = [[i, None] for i in range(0, duration_ms, sample_ms)]
    csv_idx = 0
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
            rows[csv_idx][1] = results[0].keypoints.xy[0].cpu().numpy()

    cap.release()
    add_huge_shit(song_name, rows)
    print(f"Loaded {csv_idx} pose keyframes for '{song_name}'")


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

    video_path = SONG_DIR / song_name / f"{song_name}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    timestamps_and_poses = get_huge_shit(song_name)
    begin_song_time_ms = int(timestamps_and_poses[0][0])  # ms from start

    video_offset = 0.20  # seconds, for audio-video sync
    real_start = time.time()
    prev_beat = 0

    shared: dict = {"latest_scores": {}, "effect_start": None, "face_events": []}
    lock = threading.Lock()

    def process_beat(beat_idx: int, elapsed_s: float):
        if beat_idx >= len(timestamps_and_poses):
            return
        current_pose = timestamps_and_poses[beat_idx][1]
        ret, frame = _webcam.read()
        if not ret:
            return
        raw = _pose_comparator.compare_all_players(current_pose, frame)

        def to_rating(sim: float) -> str:
            if sim < 0.45:
                return "GREAT"
            if sim < 0.8:
                return "OK"
            return "BAD"

        ratings = {str(tid): to_rating(s) for tid, s in raw.items()} if raw else {}
        with lock:
            shared["latest_scores"] = ratings
            shared["effect_start"] = time.time()
            shared["face_events"].append((elapsed_s * 1000, ratings))
        print(f"Beat {beat_idx}: {ratings}")

    while cap.isOpened() and _game_running:
        elapsed_s = time.time() - real_start - video_offset
        elapsed_s = max(0.001, elapsed_s)
        cap.set(cv2.CAP_PROP_POS_MSEC, elapsed_s * 1000)
        ret, frame = cap.read()
        if not ret:
            break

        # Encode frame to JPEG base64 for WebSocket transport
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 55])
        frame_b64 = base64.b64encode(buf).decode()

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
            scores_snap = dict(shared["latest_scores"])
            effect_start = shared["effect_start"]

        effect_ms = (time.time() - effect_start) * 1000 if effect_start else None

        msg = {
            "type": "frame",
            "frame": frame_b64,
            "scores": scores_snap,
            "beat": current_beat,
            "effect_ms": effect_ms,
        }

        if _main_loop and not _main_loop.is_closed():
            asyncio.run_coroutine_threadsafe(broadcast(msg), _main_loop)

        time.sleep(1 / 30)  # cap at ~30 fps

    cap.release()
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


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _connected.append(ws)
    try:
        while True:
            await ws.receive_text()  # keep connection alive
    except WebSocketDisconnect:
        if ws in _connected:
            _connected.remove(ws)


# Serve the built TypeScript frontend (after running `npm run build` in frontend/)
_frontend_dist = Path("frontend/dist")
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
