import os
import uuid
import json
import sqlite3
import threading
import traceback
from flask import Flask, request, render_template, send_from_directory, jsonify

from analyzer import analyze_and_clip

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
DB_PATH = "jobs.db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2GB upload cap (podcasts are big)

db_lock = threading.Lock()


def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db_lock:
        conn = get_db()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT,
                stage TEXT,
                progress INTEGER,
                error TEXT,
                data TEXT
            )
        """)
        conn.commit()
        conn.close()


init_db()


def set_job(job_id, **kwargs):
    with db_lock:
        conn = get_db()
        row = conn.execute("SELECT data FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        existing = json.loads(row["data"]) if row and row["data"] else {}
        existing.update(kwargs)

        conn.execute("""
            INSERT INTO jobs (job_id, status, stage, progress, error, data)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                status=excluded.status, stage=excluded.stage,
                progress=excluded.progress, error=excluded.error, data=excluded.data
        """, (
            job_id,
            existing.get("status"),
            existing.get("stage"),
            existing.get("progress", 0),
            existing.get("error"),
            json.dumps(existing),
        ))
        conn.commit()
        conn.close()


def get_job(job_id):
    with db_lock:
        conn = get_db()
        row = conn.execute("SELECT data FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        conn.close()
    if not row or not row["data"]:
        return None
    return json.loads(row["data"])


def run_job(job_id, input_path, clip_len, n_clips, vertical_crop):
    def progress_callback(stage, pct):
        set_job(job_id, stage=stage, progress=pct)

    try:
        set_job(job_id, status="processing", stage="Starting", progress=0)
        result = analyze_and_clip(
            input_path, OUTPUT_DIR, job_id, clip_len_sec=clip_len, n_clips=n_clips,
            vertical_crop=vertical_crop, progress_callback=progress_callback
        )

        clips_out = [
            {
                "download_url": f"/download/{c['filename']}",
                "start_time": c["start_time"],
                "end_time": c["end_time"],
                "duration": c["duration"],
                "hook": c["hook"],
                "vibe": c["vibe"],
            }
            for c in result["clips"]
        ]

        set_job(
            job_id,
            status="done",
            progress=100,
            stage="Done",
            source_duration=result["video_duration"],
            clips=clips_out,
        )
    except Exception as e:
        set_job(job_id, status="error", error=str(e))
        traceback.print_exc()
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    file = request.files["video"]
    clip_len = int(request.form.get("clip_len", 30))
    clip_len = max(15, min(60, clip_len))

    n_clips = int(request.form.get("n_clips", 1))
    n_clips = max(1, min(10, n_clips))

    vertical_crop = request.form.get("vertical_crop", "false").lower() == "true"

    job_id = str(uuid.uuid4())[:8]
    input_ext = os.path.splitext(file.filename)[1] or ".mp4"
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}{input_ext}")

    file.save(input_path)

    set_job(job_id, status="queued", stage="Queued", progress=0)

    thread = threading.Thread(
        target=run_job, args=(job_id, input_path, clip_len, n_clips, vertical_crop), daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Unknown job id"}), 404
    return jsonify(job)


@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
