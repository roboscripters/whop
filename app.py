import os
import uuid
import threading
import traceback
from flask import Flask, request, render_template, send_from_directory, jsonify

from analyzer import analyze_and_clip

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB upload cap

# In-memory job status store. Fine for single-worker/personal use.
# Note: gunicorn -w 2 means 2 separate processes, each with its own copy of
# this dict — a status poll could hit the "wrong" worker and see nothing.
# The included service file should use -w 1 for this reason (see README).
jobs = {}
jobs_lock = threading.Lock()


def set_job(job_id, **kwargs):
    with jobs_lock:
        jobs[job_id].update(kwargs)


def run_job(job_id, input_path, output_path, clip_len):
    def progress_callback(stage, pct):
        set_job(job_id, stage=stage, progress=pct)

    try:
        set_job(job_id, status="processing", stage="Starting", progress=0)
        result = analyze_and_clip(
            input_path, output_path, clip_len_sec=clip_len,
            progress_callback=progress_callback
        )
        set_job(
            job_id,
            status="done",
            progress=100,
            stage="Done",
            download_url=f"/download/{os.path.basename(output_path)}",
            start_time=result["start_time"],
            end_time=result["end_time"],
            clip_duration=result["duration"],
            source_duration=result["video_duration"],
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
    clip_len = max(15, min(60, clip_len))  # enforce 15-60s range

    job_id = str(uuid.uuid4())[:8]
    input_ext = os.path.splitext(file.filename)[1] or ".mp4"
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}{input_ext}")
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}_clip.mp4")

    file.save(input_path)

    with jobs_lock:
        jobs[job_id] = {"status": "queued", "stage": "Queued", "progress": 0}

    thread = threading.Thread(
        target=run_job, args=(job_id, input_path, output_path, clip_len), daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job id"}), 404
    return jsonify(job)


@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
