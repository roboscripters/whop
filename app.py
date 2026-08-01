import os
import uuid
from flask import Flask, request, render_template, send_from_directory, jsonify

from analyzer import analyze_and_clip

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB upload cap


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

    try:
        result = analyze_and_clip(input_path, output_path, clip_len_sec=clip_len)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

    return jsonify({
        "success": True,
        "download_url": f"/download/{os.path.basename(output_path)}",
        "start_time": result["start_time"],
        "end_time": result["end_time"],
        "clip_duration": result["duration"],
        "source_duration": result["video_duration"],
    })


@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
