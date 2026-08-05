"""
analyzer.py
Core logic for finding the best clip-worthy moment(s) in a video.

Primary approach: upload the actual video to Gemini and let it watch/listen
to the full thing natively — real audio-visual content understanding, not
pattern-matching on loudness/motion. Gemini picks its own start/end times
(up to 60s each, no fixed length) and writes the hook itself.

Fallback tiers (used if Gemini is unavailable or fails):
1. Claude reads the Whisper transcript only (no video) and picks moments
   from dialogue alone.
2. Pure audio/motion energy scoring with a generic vibe-matched hook.

This ensures the tool never simply breaks, even with no API keys configured.

Supports two modes:
- Single clip (short trailers/scenes): one best moment
- Multi-clip (long podcasts/pilots): N best non-overlapping moments
"""

import subprocess
import numpy as np
import librosa
import cv2
import os
import json
import random
from faster_whisper import WhisperModel
from anthropic import Anthropic

WINDOW_SEC = 0.5  # resolution of the fallback excitement timeline

# Videos longer than this get sparser frame sampling for motion analysis,
# so a 2-hour podcast doesn't take forever to process even on strong hardware.
LONG_VIDEO_THRESHOLD_SEC = 20 * 60

MAX_CLIP_LEN_SEC = 60
MIN_CLIP_LEN_SEC = 8

# Model used for moment selection. Haiku is cheap and plenty capable for
# this — it's a judgment/ranking task on text, not something that needs
# the largest model. Bump to a Sonnet model string if you want higher
# quality picks and don't mind the extra cost per video.
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Whisper transcription
# ---------------------------------------------------------------------------

_whisper_model = None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model


def transcribe_video(audio_path):
    """
    Run Whisper on the extracted audio track. Returns a list of
    {start, end, text} segments with timestamps in the source video's
    timeline (seconds). Returns [] if transcription finds no speech.
    """
    model = _get_whisper_model()
    segments, _info = model.transcribe(audio_path, beam_size=1, vad_filter=True)

    result = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            result.append({"start": seg.start, "end": seg.end, "text": text})
    return result


# ---------------------------------------------------------------------------
# AI-driven moment selection (the core upgrade)
# ---------------------------------------------------------------------------

def _format_transcript_for_prompt(transcript):
    lines = []
    for seg in transcript:
        lines.append(f"[{seg['start']:.1f}-{seg['end']:.1f}] {seg['text']}")
    return "\n".join(lines)


def find_viral_moments_with_ai(transcript, video_duration, n_clips=1):
    """
    Send the transcript to Claude and ask it to pick the N best clip-worthy
    moments, each up to MAX_CLIP_LEN_SEC long, with a hook written for each.

    Returns a list of {start, end, hook} dicts, or None if the AI call
    fails for any reason (missing key, API error, bad response) — callers
    should fall back to the heuristic method in that case.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    transcript_text = _format_transcript_for_prompt(transcript)

    prompt = f"""You are a viral short-form video clip scout. You're given the full
transcript of a video (a film, TV pilot, or podcast) with timestamps in
seconds. Your job is to find the {n_clips} single best moment(s) to cut
into standalone short clips for Instagram Reels / TikTok / YouTube Shorts.

Look for: the sharpest joke, the biggest twist or reveal, the most
emotionally intense exchange, or dialogue that's quotable on its own
without needing the rest of the story for context.

Rules:
- Each moment must be between {MIN_CLIP_LEN_SEC} and {MAX_CLIP_LEN_SEC} seconds long —
  choose whatever length within that range best captures the complete moment.
  Don't default to a fixed length; a sharp 10-second exchange beats a padded 45-second one.
- Pick exactly {n_clips} moment(s), and they must not overlap in time.
- For each moment, write one short, punchy hook caption (under 12 words) in the
  style of viral relatable-caption accounts — e.g. "POV: the moment everything changes",
  or a real quoted line from the transcript if it's strong enough on its own.
- Respond with ONLY a JSON array, no other text, no markdown formatting. Format:
[{{"start": 123.4, "end": 145.0, "hook": "your hook text here"}}]

Video duration: {video_duration:.1f} seconds

Transcript:
{transcript_text}
"""

    try:
        client = Anthropic(api_key=api_key)
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = message.content[0].text.strip()
        moments = _parse_ai_json_response(raw_text)
        return _validate_moments(moments, video_duration)

    except Exception as e:
        print(f"Text-only AI moment selection failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Gemini native video understanding (the real upgrade) — Gemini watches the
# ACTUAL video file directly (frames + audio, natively, continuously), not
# a handful of sampled frames from AI-guessed candidate windows. This is
# genuinely closer to how a human editor reviews footage.
# ---------------------------------------------------------------------------

# gemini-2.5-flash is the sensible cost/quality tier for this — a judgment
# task on footage, not something that needs the priciest Pro-tier model.
GEMINI_MODEL = "gemini-2.5-flash"

# Gemini video uploads are processed asynchronously — poll until ready,
# but don't wait forever if something's stuck.
GEMINI_UPLOAD_TIMEOUT_SEC = 300


def find_viral_moments_with_gemini(video_path, video_duration, n_clips=1):
    """
    Upload the actual video file to Gemini and ask it to find the best
    moment(s) — it natively processes both video frames and audio together,
    so it can judge visual moments (a reaction, a stunt) as well as dialogue,
    without needing a separate transcript or frame-sampling step.

    Returns a list of {start, end, hook} dicts, or None if anything fails —
    callers should fall back to Claude's text-only tier or the energy
    heuristic in that case.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    uploaded_file = None
    try:
        from google import genai
        from google.genai import types
        import time

        client = genai.Client(api_key=api_key)

        uploaded_file = client.files.upload(file=video_path)

        waited = 0
        while uploaded_file.state.name == "PROCESSING" and waited < GEMINI_UPLOAD_TIMEOUT_SEC:
            time.sleep(3)
            waited += 3
            uploaded_file = client.files.get(name=uploaded_file.name)

        if uploaded_file.state.name != "ACTIVE":
            print(f"Gemini file upload did not become ACTIVE (state: {uploaded_file.state.name})")
            return None

        prompt = f"""You are a viral short-form video clip scout reviewing a film, TV
episode, or podcast to find the best moment(s) to cut into standalone clips
for Instagram Reels / TikTok / YouTube Shorts.

Watch and listen to the full video. Look for: the sharpest joke, the biggest
twist or reveal, the most emotionally intense exchange, a striking visual
moment (a reaction, a stunt, a visual gag), or dialogue that's quotable on
its own without needing the rest of the story for context. Judge holistically
based on both what's said AND what's shown — like a real editor would.

Rules:
- Find exactly {n_clips} moment(s), non-overlapping.
- Each moment must be between {MIN_CLIP_LEN_SEC} and {MAX_CLIP_LEN_SEC} seconds long —
  choose whatever length within that range best captures the complete moment.
  Don't default to a fixed length; a sharp 10-second exchange beats a padded 45-second one.
- For each moment, write one short, punchy hook caption (under 12 words) in the
  style of viral relatable-caption accounts, or a real quoted line if it's strong enough.
- Respond with ONLY a JSON array, no other text, no markdown formatting. Format:
[{{"start": 123.4, "end": 145.0, "hook": "your hook text here"}}]

Video duration: {video_duration:.1f} seconds
"""

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[uploaded_file, prompt],
        )
        raw_text = response.text.strip()
        moments = _parse_ai_json_response(raw_text)
        return _validate_moments(moments, video_duration)

    except Exception as e:
        print(f"Gemini moment selection failed: {e}")
        return None

    finally:
        # Clean up the uploaded file from Gemini's storage — no need to
        # keep it around after this one analysis.
        if uploaded_file is not None:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass


def _parse_ai_json_response(raw_text):
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()
    return json.loads(raw_text)


def _validate_moments(moments, video_duration):
    """Never trust external AI output blindly — clamp everything to sane bounds."""
    cleaned = []
    for m in moments:
        start = float(m["start"])
        end = float(m["end"])
        hook = str(m["hook"]).strip()
        if end <= start:
            continue
        length = min(end - start, MAX_CLIP_LEN_SEC)
        end = start + length
        start = max(0, start)
        end = min(end, video_duration)
        if end - start < 1:
            continue
        cleaned.append({"start": start, "end": end, "hook": hook})
    return cleaned if cleaned else None


# ---------------------------------------------------------------------------
# Fallback: audio/motion energy scoring (used only if AI is unavailable)
# ---------------------------------------------------------------------------

def get_video_duration(video_path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", video_path],
        capture_output=True, text=True
    )
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def get_video_dimensions(video_path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", video_path],
        capture_output=True, text=True
    )
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    return int(stream["width"]), int(stream["height"])


def extract_audio(video_path, audio_path):
    """Pull audio out as mono wav for librosa/Whisper analysis."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1",
         "-ar", "22050", "-f", "wav", audio_path],
        check=True, capture_output=True
    )


def audio_energy_timeline(audio_path, duration, window_sec=WINDOW_SEC):
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    hop_length = int(sr * window_sec)
    rms = librosa.feature.rms(y=y, frame_length=hop_length * 2, hop_length=hop_length)[0]

    n_windows = int(np.ceil(duration / window_sec))
    if len(rms) < n_windows:
        rms = np.pad(rms, (0, n_windows - len(rms)), mode="edge")
    else:
        rms = rms[:n_windows]

    return normalize(rms)


def motion_energy_timeline(video_path, duration, window_sec=WINDOW_SEC):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    samples_per_sec = 1 if duration > LONG_VIDEO_THRESHOLD_SEC else 5
    sample_every_n_frames = max(1, int(fps / samples_per_sec))

    n_windows = int(np.ceil(duration / window_sec))
    window_scores = [[] for _ in range(n_windows)]

    prev_gray = None
    frame_idx = 0
    ret, frame = cap.read()
    while ret:
        if frame_idx % sample_every_n_frames == 0:
            small = cv2.resize(frame, (160, 90))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                score = float(np.mean(diff))
                t = frame_idx / fps
                w = int(t / window_sec)
                if w < n_windows:
                    window_scores[w].append(score)
            prev_gray = gray
        ret, frame = cap.read()
        frame_idx += 1

    cap.release()

    timeline = np.array([
        np.mean(w) if len(w) > 0 else 0.0 for w in window_scores
    ])
    return normalize(timeline)


def normalize(arr):
    arr = np.array(arr, dtype=float)
    if arr.max() - arr.min() < 1e-8:
        return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())


def find_top_segments(combined_score, window_sec, clip_len_sec, n_clips=1,
                       hook_boost_sec=2.0, min_gap_sec=None):
    if min_gap_sec is None:
        min_gap_sec = clip_len_sec

    n_windows = len(combined_score)
    clip_windows = max(1, int(round(clip_len_sec / window_sec)))
    gap_windows = max(1, int(round(min_gap_sec / window_sec)))

    if clip_windows >= n_windows:
        return [(0.0, n_windows * window_sec)]

    cumsum = np.cumsum(np.insert(combined_score, 0, 0))
    window_sums = [
        (cumsum[start + clip_windows] - cumsum[start], start)
        for start in range(0, n_windows - clip_windows + 1)
    ]
    window_sums.sort(key=lambda x: -x[0])

    chosen_starts = []
    for score, start in window_sums:
        if len(chosen_starts) >= n_clips:
            break
        if any(abs(start - c) < gap_windows for c in chosen_starts):
            continue
        chosen_starts.append(start)

    segments = []
    for start_idx in sorted(chosen_starts):
        boost_windows = int(hook_boost_sec / window_sec)
        search_end = min(start_idx + boost_windows, n_windows - 1)
        best_rise_idx = start_idx
        best_rise_val = -1
        for i in range(start_idx, search_end):
            if i + 1 < n_windows:
                rise = combined_score[i + 1] - combined_score[i]
                if rise > best_rise_val:
                    best_rise_val = rise
                    best_rise_idx = i

        start_time = best_rise_idx * window_sec
        end_time = start_time + clip_len_sec
        segments.append((start_time, end_time))

    return segments


HOOK_POOLS = {
    "action": [
        "Wait for it...",
        "This is why you don't blink",
        "The moment everything goes wrong",
        "Nobody was ready for this",
        "This escalated fast",
    ],
    "dialogue": [
        "This line hits different",
        "Nobody expected this response",
        "This is the moment everything changed",
        "Read between the lines on this one",
    ],
    "relatable": [
        "POV: this is literally you",
        "This is way too accurate",
        "Why is this so real though",
    ],
    "dramatic": [
        "This scene will wreck you",
        "This is the turning point",
        "Everything builds to this",
    ],
}


def classify_vibe(audio_seg, motion_seg):
    avg_audio = float(np.mean(audio_seg)) if len(audio_seg) else 0.0
    avg_motion = float(np.mean(motion_seg)) if len(motion_seg) else 0.0
    volatility = float(np.std(audio_seg) + np.std(motion_seg)) if len(audio_seg) else 0.0

    high_audio = avg_audio > 0.5
    high_motion = avg_motion > 0.5
    high_volatility = volatility > 0.35

    if high_volatility:
        return "dramatic"
    if high_audio and high_motion:
        return "action"
    if high_audio and not high_motion:
        return "dialogue"
    return "relatable"


def fallback_find_and_hook(video_path, audio_path, duration, n_clips, precomputed=None):
    """
    The old heuristic method: audio/motion energy scoring with a fixed
    clip length and generic vibe-matched hooks. Used only when both AI
    tiers are unavailable.

    precomputed: optionally pass (audio_scores, motion_scores) if already
    computed by the caller, to avoid redundant work. If None, computes
    them fresh from audio_path/video_path.
    """
    if precomputed is not None:
        audio_scores, motion_scores = precomputed
    else:
        audio_scores = audio_energy_timeline(audio_path, duration)
        motion_scores = motion_energy_timeline(video_path, duration)
        n = min(len(audio_scores), len(motion_scores))
        audio_scores, motion_scores = audio_scores[:n], motion_scores[:n]

    combined = 0.5 * audio_scores + 0.5 * motion_scores

    default_len = min(30, duration)
    segments = find_top_segments(combined, WINDOW_SEC, default_len, n_clips=n_clips)

    results = []
    for start_time, end_time in segments:
        start_idx = int(start_time / WINDOW_SEC)
        end_idx = int(end_time / WINDOW_SEC)
        vibe = classify_vibe(audio_scores[start_idx:end_idx], motion_scores[start_idx:end_idx])
        hook = random.choice(HOOK_POOLS[vibe])
        results.append({"start": start_time, "end": end_time, "hook": hook})
    return results


# ---------------------------------------------------------------------------
# Vertical crop (face-aware)
# ---------------------------------------------------------------------------

_face_cascade = None


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def find_crop_center_x(video_path, start_time, end_time, video_width, sample_count=6):
    cascade = _get_face_cascade()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    duration = end_time - start_time
    sample_times = [
        start_time + duration * (i + 1) / (sample_count + 1)
        for i in range(sample_count)
    ]

    face_centers = []
    for t in sample_times:
        frame_num = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) > 0:
            largest = max(faces, key=lambda f: f[2] * f[3])
            face_center_x = largest[0] + largest[2] / 2
            face_centers.append(face_center_x)

    cap.release()

    if face_centers:
        return float(np.median(face_centers))
    return video_width / 2.0


def build_crop_filter(video_width, video_height, center_x, target_ratio=9 / 16):
    crop_width = int(video_height * target_ratio)
    if crop_width > video_width:
        crop_width = video_width

    crop_x = int(center_x - crop_width / 2)
    crop_x = max(0, min(crop_x, video_width - crop_width))

    return f"crop={crop_width}:{video_height}:{crop_x}:0"


# ---------------------------------------------------------------------------
# Cutting
# ---------------------------------------------------------------------------

def cut_clip(video_path, start_time, end_time, output_path, crop_filter=None,
             target_width=1080, target_height=1920):
    """Cut using ffmpeg with re-encode for frame-accurate trimming.

    -movflags +faststart moves the mp4 index (moov atom) to the front of
    the file so browsers can stream/seek immediately instead of needing
    the whole file downloaded first.
    """
    duration = end_time - start_time
    cmd = ["ffmpeg", "-y", "-ss", str(start_time), "-i", video_path, "-t", str(duration)]

    if crop_filter:
        vf = f"{crop_filter},scale={target_width}:{target_height}"
        cmd += ["-vf", vf]

    cmd += ["-c:v", "libx264", "-c:a", "aac", "-preset", "fast",
            "-movflags", "+faststart", output_path]

    subprocess.run(cmd, check=True, capture_output=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_and_clip(video_path, output_dir, job_id, n_clips=1,
                      vertical_crop=False, progress_callback=None):
    """
    Main entry point. Returns a dict with a 'clips' list (each with its own
    timing, hook, output path) plus overall video_duration.

    n_clips: how many best moments to extract. Use 1 for a single scene/
    trailer, 3-5+ for a full episode/podcast.

    vertical_crop: if True, output clips are cropped to 9:16 (centered on
    detected faces where possible) instead of keeping the source aspect ratio.

    Moment selection tries three tiers, in order, each falling back to the
    next if it's unavailable or fails:
    1. Gemini native video understanding: Gemini watches the actual video
       file directly — frames AND audio, continuously, natively — and
       picks the best moment(s) + writes the hook. The strongest tier,
       since it isn't limited to sampled frames or transcript text alone.
    2. Claude text-only: reads the full Whisper transcript (no video) and
       picks moments based on dialogue alone. Used if Gemini is
       unavailable/fails but transcription succeeded.
    3. Energy heuristic: audio/motion energy scoring with a fixed 30s
       length and a generic vibe-matched hook. Used if neither AI tier
       is available/works, or transcription found no speech.

    progress_callback(stage: str, pct: int) is called at each step, if
    provided, so a caller (e.g. a Flask job-status endpoint) can report
    live progress.
    """
    def report(stage, pct):
        if progress_callback:
            progress_callback(stage, pct)

    report("Reading video info", 5)
    duration = get_video_duration(video_path)
    video_width, video_height = get_video_dimensions(video_path)

    moments = None
    hook_source = "fallback"

    report("AI watching the video (Gemini)", 20)
    moments = find_viral_moments_with_gemini(video_path, duration, n_clips=n_clips)
    if moments is not None:
        hook_source = "ai_video"

    # Whisper/energy work is only needed for the fallback tiers below —
    # skip it entirely if Gemini already succeeded, saving real time.
    if moments is None:
        report("Extracting audio", 45)
        audio_path = video_path + ".wav"
        extract_audio(video_path, audio_path)

        report("Transcribing speech", 55)
        try:
            transcript = transcribe_video(audio_path)
        except Exception as e:
            print(f"Transcription failed: {e}")
            transcript = None

        report("Analyzing audio energy", 65)
        audio_scores = audio_energy_timeline(audio_path, duration)
        os.remove(audio_path)

        report("Analyzing motion energy", 72)
        motion_scores = motion_energy_timeline(video_path, duration)

        n = min(len(audio_scores), len(motion_scores))
        audio_scores, motion_scores = audio_scores[:n], motion_scores[:n]

        if transcript:
            report("Falling back to transcript-only AI analysis", 78)
            moments = find_viral_moments_with_ai(transcript, duration, n_clips=n_clips)
            if moments is not None:
                hook_source = "ai_text"

        if moments is None:
            report("Falling back to energy-based scoring", 82)
            moments = fallback_find_and_hook(video_path, None, duration, n_clips,
                                              precomputed=(audio_scores, motion_scores))
            hook_source = "fallback"

    clips = []
    total = len(moments)
    for i, m in enumerate(moments):
        start_time, end_time, hook = m["start"], m["end"], m["hook"]
        pct = 85 + int(10 * (i + 1) / max(1, total))

        crop_filter = None
        if vertical_crop:
            report(f"Finding subject for clip {i + 1}/{total}", pct)
            center_x = find_crop_center_x(video_path, start_time, end_time, video_width)
            crop_filter = build_crop_filter(video_width, video_height, center_x)

        report(f"Cutting clip {i + 1}/{total}", pct)
        output_path = os.path.join(output_dir, f"{job_id}_clip{i + 1}.mp4")
        cut_clip(video_path, start_time, end_time, output_path, crop_filter=crop_filter)

        clips.append({
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
            "duration": round(end_time - start_time, 2),
            "output_path": output_path,
            "filename": os.path.basename(output_path),
            "hook": hook,
            "hook_source": hook_source,
        })

    report("Done", 100)
    return {
        "video_duration": round(duration, 2),
        "clips": clips,
    }
