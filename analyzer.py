"""
analyzer.py
Core logic for finding the "best hook" segment(s) in a video using
audio energy + motion energy, then cutting them with ffmpeg.

Supports two modes:
- Single clip (short trailers): one best segment
- Multi-clip (long podcasts): N best non-overlapping segments
"""

import subprocess
import numpy as np
import librosa
import cv2
import os
import json
import random
import re
from faster_whisper import WhisperModel

WINDOW_SEC = 0.5  # resolution of the excitement timeline

# Videos longer than this get sparser frame sampling for motion analysis,
# so a 2-hour podcast doesn't take forever to process even on strong hardware.
LONG_VIDEO_THRESHOLD_SEC = 20 * 60

# Whisper model, loaded once and reused across jobs (loading takes a few
# seconds, so we don't want to redo it per-clip or per-request).
# "base" is a good speed/accuracy tradeoff for CPU; "small" is more accurate
# but slower — bump this if hook quality matters more than processing time.
_whisper_model = None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model


def transcribe_video(video_path, audio_path):
    """
    Run Whisper on the extracted audio track. Returns a list of
    {start, end, text} segments with timestamps in the source video's
    timeline (seconds).
    """
    model = _get_whisper_model()
    segments, _info = model.transcribe(audio_path, beam_size=1, vad_filter=True)

    result = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            result.append({"start": seg.start, "end": seg.end, "text": text})
    return result


def find_best_line(transcript, start_time, end_time):
    """
    From the transcript segments overlapping [start_time, end_time], pick
    the single most 'quotable' line to use as the hook — preferring
    questions, exclamations, and emphatic short lines over flat narration.

    Returns the line text, or None if no transcript overlaps this range.
    """
    overlapping = [
        seg for seg in transcript
        if seg["end"] > start_time and seg["start"] < end_time
    ]
    if not overlapping:
        return None

    def quotability(seg):
        text = seg["text"]
        score = 0
        if "?" in text:
            score += 3
        if "!" in text:
            score += 3
        word_count = len(text.split())
        # Prefer punchy lines — not too short (fragments), not too long (rambling)
        if 4 <= word_count <= 14:
            score += 2
        elif word_count > 20:
            score -= 2
        # Emphatic/superlative language bumps quotability
        emphasis_words = ["never", "always", "worst", "best", "can't believe",
                           "insane", "crazy", "no way", "literally", "actually"]
        lowered = text.lower()
        score += sum(1 for w in emphasis_words if w in lowered)
        return score

    best = max(overlapping, key=quotability)
    return best["text"]


def write_srt(transcript, start_time, end_time, srt_path):
    """
    Write an SRT subtitle file for the transcript segments overlapping
    [start_time, end_time], with timestamps re-based to start at 0
    (since the output clip itself starts at 0, not at start_time).
    """
    overlapping = [
        seg for seg in transcript
        if seg["end"] > start_time and seg["start"] < end_time
    ]

    def fmt_ts(t):
        t = max(0, t)
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t - int(t)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(overlapping, start=1):
            rel_start = max(0, seg["start"] - start_time)
            rel_end = max(0, seg["end"] - start_time)
            if rel_end <= rel_start:
                continue
            f.write(f"{i}\n")
            f.write(f"{fmt_ts(rel_start)} --> {fmt_ts(rel_end)}\n")
            f.write(f"{seg['text'].strip()}\n\n")


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


_face_cascade = None


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def find_crop_center_x(video_path, start_time, end_time, video_width, sample_count=6):
    """
    Sample a handful of frames within the clip's time range and look for
    faces, to find a good horizontal center point for a 9:16 crop.

    Returns the x-coordinate (in source pixels) to center the crop on.
    Falls back to the frame's horizontal center if no faces are found
    anywhere in the sample (e.g. scenery/action shots).
    """
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
            # Use the largest face found (most likely the main subject)
            largest = max(faces, key=lambda f: f[2] * f[3])
            face_center_x = largest[0] + largest[2] / 2
            face_centers.append(face_center_x)

    cap.release()

    if face_centers:
        return float(np.median(face_centers))
    return video_width / 2.0


def build_crop_filter(video_width, video_height, center_x, target_ratio=9 / 16):
    """
    Build an ffmpeg crop filter string for a 9:16 vertical crop, centered
    on center_x but clamped so the crop window stays within frame bounds.
    """
    crop_width = int(video_height * target_ratio)
    if crop_width > video_width:
        # Source is already narrower than a 9:16 crop would need —
        # just use the full width instead (can't crop wider than source).
        crop_width = video_width

    crop_x = int(center_x - crop_width / 2)
    crop_x = max(0, min(crop_x, video_width - crop_width))

    return f"crop={crop_width}:{video_height}:{crop_x}:0"


def extract_audio(video_path, audio_path):
    """Pull audio out as mono wav for librosa analysis."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1",
         "-ar", "22050", "-f", "wav", audio_path],
        check=True, capture_output=True
    )


def audio_energy_timeline(audio_path, duration, window_sec=WINDOW_SEC):
    """Return an array of RMS energy values, one per window_sec chunk."""
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
    """
    Sample frames at a fixed rate, compute frame-to-frame pixel diff
    (downscaled + grayscale for speed) as a proxy for visual 'busyness'.

    For long videos (podcasts), sample less densely — we don't need 5
    frames/sec of motion data for a 2-hour recording, and it keeps
    processing time reasonable even with plenty of CPU available.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    if duration > LONG_VIDEO_THRESHOLD_SEC:
        samples_per_sec = 1  # 1 frame/sec is plenty for a talking-head podcast
    else:
        samples_per_sec = 5

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
    """
    Find the top N highest-scoring non-overlapping segments.

    min_gap_sec: minimum spacing enforced between clip starts, so clips
    from a long podcast don't cluster all in the same few minutes.
    Defaults to clip_len_sec (i.e. clips can't overlap at all).
    """
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
        # Reject if too close to an already-chosen clip
        if any(abs(start - c) < gap_windows for c in chosen_starts):
            continue
        chosen_starts.append(start)

    # If we couldn't find enough non-overlapping segments (short source video),
    # just return what we found
    segments = []
    for start_idx in sorted(chosen_starts):
        # Snap to strongest rising edge nearby, like the single-clip version
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


def cut_clip(video_path, start_time, end_time, output_path, crop_filter=None,
             target_width=1080, target_height=1920, srt_path=None):
    """Cut using ffmpeg with re-encode for frame-accurate trimming.

    -movflags +faststart moves the mp4 index (moov atom) to the front of
    the file. Without this, browsers can't stream/seek the video until
    the whole file downloads.

    If crop_filter is provided, the output is cropped to that region and
    scaled to target_width x target_height (9:16 vertical format).

    If srt_path is provided, captions are burned into the video using
    that subtitle file (already time-shifted to start at 0 for this clip).
    """
    duration = end_time - start_time
    cmd = ["ffmpeg", "-y", "-ss", str(start_time), "-i", video_path, "-t", str(duration)]

    vf_parts = []
    if crop_filter:
        vf_parts.append(crop_filter)
        vf_parts.append(f"scale={target_width}:{target_height}")
    if srt_path and os.path.exists(srt_path):
        # Escape path for ffmpeg filter syntax (colons need escaping on most platforms)
        escaped_srt = srt_path.replace(":", "\\:")
        style = "FontSize=16,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,BorderStyle=1,Outline=2,Alignment=2,MarginV=60"
        vf_parts.append(f"subtitles={escaped_srt}:force_style='{style}'")

    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]

    cmd += ["-c:v", "libx264", "-c:a", "aac", "-preset", "fast",
            "-movflags", "+faststart", output_path]

    subprocess.run(cmd, check=True, capture_output=True)


HOOK_POOLS = {
    "action": [
        "Wait for it...",
        "This is why you don't blink",
        "The moment everything goes wrong",
        "Nobody was ready for this",
        "This escalated fast",
        "The chaos speaks for itself",
        "This is what panic looks like",
    ],
    "dialogue": [
        "This line hits different",
        "The way she said it though...",
        "Nobody expected this response",
        "This confession came out of nowhere",
        "The silence after this line says it all",
        "This is the moment everything changed",
        "Read between the lines on this one",
    ],
    "relatable": [
        "POV: this is literally you",
        "This is way too accurate",
        "Tell me you've felt this without telling me",
        "The way this hit too close to home",
        "This is everyone's villain era starting",
        "Why is this so real though",
        "POV: the moment you realize the truth",
    ],
    "dramatic": [
        "This scene will wreck you",
        "The tension in this moment is unmatched",
        "This is the turning point",
        "Everything builds to this",
        "This is the scene nobody talks about enough",
        "The shift in energy here is insane",
        "This is where it all falls apart",
    ],
}


def classify_vibe(audio_seg, motion_seg):
    """
    Classify a clip segment's 'vibe' from its audio/motion energy signature,
    so we can pick a hook pool that actually fits the footage's energy.
    """
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


def generate_hook(audio_scores, motion_scores, start_time, end_time,
                   window_sec=WINDOW_SEC, transcript=None):
    """
    Pick a hook line for the clip. If a transcript is available and has a
    quotable line within this segment, use that real spoken line (quoted).
    Otherwise fall back to a vibe-matched generic hook.

    Returns (hook_text, vibe_label, is_real_quote).
    """
    start_idx = int(start_time / window_sec)
    end_idx = int(end_time / window_sec)
    audio_seg = audio_scores[start_idx:end_idx]
    motion_seg = motion_scores[start_idx:end_idx]
    vibe = classify_vibe(audio_seg, motion_seg)

    if transcript:
        line = find_best_line(transcript, start_time, end_time)
        if line:
            # Keep it punchy — truncate very long lines rather than showing a paragraph
            words = line.split()
            if len(words) > 16:
                line = " ".join(words[:16]) + "..."
            return f'"{line}"', vibe, True

    hook = random.choice(HOOK_POOLS[vibe])
    return hook, vibe, False


def analyze_and_clip(video_path, output_dir, job_id, clip_len_sec=30,
                      audio_weight=0.5, motion_weight=0.5, n_clips=1,
                      vertical_crop=False, burn_captions=False,
                      progress_callback=None):
    """
    Main entry point. Returns a dict with a 'clips' list (each with its own
    timing, hook, vibe, output path) plus overall video_duration.

    n_clips: how many top non-overlapping clips to extract. Use 1 for
    short trailers, 3-5+ for long podcasts/streams.

    vertical_crop: if True, output clips are cropped to 9:16 (centered on
    detected faces where possible) instead of keeping the source aspect ratio.

    burn_captions: if True, runs Whisper transcription on the source audio
    and (a) uses real spoken lines as hooks where possible, and (b) burns
    matching captions into each output clip.

    progress_callback(stage: str, pct: int) is called at each step, if
    provided, so a caller (e.g. a Flask job-status endpoint) can report
    live progress.
    """
    def report(stage, pct):
        if progress_callback:
            progress_callback(stage, pct)

    report("Reading video info", 5)
    duration = get_video_duration(video_path)
    clip_len_sec = min(clip_len_sec, duration)
    video_width, video_height = get_video_dimensions(video_path)

    report("Extracting audio", 12)
    audio_path = video_path + ".wav"
    extract_audio(video_path, audio_path)

    transcript = None
    if burn_captions:
        report("Transcribing speech", 25)
        transcript = transcribe_video(video_path, audio_path)

    report("Analyzing audio energy", 45)
    audio_scores = audio_energy_timeline(audio_path, duration)
    os.remove(audio_path)

    report("Analyzing motion energy", 65)
    motion_scores = motion_energy_timeline(video_path, duration)

    report("Scoring best segments", 78)
    n = min(len(audio_scores), len(motion_scores))
    audio_scores, motion_scores = audio_scores[:n], motion_scores[:n]

    combined = audio_weight * audio_scores + motion_weight * motion_scores
    segments = find_top_segments(combined, WINDOW_SEC, clip_len_sec, n_clips=n_clips)

    clips = []
    total_segments = len(segments)
    for i, (start_time, end_time) in enumerate(segments):
        pct = 80 + int(15 * (i + 1) / max(1, total_segments))

        crop_filter = None
        if vertical_crop:
            report(f"Finding subject for clip {i + 1}/{total_segments}", pct)
            center_x = find_crop_center_x(video_path, start_time, end_time, video_width)
            crop_filter = build_crop_filter(video_width, video_height, center_x)

        srt_path = None
        if burn_captions and transcript:
            srt_path = os.path.join(output_dir, f"{job_id}_clip{i + 1}.srt")
            write_srt(transcript, start_time, end_time, srt_path)

        report(f"Cutting clip {i + 1}/{total_segments}", pct)
        output_path = os.path.join(output_dir, f"{job_id}_clip{i + 1}.mp4")
        cut_clip(video_path, start_time, end_time, output_path,
                 crop_filter=crop_filter, srt_path=srt_path)

        # Clean up the temp SRT file now that it's burned in — don't need
        # to serve it separately.
        if srt_path and os.path.exists(srt_path):
            os.remove(srt_path)

        hook_text, vibe, is_real_quote = generate_hook(
            audio_scores, motion_scores, start_time, end_time, transcript=transcript
        )

        clips.append({
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
            "duration": round(end_time - start_time, 2),
            "output_path": output_path,
            "filename": os.path.basename(output_path),
            "hook": hook_text,
            "vibe": vibe,
            "hook_is_quote": is_real_quote,
        })

    report("Done", 100)
    return {
        "video_duration": round(duration, 2),
        "clips": clips,
    }
