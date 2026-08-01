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

WINDOW_SEC = 0.5  # resolution of the excitement timeline

# Videos longer than this get sparser frame sampling for motion analysis,
# so a 2-hour podcast doesn't take forever to process even on strong hardware.
LONG_VIDEO_THRESHOLD_SEC = 20 * 60


def get_video_duration(video_path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", video_path],
        capture_output=True, text=True
    )
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


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


def cut_clip(video_path, start_time, end_time, output_path):
    """Cut using ffmpeg with re-encode for frame-accurate trimming.

    -movflags +faststart moves the mp4 index (moov atom) to the front of
    the file. Without this, browsers can't stream/seek the video until
    the whole file downloads — causing a "only plays a few seconds,
    then dumps the whole download at once" symptom.
    """
    duration = end_time - start_time
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start_time), "-i", video_path,
         "-t", str(duration), "-c:v", "libx264", "-c:a", "aac",
         "-preset", "fast", "-movflags", "+faststart", output_path],
        check=True, capture_output=True
    )


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


def generate_hook(audio_scores, motion_scores, start_time, end_time, window_sec=WINDOW_SEC):
    """
    Pick a fully-written hook line whose vibe matches the selected clip segment.
    Returns (hook_text, vibe_label).
    """
    start_idx = int(start_time / window_sec)
    end_idx = int(end_time / window_sec)
    audio_seg = audio_scores[start_idx:end_idx]
    motion_seg = motion_scores[start_idx:end_idx]

    vibe = classify_vibe(audio_seg, motion_seg)
    hook = random.choice(HOOK_POOLS[vibe])
    return hook, vibe


def analyze_and_clip(video_path, output_dir, job_id, clip_len_sec=30,
                      audio_weight=0.5, motion_weight=0.5, n_clips=1,
                      progress_callback=None):
    """
    Main entry point. Returns a dict with a 'clips' list (each with its own
    timing, hook, vibe, output path) plus overall video_duration.

    n_clips: how many top non-overlapping clips to extract. Use 1 for
    short trailers, 3-5+ for long podcasts/streams.

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

    report("Extracting audio", 15)
    audio_path = video_path + ".wav"
    extract_audio(video_path, audio_path)

    report("Analyzing audio energy", 35)
    audio_scores = audio_energy_timeline(audio_path, duration)
    os.remove(audio_path)

    report("Analyzing motion energy", 65)
    motion_scores = motion_energy_timeline(video_path, duration)

    report("Scoring best segments", 80)
    n = min(len(audio_scores), len(motion_scores))
    audio_scores, motion_scores = audio_scores[:n], motion_scores[:n]

    combined = audio_weight * audio_scores + motion_weight * motion_scores
    segments = find_top_segments(combined, WINDOW_SEC, clip_len_sec, n_clips=n_clips)

    clips = []
    total_segments = len(segments)
    for i, (start_time, end_time) in enumerate(segments):
        pct = 85 + int(10 * (i + 1) / max(1, total_segments))
        report(f"Cutting clip {i + 1}/{total_segments}", pct)

        output_path = os.path.join(output_dir, f"{job_id}_clip{i + 1}.mp4")
        cut_clip(video_path, start_time, end_time, output_path)

        hook_text, vibe = generate_hook(audio_scores, motion_scores, start_time, end_time)

        clips.append({
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
            "duration": round(end_time - start_time, 2),
            "output_path": output_path,
            "filename": os.path.basename(output_path),
            "hook": hook_text,
            "vibe": vibe,
        })

    report("Done", 100)
    return {
        "video_duration": round(duration, 2),
        "clips": clips,
    }
