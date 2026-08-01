"""
analyzer.py
Core logic for finding the "best hook" segment in a video using
audio energy + motion energy, then cutting it with ffmpeg.

No AI/ML models required — pure signal analysis, so it's fast and cheap
to run on a small VPS.
"""

import subprocess
import numpy as np
import librosa
import cv2
import os
import json

WINDOW_SEC = 0.5  # resolution of the excitement timeline


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
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    sample_every_n_frames = max(1, int(fps * 0.2))  # sample ~5 frames/sec

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


def find_best_segment(combined_score, window_sec, clip_len_sec, hook_boost_sec=2.0):
    """
    Slide a window of clip_len_sec across the combined excitement score
    to find the highest-total-score contiguous segment.

    Then nudge the start point to a local rising edge within the first
    couple seconds so the clip doesn't start mid-lull.
    """
    n_windows = len(combined_score)
    clip_windows = max(1, int(round(clip_len_sec / window_sec)))

    if clip_windows >= n_windows:
        return 0.0, n_windows * window_sec

    # cumulative sum for fast window-sum lookups
    cumsum = np.cumsum(np.insert(combined_score, 0, 0))
    best_start_idx = 0
    best_sum = -1

    for start in range(0, n_windows - clip_windows + 1):
        window_sum = cumsum[start + clip_windows] - cumsum[start]
        if window_sum > best_sum:
            best_sum = window_sum
            best_start_idx = start

    # Snap start to the strongest rising edge within hook_boost_sec of the window start
    boost_windows = int(hook_boost_sec / window_sec)
    search_end = min(best_start_idx + boost_windows, n_windows - 1)
    best_rise_idx = best_start_idx
    best_rise_val = -1
    for i in range(best_start_idx, search_end):
        if i + 1 < n_windows:
            rise = combined_score[i + 1] - combined_score[i]
            if rise > best_rise_val:
                best_rise_val = rise
                best_rise_idx = i

    start_time = best_rise_idx * window_sec
    end_time = start_time + clip_len_sec
    return start_time, end_time


def cut_clip(video_path, start_time, end_time, output_path):
    """Cut using ffmpeg with re-encode for frame-accurate trimming."""
    duration = end_time - start_time
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start_time), "-i", video_path,
         "-t", str(duration), "-c:v", "libx264", "-c:a", "aac",
         "-preset", "fast", output_path],
        check=True, capture_output=True
    )


def analyze_and_clip(video_path, output_path, clip_len_sec=30,
                      audio_weight=0.5, motion_weight=0.5):
    """
    Main entry point. Returns dict with timing info + output path.
    """
    duration = get_video_duration(video_path)
    clip_len_sec = min(clip_len_sec, duration)

    audio_path = video_path + ".wav"
    extract_audio(video_path, audio_path)

    audio_scores = audio_energy_timeline(audio_path, duration)
    motion_scores = motion_energy_timeline(video_path, duration)

    os.remove(audio_path)

    # align lengths (audio/motion window counts should match, but just in case)
    n = min(len(audio_scores), len(motion_scores))
    audio_scores, motion_scores = audio_scores[:n], motion_scores[:n]

    combined = audio_weight * audio_scores + motion_weight * motion_scores

    start_time, end_time = find_best_segment(combined, WINDOW_SEC, clip_len_sec)
    cut_clip(video_path, start_time, end_time, output_path)

    return {
        "start_time": round(start_time, 2),
        "end_time": round(end_time, 2),
        "duration": round(end_time - start_time, 2),
        "video_duration": round(duration, 2),
        "output_path": output_path,
    }
