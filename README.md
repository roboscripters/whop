# Whop Clipper — Auto Hook Finder

Upload a trailer, get back the most exciting 15–60s segment, auto-cut.
Uses audio energy (librosa) + motion energy (OpenCV frame diffing) —
no AI model calls, so it's fast and free to run.

## How it picks the clip
1. Splits the video into 0.5s windows
2. Scores each window on loudness (audio RMS) and visual movement (frame diff)
3. Combines both into one "excitement" timeline
4. Slides a window of your chosen length across that timeline to find the
   highest-total-score segment
5. Snaps the start point to the sharpest rising edge nearby, so the clip
   doesn't start mid-lull
6. Cuts it with ffmpeg

## Deploying on a Google Cloud free-trial VPS (all from your phone browser)

### 1. Create the VPS
- Go to console.cloud.google.com in your phone browser, sign up for the free trial ($300 credit)
- Compute Engine → Create Instance
- Choose **e2-small** or **e2-medium** (2 vCPU / 2-4GB RAM — video processing needs some headroom)
- Boot disk: Ubuntu 22.04 LTS, at least 20GB
- Allow HTTP/HTTPS traffic in the firewall settings
- Create it

### 2. Open the browser-based SSH terminal
On the VM's row in the console, tap the **SSH** button — this opens a full terminal in your browser, no app needed.

### 3. Install dependencies
Run these in the SSH terminal:
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv ffmpeg git
```

### 4. Get the code onto the VM
Push this project to a GitHub repo (same GitHub browser workflow you already use for TradeBotX), then on the VM:
```bash
git clone https://github.com/YOUR_USERNAME/whop-clipper.git
cd whop-clipper
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 5. Run it (quick test)
```bash
python3 app.py
```
It'll listen on port 8080. To check it's alive, open a new SSH tab and run:
```bash
curl localhost:8080
```

### 6. Open the firewall for your app port
In Google Cloud Console → VPC Network → Firewall → Create Firewall Rule:
- Targets: All instances
- Source IP: 0.0.0.0/0
- Protocol/port: tcp:8080

Then visit `http://YOUR_VM_EXTERNAL_IP:8080` from your phone.

### 7. Keep it running permanently (production)
Don't rely on the SSH session staying open — set up a systemd service so it survives reboots and restarts if it crashes:
```bash
sudo tee /etc/systemd/system/whop-clipper.service << 'EOF'
[Unit]
Description=Whop Clipper
After=network.target

[Service]
User=root
WorkingDirectory=/root/whop-clipper
ExecStart=/root/whop-clipper/venv/bin/gunicorn -w 2 -t 300 -b 0.0.0.0:8080 app:app
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable whop-clipper
sudo systemctl start whop-clipper
```
Check it's running: `sudo systemctl status whop-clipper`

Note: gunicorn's `-t 300` gives each request 5 minutes before timing out —
video analysis on longer trailers can take a bit, so this prevents premature
cutoffs.

## Files
- `app.py` — Flask server (upload endpoint + download endpoint)
- `analyzer.py` — the actual scoring/cutting logic
- `templates/index.html` — mobile upload page
- `requirements.txt` — Python deps

## Tuning
In `analyzer.py`, `analyze_and_clip()` takes `audio_weight` and `motion_weight`
(default 0.5/0.5). If your trailers are more about visual action, bump
`motion_weight` to 0.7. If they're more about VO/music stingers, bump
`audio_weight` instead.

## Known limitations (v1)
- No speech/dialogue awareness — it can't tell if a quiet line is a killer
  one-liner. If you want that later, we can add Whisper transcription and
  score sentences for "hook-worthy" phrasing.
- Single best segment per video — doesn't yet return multiple candidate
  clips. Easy to extend if you want 3 options per trailer instead of 1.
- No auth on the upload endpoint — fine for personal/testing use, but add
  a simple password check before sharing the link with a team.
