# Deploying Whop Clipper on Oracle Cloud (Always Free tier)

## 1. Create the instance
- Go to cloud.oracle.com in your phone browser, sign in
- Menu → Compute → Instances → Create Instance
- Name it (e.g. `whop-clipper`)
- **Image and shape** → Edit:
  - Image: **Ubuntu 22.04**
  - Shape: click "Change shape" → **Ampere** → **VM.Standard.A1.Flex**
    - This is the free-tier ARM shape. Set 4 OCPU / 24GB memory (max free allowance)
    - If Ampere capacity isn't available in your region, fall back to shape **VM.Standard.E2.1.Micro** (also Always Free, x86, 1GB RAM — works fine for a few videos/day, just slower)
- **Networking**: leave default VCN, make sure "Assign a public IPv4 address" is checked
- **Add SSH key**: choose "Generate a key pair for me" and **download the private key** when prompted — you'll need it to SSH in (unless you use the browser Cloud Shell method below, which skips this)
- Create

## 2. Connect to the instance
Easiest mobile-friendly option — Oracle has a browser-based terminal:
- On the instance page, tap **Connect** → or open **Cloud Shell** from the top menu bar (the `>_` icon)
- From Cloud Shell, SSH into your instance:
```bash
ssh -i /path/to/your/downloaded/key ubuntu@YOUR_INSTANCE_PUBLIC_IP
```
(Cloud Shell has its own file upload button if you need to get the key file into it — or just generate the key pair *inside* Cloud Shell instead and paste the public key into the instance creation screen.)

## 3. Install dependencies
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv ffmpeg git
```

## 4. Get the code
Push the project to GitHub first (same flow as your other repos), then:
```bash
git clone https://github.com/YOUR_USERNAME/whop-clipper.git
cd whop-clipper
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
> Note: if you're on the Ampere (ARM) shape, `opencv-python-headless` and `librosa` both have prebuilt ARM wheels on PyPI, so this should install cleanly. If pip tries to build from source and it's slow/fails, run `pip install --upgrade pip` first and retry.

## 5. Open the firewall — BOTH layers

### Layer 1: Oracle's Security List (cloud-side)
- Go to your instance's VCN → Subnet → Security List
- Add Ingress Rule:
  - Source CIDR: `0.0.0.0/0`
  - IP Protocol: TCP
  - Destination Port Range: `8080`

### Layer 2: The instance's own firewall (iptables, on by default on Oracle Ubuntu images)
SSH'd into the instance:
```bash
sudo iptables -I INPUT -p tcp --dport 8080 -j ACCEPT
sudo netfilter-persistent save
```
(If `netfilter-persistent` isn't found: `sudo apt install -y iptables-persistent` then rerun the save command.)

**This second layer is the #1 reason people get "connection refused" on Oracle** even after opening the Security List — don't skip it.

## 6. Run it as a permanent service
```bash
sudo tee /etc/systemd/system/whop-clipper.service << 'EOF'
[Unit]
Description=Whop Clipper
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/whop-clipper
ExecStart=/home/ubuntu/whop-clipper/venv/bin/gunicorn -w 2 -t 300 -b 0.0.0.0:8080 app:app
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable whop-clipper
sudo systemctl start whop-clipper
sudo systemctl status whop-clipper
```
(Adjust `WorkingDirectory`/paths if you cloned somewhere other than `/home/ubuntu/whop-clipper`.)

## 7. Test it
From your phone, visit:
```
http://YOUR_INSTANCE_PUBLIC_IP:8080
```
Upload a short test trailer and confirm you get a clip back.

## Troubleshooting
- **Site won't load at all** → check Security List rule (step 5, layer 1) and iptables rule (layer 1, layer 2) — 90% of Oracle connection issues are one of these two
- **Service won't start** → `sudo journalctl -u whop-clipper -n 50` to see the error
- **Upload times out on longer videos** → the gunicorn `-t 300` gives 5 minutes per request; bump it higher (e.g. `-t 600`) in the service file if trailers are long, then `sudo systemctl daemon-reload && sudo systemctl restart whop-clipper`
- **Out of disk space** → uploaded videos get deleted after processing, but `outputs/` accumulates clips over time; periodically clear old ones: `rm outputs/*.mp4`
