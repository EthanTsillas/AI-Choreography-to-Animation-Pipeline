# AI Choreography-to-Animation Pipeline

Convert a video of real human movement into a stylized AI-generated animation — automatically, end to end, with no per-frame manual input.

Built as a UCF Senior Design project using AnimateDiff, ControlNet, a custom-trained LoRA character model, and BiRefNet background removal.

---

## Demo

> Drop a GIF or short video clip here once you have output to show.  
> Even a 3–4 second loop of the final animation will get 10x more attention than any description.

---

## How It Works

The pipeline runs in 5 sequential stages:

```
Input Video
    │
    ▼
[1] FFmpeg — Frame Extraction & Preprocessing
    Extract every frame, correct aspect ratio,
    add zero bars if needed, LANCZOS resize to 768x768
    │
    ▼
[2] OpenPose — Skeletal Pose Detection
    Run lllyasviel/ControlNet OpenPose detector
    on each frame to extract body keypoints
    │
    ▼
[3] AnimateDiff + ControlNet — Chunk Inference
    Feed pose frames into AnimateDiffControlNetPipeline
    in chunks of 16 frames to fit 16GB VRAM.
    Custom LoRA fused into pipeline for character identity.
    │
    ▼
[4] BiRefNet — Background Removal
    High-precision segmentation with alpha matting
    to cleanly isolate the character
    │
    ▼
[5] Video-to-Video Upscale
    AnimateDiffVideoToVideoControlNetPipeline
    refines frames from 768x768 to 1024x1024
    │
    ▼
Output Animation (MP4)
```

---

## The Character Model — Custom LoRA Training

The character (Ratman) is a custom LoRA trained from scratch specifically for this pipeline.

**Training setup:**
- **Dataset:** 53 hand-curated character images
- **Captioning:** Automated via BLIP (`Salesforce/blip-image-captioning-base`) — no manual captions written
- **Base model:** `runwayml/stable-diffusion-v1-5`
- **Training script:** `kohya_ss` (`train_network.py`)
- **Optimizer:** AdamW8bit
- **LR Scheduler:** Cosine
- **Mixed precision:** fp16
- **Network dim / alpha:** 64 / 32
- **Resolution:** 768x768 with bucket sizing
- **Epochs:** 10

The trained LoRA is fused directly into the AnimateDiff pipeline before inference so the character identity is locked across every generated frame without per-frame prompting.

---

## Project Structure

```
/
├── main.py                        # Stage 3: AnimateDiff + ControlNet inference
├── upscaler.py                    # Stage 5: Video-to-video upscale pass
├── background_remover.py          # Stage 4: BiRefNet background removal
├── train.py                       # LoRA training — BLIP captioning
├── run_train.py                   # LoRA training — kohya_ss launcher
│
├── FFmpeg/
│   ├── FFmpeg_video_to_frames.py  # Stage 1: Frame extraction + preprocessing
│   ├── FFmpeg_frames_to_video.py  # Final: Stitch frames back to MP4
│   └── videos/
│       └── input.mp4              # Your input video goes here
│
├── Openpose/
│   ├── Openpose.py                # Stage 2: Pose detection
│   └── results/                   # Pose frames saved here
│
├── img/
│   └── 5_ratman/                  # Training images (53 images + .txt captions)
│
├── generated_frames/              # Stage 3 output
├── black_bg_frames/               # Stage 4 output
├── upscaled_frames/               # Stage 5 output
└── videos/
    └── output.mp4                 # Final video
```

---

## Requirements

**Hardware:**
- NVIDIA GPU with 16GB+ VRAM (tested on RTX 4080 / 4090)
- CUDA 12.x

**External dependencies:**
- [FFmpeg](https://ffmpeg.org/download.html) installed and on PATH
- [kohya_ss sd-scripts](https://github.com/kohya-ss/sd-scripts) cloned to `sd-scripts/`

**Python packages:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate controlnet-aux
pip install rembg imageio-ffmpeg pillow
```

**Model files you need to supply** (not included in this repo due to size):
- `M4RV3LSDUNGEONSNEWV40COMICS_mD40.safetensors` — base Stable Diffusion checkpoint
- `Ratman_v1.safetensors` — trained LoRA output from `run_train.py`
- `easynegative.safetensors` — textual inversion embedding
- `verybadimagenegative_v1.3.pt` — textual inversion embedding

---

## Usage

### Step 1 — Train your character LoRA

Place your character images in `img/your_character/` then run BLIP captioning:

```bash
python train.py
```

This auto-generates a `.txt` caption file next to every image.

Then launch LoRA training:

```bash
python run_train.py
```

Output: `model/Ratman_v1.safetensors`

---

### Step 2 — Prepare your input video

Place your video at `FFmpeg/videos/input.mp4`.

Then extract and preprocess frames (run once):

```python
from FFmpeg.FFmpeg_video_to_frames import get_frames
get_frames("FFmpeg/videos/input.mp4")
```

Then run OpenPose:

```python
from Openpose.Openpose import run_openpose
run_openpose()
```

---

### Step 3 — Run the main generation pipeline

```bash
python main.py
```

This runs AnimateDiff + ControlNet inference in chunks of 16 frames, saving output to `generated_frames/`.

---

### Step 4 — Background removal (optional)

```bash
python background_remover.py
```

Runs BiRefNet with alpha matting. Output saved to `black_bg_frames/` and stitched to video automatically.

---

### Step 5 — Upscale pass

```bash
python upscaler.py
```

Runs the video-to-video pipeline at 1024x1024. Output saved to `upscaled_frames/` and stitched to video.

---

## Key Technical Details

**16-frame chunking:** AnimateDiff generates temporally consistent motion within a chunk. Processing in chunks of 16 is required to fit within 16GB VRAM while maintaining motion coherence.

**VAE slicing and tiling:** Enabled on the VAE to handle 768x768 and 1024x1024 resolutions without OOM errors on consumer GPUs.

**LoRA fusion:** The character LoRA is fused into the pipeline weights before inference (`pipe.fuse_lora()`) rather than applied at runtime, which improves inference speed and ensures consistent weight application across all chunks.

**Two-pass architecture:** The first pass (main.py) generates at 768x768 for speed. The second pass (upscaler.py) uses video-to-video at strength=0.35 — high enough to sharpen detail, low enough to preserve the motion from pass one.

**BLIP captioning:** Rather than writing training captions by hand, BLIP generates a natural language description of each image automatically. A fixed prefix (`ratman, `) is prepended so the trigger word is always present in every caption.

---

## Results

| Stage | Resolution | Frames |
|---|---|---|
| Generation (main.py) | 768x768 | 16 per chunk |
| Upscale (upscaler.py) | 1024x1024 | 16 per chunk |
| Final video | 1024x1024 | Full sequence @ 24fps |

---

## Built With

- [Diffusers](https://github.com/huggingface/diffusers) — AnimateDiff, ControlNet, pipeline management
- [kohya_ss](https://github.com/kohya-ss/sd-scripts) — LoRA training
- [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) — automated image captioning
- [ControlNet OpenPose](https://huggingface.co/lllyasviel/control_v11p_sd15_openpose) — skeleton detection
- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) via rembg — background removal
- [FFmpeg](https://ffmpeg.org) — frame extraction and video assembly

---

## Author

**Ethan Tsillas**  
github.com/EthanTsillas  
linkedin.com/in/ethan-tsillas
