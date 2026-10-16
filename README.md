# Video Processing & Subtitle/Watermark Removal Suite

This repository contains scripts designed for Linux video processing workflows:
1. [`split_scenes.py`](file:///Users/zaid/Documents/CollegeGPU/split_scenes.py): Fast, accurate scene detection and splitting into sub-videos every time the scene changes.
2. [`process_videos_comfyui.py`](file:///Users/zaid/Documents/CollegeGPU/process_videos_comfyui.py): End-to-end automated batch processing pipeline for ComfyUI Subtitle/Watermark removal (`Reddit-removal.json`).
3. [`process_character_swap_comfyui.py`](file:///Users/zaid/Documents/CollegeGPU/process_character_swap_comfyui.py): End-to-end automated character swap pipeline for ComfyUI (`working_character_swap.json`) with tuned ~5s scene splitting.

4. [`generate_asmr_videos_minimax.py`](file:///Users/zaid/Documents/CollegeGPU/generate_asmr_videos_minimax.py): Automated batch generation of ASMR videos using MiniMax H3 (`video_minimax_h3_t2v.json`) and Hermes Agent LLM for detailed prompt craft.

---

## 🛠️ System Prerequisites (Linux / Ubuntu)

```bash
# 1. System packages
sudo apt update && sudo apt install -y ffmpeg

# 2. Python packages
pip install scenedetect[opencv] requests websocket-client tqdm pyyaml
```

---

## 🎬 1. Scene-Based Video Splitter (`split_scenes.py`)

Splits any full video into clean sub-videos whenever a scene change occurs. Also extracts the first frame of each scene.

### Features
* **PySceneDetect & FFmpeg fallbacks**: Uses `scenedetect` (Content/Adaptive detectors) with a fallback to native FFmpeg scene filters.
* **Max duration capping (`--max-duration 5.0`)**: If a single scene is long, it automatically sub-splits it into $\le 5$s chunks so it's ready for diffusion models (LTX-Video).
* **Min duration filter (`--min-duration 1.0`)**: Prevents noisy micro-cuts.
* **1st Frame Extraction**: Exports the starting keyframe for each scene (`scene_0000_frame0.png`).
* **Metadata & Re-stitch List**: Produces `scenes_metadata.json` and `concat_list.txt`.

### Usage
```bash
# Process a single video
python split_scenes.py -i my_video.mp4 -o ./scene_outputs

# Process all videos in a folder
python split_scenes.py -i ./input_videos -o ./scene_outputs --max-duration 5.0 --detector content
```

---

## 🤖 2. ComfyUI Video Inpainting Pipeline (`process_videos_comfyui.py`)

Automates the subtitle/watermark removal workflow using ComfyUI API (`Reddit-removal.json`):
1. Splits raw video into $\le 5$ second chunks and extracts first frame.
2. Feeds chunks into Flux 2 inpainting + MoGe depth estimation + LTX-Video 2.3.
3. Downloads and concatenates all sub-videos with synchronized audio.

### Usage
```bash
python process_videos_comfyui.py -i ./input_videos -o ./output_videos
```

---

## 🎭 3. ComfyUI Character Swap Pipeline (`process_character_swap_comfyui.py`)

Automates the MiniMax H3 Ref2VA + SAM3 character swap workflow (`working_character_swap2.json` / `working_character_swap.json`):
1. Splits video into **~5 second sub-videos** using less-sensitive scene cut detection (`--scene-threshold 40.0`, `--min-scene-duration 3.5s`).
2. Supports **random selection of character reference images** per video from a folder (`--ref-images-dir` / `--ref-folder`).
3. Supports single reference image (`--ref-image`), tracking prompt (`--track-prompt`), and custom edit prompts (`--prompt`).
4. Handles both ComfyUI API prompt JSON format and UI Graph JSON formats automatically.
5. Downloads and concatenates all character-swapped sub-videos into the final output.

### Usage
```bash
# Batch process with random character selection from an image folder per video
python process_character_swap_comfyui.py \
  -i ./input_videos \
  -o ./output_videos \
  --ref-images-dir ./ref_characters \
  --track-prompt "people"

# Process a single video with a specific reference character image
python process_character_swap_comfyui.py \
  -i my_video.mp4 \
  -o ./output_videos \
  --ref-image /path/to/character.png \
  --track-prompt "people"
```

---

## 🎧 4. MiniMax H3 Batch ASMR Generator (`generate_asmr_videos_minimax.py`)

Automates batch video creation with **MiniMax H3 (`video_minimax_h3_t2v.json`)** by querying the **Hermes Agent** model to generate a rich, distinct cinematic text prompt for each video:
1. **Interactive or CLI inputs**: Asks for the **number of videos** (e.g. `100`) and **duration/length in seconds** (e.g. `5.0`) upfront before generating anything.
2. **Hermes Agent Prompt Engine**: Calls the local Hermes Agent (`hermes -z` using `nvidia/nemotron-3-ultra-550b-a55b` or configured model) to craft unique, timestamped, sensory ASMR prompts with camera cues and binaural audio specifications.
3. **ComfyUI Automation**: Dynamically sets prompt (`Node 105:104`), duration (`Node 105:111`), noise seed (`Node 105:15`), aspect ratio (`Node 115`), and filename prefix (`Node 92`).
4. **Live WebSocket Monitoring**: Shows real-time node execution progress with ping keep-alives and downloads each completed MP4 file.
5. **Manifest Generation**: Generates `generation_manifest.json` and `generation_manifest.md` with prompt records and metadata.

### Usage
```bash
# Interactive mode (prompts for count, duration, and theme before starting)
python generate_asmr_videos_minimax.py

# CLI batch generation (e.g., 100 ASMR videos, 5 seconds each)
python generate_asmr_videos_minimax.py -n 100 -l 5.0 -t "Satisfying ASMR" -o ./asmr_outputs

# Vertical format for Reels / Shorts / TikTok (9:16)
python generate_asmr_videos_minimax.py -n 50 -l 6.0 --aspect-ratio "9:16 (Vertical)"

# Remote GPU Execution (RTX 6000 Ada Ubuntu Server)
python generate_asmr_videos_minimax.py \
  --server 192.168.1.50:8188 \
  -n 100 \
  -l 5.0 \
  -o ./asmr_outputs
```

### 🚀 Running on a Remote Ubuntu Machine (RTX 6000 Ada GPU)

#### 1. On the Ubuntu GPU Host (RTX 6000 Ada):
Start ComfyUI so it listens on the network (or on localhost if using SSH port forwarding):
```bash
# Launch ComfyUI accepting external connections
python main.py --listen 0.0.0.0 --port 8188
```

#### 2. From Your Client Machine (Laptop / Workstation):
Point the script directly to your Ubuntu GPU machine using `--server`:
```bash
python generate_asmr_videos_minimax.py \
  --server <UBUNTU_GPU_IP>:8188 \
  -n 100 \
  -l 5.0 \
  -o ./asmr_outputs
```
*The script will connect, report the detected GPU hardware (`NVIDIA RTX 6000 Ada Generation | 48.0 GB free VRAM`), run prompts through Hermes locally, submit workflows to the remote GPU, monitor progress via WebSocket, and automatically download every rendered MP4 to your local `./asmr_outputs`.*

#### 3. Or Running Directly ON the Ubuntu System:
If you copy this script onto the Ubuntu server itself:
```bash
# 1. Install dependencies
pip install requests websocket-client tqdm pyyaml

# 2. Export your OpenRouter key (if Hermes CLI is not installed on that server)
export OPENROUTER_API_KEY="your-openrouter-key"

# 3. Run the generator locally against ComfyUI
python generate_asmr_videos_minimax.py \
  --server 127.0.0.1:8188 \
  -n 100 \
  -l 5.0 \
  -o ./asmr_outputs
```



