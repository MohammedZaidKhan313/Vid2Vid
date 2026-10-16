#!/usr/bin/env python3
"""
=============================================================================
ComfyUI Automated Video Batch Processing & Sub-video Concatenation Pipeline
=============================================================================

Features:
1. Flexible Input Modes:
   - Mode A (Raw Videos): Scans input folder, splits each video by Scene Cuts
     (via PySceneDetect / FFmpeg) OR fixed time chunks (<= 5s), processes via
     ComfyUI API, and stitches them back together.
   - Mode B (Pre-Split Scene Folders): Directly consumes output folders from
     `split_scenes.py` (containing scene_0000.mp4, scene_0001.mp4...), processes
     each scene through ComfyUI, and re-stitches each video into a final video.
2. Injects sub-videos into ComfyUI workflow (`Reddit-removal.json`).
3. Monitors live progress via WebSocket (with HTTP polling fallback).
4. Downloads cleaned sub-videos in chronological order.
5. Concatenates all sub-videos into a single final video preserving audio sync.

Requirements:
  - Python 3.8+
  - ffmpeg & ffprobe installed (`sudo apt install ffmpeg` on Ubuntu/Debian)
  - Optional pip packages: `pip install requests websocket-client tqdm scenedetect[opencv]`
=============================================================================
"""

import os
import sys
import json
import time
import uuid
import random
import shutil
import argparse
import subprocess
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Optional enhanced dependencies
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from scenedetect import open_video, SceneManager, ContentDetector, AdaptiveDetector
    HAS_SCENEDETECT = True
except ImportError:
    HAS_SCENEDETECT = False


class ComfyUIPipeline:
    def __init__(
        self,
        server_address: str = "127.0.0.1:8188",
        workflow_path: str = "Reddit-removal.json",
        split_mode: str = "scene",
        chunk_duration: float = 5.0,
        scene_threshold: float = 27.0,
        min_scene_duration: float = 1.0,
        output_dir: str = "./output_videos",
        temp_dir: str = "./temp_chunks",
        keep_temp: bool = False,
        random_seed: bool = True,
        enable_llm_prompt: bool = False,
        prompt_override: Optional[str] = None
    ):
        # Robust server address parsing (handles http://, https://, ws://, wss://, host:port, trailing slashes)
        raw_server = server_address.strip().rstrip("/")
        if raw_server.startswith("https://"):
            host_port = raw_server[8:].rstrip("/")
            use_ssl = True
        elif raw_server.startswith("http://"):
            host_port = raw_server[7:].rstrip("/")
            use_ssl = False
        elif raw_server.startswith("wss://"):
            host_port = raw_server[6:].rstrip("/").rstrip("/ws")
            use_ssl = True
        elif raw_server.startswith("ws://"):
            host_port = raw_server[5:].rstrip("/").rstrip("/ws")
            use_ssl = False
        else:
            host_port = raw_server
            use_ssl = False

        self.server_address = host_port
        http_proto = "https" if use_ssl else "http"
        ws_proto = "wss" if use_ssl else "ws"
        self.http_base = f"{http_proto}://{self.server_address}"
        self.ws_base = f"{ws_proto}://{self.server_address}/ws"
        self.workflow_path = Path(workflow_path)
        self.split_mode = split_mode.lower()
        self.chunk_duration = chunk_duration
        self.scene_threshold = scene_threshold
        self.min_scene_duration = min_scene_duration
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.keep_temp = keep_temp
        self.random_seed = random_seed
        self.enable_llm_prompt = enable_llm_prompt
        self.prompt_override = prompt_override
        self.client_id = str(uuid.uuid4())

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self._check_dependencies()
        self._load_workflow()
        self._check_comfyui_health()

    # --- HTTP Helpers (with urllib fallback) ---

    def _http_get_json(self, endpoint: str, timeout: int = 10) -> Tuple[int, Any]:
        url = f"{self.http_base}{endpoint}"
        if HAS_REQUESTS:
            try:
                r = requests.get(url, timeout=timeout)
                return r.status_code, r.json() if r.status_code == 200 else r.text
            except Exception as e:
                raise ConnectionError(f"HTTP GET failed to {url}: {e}")
        else:
            req = urllib.request.Request(url, headers={"User-Agent": "ComfyUIPipeline/1.0"})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    status = response.getcode()
                    content = response.read().decode("utf-8")
                    return status, json.loads(content)
            except urllib.error.HTTPError as e:
                return e.code, e.read().decode("utf-8", errors="ignore")
            except Exception as e:
                raise ConnectionError(f"HTTP GET failed to {url}: {e}")

    def _http_post_json(self, endpoint: str, payload: Dict[str, Any], timeout: int = 15) -> Tuple[int, Any]:
        url = f"{self.http_base}{endpoint}"
        if HAS_REQUESTS:
            try:
                r = requests.post(url, json=payload, timeout=timeout)
                return r.status_code, r.json() if r.status_code == 200 else r.text
            except Exception as e:
                raise ConnectionError(f"HTTP POST failed to {url}: {e}")
        else:
            data_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data_bytes,
                headers={"Content-Type": "application/json", "User-Agent": "ComfyUIPipeline/1.0"}
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    status = response.getcode()
                    content = response.read().decode("utf-8")
                    return status, json.loads(content)
            except urllib.error.HTTPError as e:
                return e.code, e.read().decode("utf-8", errors="ignore")
            except Exception as e:
                raise ConnectionError(f"HTTP POST failed to {url}: {e}")

    def _check_dependencies(self):
        """Check if ffmpeg and ffprobe are available."""
        for tool in ["ffmpeg", "ffprobe"]:
            if shutil.which(tool) is None:
                raise SystemError(
                    f"Required system tool '{tool}' is not installed or not in PATH. "
                    "Install it using: sudo apt update && sudo apt install -y ffmpeg"
                )

    def _check_comfyui_health(self):
        """Check if ComfyUI server is reachable."""
        try:
            status, res = self._http_get_json("/system_stats", timeout=5)
            if status == 200 and isinstance(res, dict):
                devices = res.get("devices", [])
                gpu_info = devices[0].get("name", "Unknown GPU") if devices else "CPU"
                print(f"[✓] Connected to ComfyUI at {self.http_base} (Device: {gpu_info})")
            else:
                print(f"[!] Warning: ComfyUI returned status {status}")
        except Exception as e:
            raise ConnectionError(
                f"Could not connect to ComfyUI at {self.http_base}. "
                f"Please ensure ComfyUI is running on port 8188.\nDetails: {e}"
            )

    def _load_workflow(self):
        """Load and validate the workflow JSON file."""
        if not self.workflow_path.exists():
            raise FileNotFoundError(f"Workflow file not found: {self.workflow_path}")
        with open(self.workflow_path, "r", encoding="utf-8") as f:
            self.base_workflow = json.load(f)
        print(f"[✓] Loaded base workflow: {self.workflow_path} ({len(self.base_workflow)} nodes)")

    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get video duration and specs using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,duration:format=duration",
            "-of", "json",
            str(video_path)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"ffprobe error on {video_path}: {res.stderr}")

        data = json.loads(res.stdout)
        format_info = data.get("format", {})
        streams = data.get("streams", [{}])
        v_stream = streams[0] if streams else {}

        duration = float(format_info.get("duration", v_stream.get("duration", 0.0)))
        r_fps = v_stream.get("r_frame_rate", "30/1")
        if "/" in r_fps:
            num, den = map(float, r_fps.split("/"))
            fps = num / den if den != 0 else 30.0
        else:
            fps = float(r_fps)

        return {"duration": duration, "fps": fps}

    def detect_scene_cuts(self, video_path: Path, video_info: Dict[str, Any]) -> List[Tuple[float, float]]:
        """Detect scenes using PySceneDetect or FFmpeg fallback."""
        if HAS_SCENEDETECT:
            video = open_video(str(video_path))
            scene_manager = SceneManager()
            min_frames = max(1, int(self.min_scene_duration * video_info["fps"]))
            scene_manager.add_detector(ContentDetector(threshold=self.scene_threshold, min_scene_len=min_frames))
            scene_manager.detect_scenes(video, show_progress=False)
            scenes = scene_manager.get_scene_list()
            raw_cuts = [(s.get_seconds(), e.get_seconds()) for s, e in scenes]
            if raw_cuts:
                return raw_cuts

        # Fallback to FFmpeg scene filter
        norm_thresh = min(max(self.scene_threshold / 100.0, 0.1), 0.9)
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-filter_complex", f"select='gt(scene,{norm_thresh:.2f})',metadata=print:file=-",
            "-f", "null", "-"
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        timestamps = [0.0]
        for line in res.stdout.splitlines():
            if "pts_time:" in line:
                try:
                    ts = float(line.split("pts_time:")[1].strip())
                    if ts - timestamps[-1] >= self.min_scene_duration:
                        timestamps.append(ts)
                except ValueError:
                    continue

        total_dur = video_info["duration"]
        if timestamps[-1] < total_dur:
            timestamps.append(total_dur)

        return [(timestamps[i], timestamps[i+1]) for i in range(len(timestamps)-1)]

    def split_video_into_chunks(self, video_path: Path, work_dir: Path) -> List[Dict[str, Any]]:
        """
        Split a video into sub-videos using Scene Cuts or Fixed Chunks,
        capping any long scene at <= self.chunk_duration (e.g. 5s).
        """
        info = self.get_video_info(video_path)
        total_duration = info["duration"]
        cut_pairs = []

        if self.split_mode == "scene":
            print(f"[*] Splitting '{video_path.name}' by SCENE CHANGES (capped at <= {self.chunk_duration}s)...")
            raw_scenes = self.detect_scene_cuts(video_path, info)
            # Cap each scene at chunk_duration
            for s, e in raw_scenes:
                curr_s = s
                while curr_s < e:
                    curr_e = min(curr_s + self.chunk_duration, e)
                    cut_pairs.append((curr_s, curr_e))
                    curr_s = curr_e
        else:
            print(f"[*] Splitting '{video_path.name}' into FIXED <= {self.chunk_duration}s chunks...")
            s = 0.0
            while s < total_duration:
                e = min(s + self.chunk_duration, total_duration)
                cut_pairs.append((s, e))
                s = e

        chunks = []
        for idx, (start_t, end_t) in enumerate(cut_pairs):
            duration = end_t - start_t
            chunk_video_name = f"chunk_{idx:04d}.mp4"
            chunk_frame_name = f"chunk_{idx:04d}_frame0.png"
            chunk_video_path = work_dir / chunk_video_name
            chunk_frame_path = work_dir / chunk_frame_name

            # 1. Cut video chunk
            cmd_cut = [
                "ffmpeg", "-y",
                "-ss", f"{start_t:.3f}",
                "-i", str(video_path),
                "-t", f"{duration:.3f}",
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "fast",
                "-c:a", "aac",
                "-b:a", "192k",
                "-avoid_negative_ts", "make_zero",
                str(chunk_video_path)
            ]
            subprocess.run(cmd_cut, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

            # 2. Extract first frame
            cmd_frame = [
                "ffmpeg", "-y",
                "-ss", "0.0",
                "-i", str(chunk_video_path),
                "-vframes", "1",
                "-q:v", "2",
                str(chunk_frame_path)
            ]
            subprocess.run(cmd_frame, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

            chunks.append({
                "index": idx,
                "start_time": start_t,
                "duration": duration,
                "video_path": chunk_video_path,
                "frame_path": chunk_frame_path,
                "video_name": chunk_video_name,
                "frame_name": chunk_frame_name
            })

        print(f"[✓] Created {len(chunks)} sub-videos in {work_dir}")
        return chunks

    def upload_file_to_comfyui(self, file_path: Path, subfolder: str = "") -> str:
        """Upload a file to ComfyUI input directory."""
        url = f"{self.http_base}/upload/image"
        if HAS_REQUESTS:
            with open(file_path, "rb") as f:
                files = {"image": (file_path.name, f)}
                data = {"overwrite": "true", "subfolder": subfolder, "type": "input"}
                r = requests.post(url, files=files, data=data)
                if r.status_code != 200:
                    raise RuntimeError(f"Failed to upload {file_path.name}: {r.text}")
                res = r.json()
                uploaded_name = res.get("name", file_path.name)
                return f"{subfolder}/{uploaded_name}" if subfolder else uploaded_name
        else:
            boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
            body = bytearray()

            def add_field(name, value):
                body.extend(f"--{boundary}\r\n".encode("utf-8"))
                body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
                body.extend(f"{value}\r\n".encode("utf-8"))

            add_field("overwrite", "true")
            add_field("subfolder", subfolder)
            add_field("type", "input")

            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(f'Content-Disposition: form-data; name="image"; filename="{file_path.name}"\r\n'.encode("utf-8"))
            body.extend(b"Content-Type: application/octet-stream\r\n\r\n")
            with open(file_path, "rb") as f:
                body.extend(f.read())
            body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

            req = urllib.request.Request(
                url,
                data=bytes(body),
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "User-Agent": "ComfyUIPipeline/1.0"}
            )
            with urllib.request.urlopen(req) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                uploaded_name = res.get("name", file_path.name)
                return f"{subfolder}/{uploaded_name}" if subfolder else uploaded_name

    def prepare_workflow_for_chunk(
        self,
        uploaded_video_name: str,
        chunk_duration: float,
        chunk_idx: int,
        run_tag: str
    ) -> Dict[str, Any]:
        """Clone and customize workflow JSON for a single sub-video chunk."""
        workflow = json.loads(json.dumps(self.base_workflow))

        if "199" in workflow:
            workflow["199"]["inputs"]["file"] = uploaded_video_name
        if "726" in workflow:
            workflow["726"]["inputs"]["video"] = uploaded_video_name
        if "692" in workflow:
            workflow["692"]["inputs"]["duration"] = chunk_duration
            workflow["692"]["inputs"]["start_time"] = 0

        output_prefix = f"video/{run_tag}_chunk_{chunk_idx:04d}"
        if "68" in workflow:
            workflow["68"]["inputs"]["filename_prefix"] = output_prefix

        if self.random_seed:
            new_seed = random.randint(100000000000000, 999999999999999)
            if "719" in workflow and "inputs" in workflow["719"]:
                workflow["719"]["inputs"]["seed"] = new_seed
            if "129:704" in workflow and "inputs" in workflow["129:704"]:
                workflow["129:704"]["inputs"]["seed"] = new_seed

        if "129:212" in workflow and "inputs" in workflow["129:212"]:
            workflow["129:212"]["inputs"]["value"] = self.enable_llm_prompt

        if self.prompt_override:
            if "711" in workflow and "inputs" in workflow["711"]:
                workflow["711"]["inputs"]["text"] = self.prompt_override
            if "129:211" in workflow and "inputs" in workflow["129:211"]:
                workflow["129:211"]["inputs"]["on_false"] = self.prompt_override

        return workflow

    def queue_prompt(self, workflow_dict: Dict[str, Any]) -> str:
        """Submit the workflow to ComfyUI queue and return prompt_id."""
        payload = {"prompt": workflow_dict, "client_id": self.client_id}
        status, data = self._http_post_json("/prompt", payload)
        if status != 200:
            raise RuntimeError(f"Failed to queue prompt (HTTP {status}): {data}")
        return data["prompt_id"]

    def wait_for_execution_ws(self, prompt_id: str, timeout: int = 1800) -> Dict[str, Any]:
        """Wait for prompt execution via WebSocket or polling fallback."""
        if not HAS_WEBSOCKET:
            return self.wait_for_execution_polling(prompt_id, timeout)

        ws_url = f"{self.ws_base}?clientId={self.client_id}"
        ws = websocket.WebSocket()
        try:
            # Increase connection/socket timeout and enable periodic pings (doubled for stability)
            ws.connect(ws_url, timeout=240, ping_interval=60, ping_timeout=20)
        except Exception as e:
            print(f"[!] WebSocket failed ({e}), using polling fallback...")
            return self.wait_for_execution_polling(prompt_id, timeout)

        start_time = time.time()
        pbar = None
        current_node_title = ""

        try:
            while True:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Execution timed out after {timeout} seconds")

                msg_str = ws.recv()
                if not msg_str:
                    time.sleep(0.1)
                    continue

                if isinstance(msg_str, str):
                    msg = json.loads(msg_str)
                    msg_type = msg.get("type")
                    data = msg.get("data", {})

                    if msg_type == "executing":
                        node_id = data.get("node")
                        if node_id is None and data.get("prompt_id") == prompt_id:
                            if pbar:
                                pbar.close()
                            break
                        elif data.get("prompt_id") == prompt_id:
                            node_meta = self.base_workflow.get(str(node_id), {}).get("_meta", {})
                            current_node_title = node_meta.get("title", f"Node {node_id}")
                            if pbar:
                                pbar.set_description(f"Running [{current_node_title}]")
                            else:
                                print(f"    -> Node: {current_node_title}")

                    elif msg_type == "progress" and data.get("prompt_id") == prompt_id:
                        val = data.get("value", 0)
                        max_val = data.get("max", 1)
                        if pbar is None and HAS_TQDM:
                            pbar = tqdm(total=max_val, desc=f"Running [{current_node_title}]", leave=False)
                        if pbar:
                            pbar.total = max_val
                            pbar.n = val
                            pbar.refresh()

                    elif msg_type == "execution_error" and data.get("prompt_id") == prompt_id:
                        if pbar:
                            pbar.close()
                        err_msg = data.get("exception_message", "Unknown ComfyUI Error")
                        raise RuntimeError(f"ComfyUI Error in node {data.get('node_id')}: {err_msg}")

                    elif msg_type == "execution_success" and data.get("prompt_id") == prompt_id:
                        if pbar:
                            pbar.close()
                        break
        finally:
            try:
                ws.close()
            except Exception:
                pass

        return self.get_history(prompt_id)

    def wait_for_execution_polling(self, prompt_id: str, timeout: int = 1800) -> Dict[str, Any]:
        """Poll /history endpoint until prompt finishes."""
        start_time = time.time()
        print(f"[*] Polling ComfyUI for prompt {prompt_id}...")
        while time.time() - start_time < timeout:
            history = self.get_history(prompt_id)
            if history and "outputs" in history:
                return history
            time.sleep(2.0)
        raise TimeoutError(f"Polling timed out after {timeout} seconds")

    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """Get history for a prompt ID."""
        status, data = self._http_get_json(f"/history/{prompt_id}")
        if status == 200 and isinstance(data, dict):
            return data.get(prompt_id, {})
        return {}

    def download_output_video(self, history: Dict[str, Any], dest_path: Path):
        """Download output video from ComfyUI."""
        outputs = history.get("outputs", {})
        video_info = None

        if "68" in outputs and "videos" in outputs["68"]:
            video_info = outputs["68"]["videos"][0]
        else:
            for node_id, node_out in outputs.items():
                if "videos" in node_out and len(node_out["videos"]) > 0:
                    video_info = node_out["videos"][0]
                    break
                elif "gifs" in node_out and len(node_out["gifs"]) > 0:
                    video_info = node_out["gifs"][0]
                    break

        if not video_info:
            raise RuntimeError(f"Could not find video in outputs: {outputs}")

        filename = video_info["filename"]
        subfolder = video_info.get("subfolder", "")
        file_type = video_info.get("type", "output")
        url = f"{self.http_base}/view?filename={filename}&subfolder={subfolder}&type={file_type}"

        if HAS_REQUESTS:
            r = requests.get(url, stream=True)
            if r.status_code != 200:
                raise RuntimeError(f"Download failed from {url}: {r.text}")
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        else:
            req = urllib.request.Request(url, headers={"User-Agent": "ComfyUIPipeline/1.0"})
            with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
                shutil.copyfileobj(resp, f)

        print(f"[✓] Downloaded chunk -> {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")

    def concatenate_videos(self, chunk_paths: List[Path], final_output_path: Path):
        """Stitch sub-videos in order into one final video using ffmpeg."""
        if not chunk_paths:
            raise ValueError("No video chunks to concatenate!")

        if len(chunk_paths) == 1:
            shutil.copy(chunk_paths[0], final_output_path)
            print(f"[✓] Single chunk copied to {final_output_path}")
            return

        concat_list_path = final_output_path.parent / f"concat_list_{uuid.uuid4().hex[:8]}.txt"
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for p in chunk_paths:
                f.write(f"file '{p.resolve()}'\n")

        print(f"[*] Concatenating {len(chunk_paths)} sub-videos into '{final_output_path.name}'...")

        cmd_copy = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list_path),
            "-c", "copy",
            str(final_output_path)
        ]
        res = subprocess.run(cmd_copy, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        if res.returncode != 0:
            print("[!] Re-encoding during concat for seamless playback...")
            cmd_reencode = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list_path),
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "fast",
                "-c:a", "aac",
                "-b:a", "192k",
                str(final_output_path)
            ]
            subprocess.run(cmd_reencode, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

        if concat_list_path.exists():
            concat_list_path.unlink()

        print(f"[✓] Final Video Created: {final_output_path} ({final_output_path.stat().st_size / (1024*1024):.2f} MB)")

    # --- Execution Modes ---

    def process_presplit_scene_folder(self, scene_folder: Path) -> Path:
        """
        Process an existing scene folder created by split_scenes.py.
        Takes scene_0000.mp4, scene_0001.mp4... feeds each to ComfyUI, and re-stitches.
        """
        scene_folder = Path(scene_folder)
        video_name = scene_folder.name
        run_tag = f"{video_name}_{uuid.uuid4().hex[:6]}"
        work_dir = self.temp_dir / run_tag
        work_dir.mkdir(parents=True, exist_ok=True)

        # Look for sub-videos in order
        scene_files = sorted([
            f for f in scene_folder.iterdir()
            if f.is_file() and f.suffix.lower() in [".mp4", ".mov", ".mkv", ".webm"] and not f.name.startswith(".")
        ])

        if not scene_files:
            raise FileNotFoundError(f"No scene video files found in '{scene_folder}'")

        print("\n" + "=" * 70)
        print(f"🎬 Processing Pre-Split Scenes from: {scene_folder.name} ({len(scene_files)} scenes)")
        print("=" * 70)

        processed_chunk_paths = []

        for i, scene_file in enumerate(scene_files):
            print(f"\n--- [Scene {i+1}/{len(scene_files)}] ({scene_file.name}) ---")
            info = self.get_video_info(scene_file)
            dur = info["duration"]

            # Upload scene video
            uploaded_name = self.upload_file_to_comfyui(scene_file, subfolder=run_tag)

            # Check if extracted frame exists alongside video
            frame_cand = scene_folder / f"{scene_file.stem}_frame0.png"
            if frame_cand.exists():
                self.upload_file_to_comfyui(frame_cand, subfolder=run_tag)

            # Build workflow
            workflow = self.prepare_workflow_for_chunk(
                uploaded_video_name=uploaded_name,
                chunk_duration=dur,
                chunk_idx=i,
                run_tag=run_tag
            )

            # Queue & Execute
            prompt_id = self.queue_prompt(workflow)
            print(f"[*] Queued in ComfyUI (Prompt ID: {prompt_id})")

            history = self.wait_for_execution_ws(prompt_id)

            out_chunk_path = work_dir / f"output_scene_{i:04d}.mp4"
            self.download_output_video(history, out_chunk_path)
            processed_chunk_paths.append(out_chunk_path)

        # Concatenate into final video
        final_output_path = self.output_dir / f"{video_name}_cleaned.mp4"
        self.concatenate_videos(processed_chunk_paths, final_output_path)

        if not self.keep_temp:
            shutil.rmtree(work_dir, ignore_errors=True)

        return final_output_path

    def process_single_video(self, video_path: Path) -> Path:
        """Process one full video: split -> comfyui inference per chunk -> stitch."""
        video_path = Path(video_path)
        run_tag = f"{video_path.stem}_{uuid.uuid4().hex[:6]}"
        work_dir = self.temp_dir / run_tag
        work_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print(f"🚀 Processing Video: {video_path.name}")
        print("=" * 70)

        chunks = self.split_video_into_chunks(video_path, work_dir)
        processed_chunk_paths = []

        for i, chunk in enumerate(chunks):
            idx = chunk["index"]
            print(f"\n--- [Chunk {i+1}/{len(chunks)}] (Duration: {chunk['duration']:.2f}s) ---")

            uploaded_name = self.upload_file_to_comfyui(chunk["video_path"], subfolder=run_tag)
            self.upload_file_to_comfyui(chunk["frame_path"], subfolder=run_tag)

            chunk_workflow = self.prepare_workflow_for_chunk(
                uploaded_video_name=uploaded_name,
                chunk_duration=chunk["duration"],
                chunk_idx=idx,
                run_tag=run_tag
            )

            prompt_id = self.queue_prompt(chunk_workflow)
            print(f"[*] Queued in ComfyUI (Prompt ID: {prompt_id})")

            history = self.wait_for_execution_ws(prompt_id)

            out_chunk_path = work_dir / f"output_chunk_{idx:04d}.mp4"
            self.download_output_video(history, out_chunk_path)
            processed_chunk_paths.append(out_chunk_path)

        final_output_path = self.output_dir / f"{video_path.stem}_cleaned.mp4"
        self.concatenate_videos(processed_chunk_paths, final_output_path)

        if not self.keep_temp:
            shutil.rmtree(work_dir, ignore_errors=True)

        return final_output_path

    def process_input(self, input_path: Path):
        """
        Smart dispatcher: handles single videos, folders of raw videos,
        OR folders of pre-split scenes generated by split_scenes.py.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        # 1. Single video file
        if input_path.is_file():
            self.process_single_video(input_path)
            return

        # 2. Check if the directory itself is a single scene folder (e.g. ./scene_outputs/my_video/)
        scene_files = [f for f in input_path.iterdir() if f.is_file() and f.name.startswith("scene_") and f.suffix.lower() in [".mp4", ".mov", ".mkv"]]
        if scene_files or (input_path / "scenes_metadata.json").exists():
            self.process_presplit_scene_folder(input_path)
            return

        # 3. Check for subdirectories (e.g. ./scene_outputs containing video1/, video2/...)
        sub_dirs = [d for d in sorted(input_path.iterdir()) if d.is_dir() and not d.name.startswith(".")]
        scene_subdirs = [d for d in sub_dirs if any(f.name.startswith("scene_") for f in d.iterdir() if f.is_file()) or (d / "scenes_metadata.json").exists()]

        if scene_subdirs:
            print(f"\n📂 Detected {len(scene_subdirs)} pre-split scene folder(s) in '{input_path}':")
            for d in scene_subdirs:
                print(f"  - {d.name}")
            for d in scene_subdirs:
                try:
                    self.process_presplit_scene_folder(d)
                except Exception as e:
                    print(f"[❌] Error processing scene folder {d.name}: {e}")
            return

        # 4. Folder of raw video files
        video_files = [
            f for f in sorted(input_path.iterdir())
            if f.is_file() and f.suffix.lower() in [".mp4", ".mov", ".mkv", ".avi", ".webm"] and not f.name.startswith(".")
        ]

        if not video_files:
            print(f"[!] No video files or scene folders found in '{input_path}'")
            return

        print(f"\n📂 Found {len(video_files)} raw video(s) in '{input_path}':")
        for vf in video_files:
            print(f"  - {vf.name}")

        for vf in video_files:
            try:
                self.process_single_video(vf)
            except Exception as e:
                print(f"[❌] Error processing {vf.name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Automated ComfyUI Video Batch Processing & Inpainting Pipeline"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="./input_videos",
        help="Path to raw video, folder of videos, OR folder of pre-split scenes from split_scenes.py (default: ./input_videos)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="./output_videos",
        help="Path to store final concatenated output videos (default: ./output_videos)"
    )
    parser.add_argument(
        "-w", "--workflow",
        type=str,
        default="Reddit-removal.json",
        help="Path to ComfyUI workflow API JSON file (default: Reddit-removal.json)"
    )
    parser.add_argument(
        "-s", "--server",
        type=str,
        default="127.0.0.1:8188",
        help="ComfyUI server address (default: 127.0.0.1:8188)"
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=["scene", "fixed"],
        default="scene",
        help="Splitting mode when processing raw videos: 'scene' (splits at scene cuts) or 'fixed' (splits every X seconds) (default: scene)"
    )
    parser.add_argument(
        "-c", "--chunk-duration",
        type=float,
        default=5.0,
        help="Max chunk duration in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=27.0,
        help="Scene cut sensitivity threshold for split-mode=scene (default: 27.0)"
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default="./temp_chunks",
        help="Directory to store intermediate sub-videos and frames (default: ./temp_chunks)"
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep intermediate video chunks and extracted frames after stitching"
    )
    parser.add_argument(
        "--no-random-seed",
        action="store_true",
        help="Disable seed randomization across chunks"
    )
    parser.add_argument(
        "--enable-llm-prompt",
        action="store_true",
        help="Enable Gemma-3 LLM prompt generation node (switch 129:212)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Override positive prompt for subtitle/watermark removal"
    )

    args = parser.parse_args()

    pipeline = ComfyUIPipeline(
        server_address=args.server,
        workflow_path=args.workflow,
        split_mode=args.split_mode,
        chunk_duration=args.chunk_duration,
        scene_threshold=args.scene_threshold,
        output_dir=args.output_dir,
        temp_dir=args.temp_dir,
        keep_temp=args.keep_temp,
        random_seed=not args.no_random_seed,
        enable_llm_prompt=args.enable_llm_prompt,
        prompt_override=args.prompt
    )

    pipeline.process_input(input_path=Path(args.input))


if __name__ == "__main__":
    main()
