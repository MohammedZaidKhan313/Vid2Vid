#!/usr/bin/env python3
"""
=============================================================================
Scene-Based Video Splitter for AI Processing Pipelines (Linux / Python)
=============================================================================

Features:
1. Detects scene cuts/transitions using:
   - Primary: PySceneDetect (`scenedetect`) with ContentDetector or AdaptiveDetector
   - Fallback 1: Native FFmpeg scene change filter (no extra python packages required)
   - Fallback 2: OpenCV (cv2) frame difference detector
2. Splits videos at exact scene change timestamps using FFmpeg.
3. Smart chunking:
   - Enforces optional `min_duration` (prevents micro-cuts / flashes).
   - Enforces optional `max_duration` (e.g., <= 5.0s for ComfyUI / LTX-Video limits).
4. Extracts the 1st frame of each detected scene (PNG/JPG).
5. Generates metadata (`scenes_metadata.json` & `concat_list.txt`) for tracking and re-stitching.
6. Works for both single video files and entire folders in batch mode.

Installation:
  sudo apt update && sudo apt install -y ffmpeg
  pip install scenedetect[opencv] tqdm
=============================================================================
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Scene detection library imports with graceful fallbacks
try:
    from scenedetect import open_video, SceneManager, ContentDetector, AdaptiveDetector, ThresholdDetector
    from scenedetect.frame_timecode import FrameTimecode
    HAS_SCENEDETECT = True
except ImportError:
    HAS_SCENEDETECT = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class SceneSplitter:
    def __init__(
        self,
        output_dir: str = "./scene_outputs",
        detector_type: str = "content",
        threshold: float = 27.0,
        min_scene_duration: float = 1.0,
        max_scene_duration: Optional[float] = 5.0,
        extract_first_frames: bool = True,
        reencode: bool = True,
        frame_format: str = "png"
    ):
        self.output_dir = Path(output_dir)
        self.detector_type = detector_type.lower()
        self.threshold = threshold
        self.min_scene_duration = max(0.2, min_scene_duration)
        self.max_scene_duration = max_scene_duration
        self.extract_first_frames = extract_first_frames
        self.reencode = reencode
        self.frame_format = frame_format.lower().strip(".")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._check_ffmpeg()

    def _check_ffmpeg(self):
        """Ensure ffmpeg and ffprobe are available."""
        for tool in ["ffmpeg", "ffprobe"]:
            if shutil.which(tool) is None:
                raise SystemError(
                    f"Required system tool '{tool}' is missing. "
                    "Install with: sudo apt update && sudo apt install -y ffmpeg"
                )

    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Extract video duration, FPS, width, height using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,duration:format=duration",
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

        # Duration
        duration = float(format_info.get("duration", v_stream.get("duration", 0.0)))

        # FPS
        r_fps = v_stream.get("r_frame_rate", "30/1")
        if "/" in r_fps:
            num, den = map(float, r_fps.split("/"))
            fps = num / den if den != 0 else 30.0
        else:
            fps = float(r_fps)

        width = int(v_stream.get("width", 0))
        height = int(v_stream.get("height", 0))

        return {
            "duration": duration,
            "fps": fps,
            "width": width,
            "height": height
        }

    # --- Scene Detection Strategies ---

    def detect_scenes_scenedetect(self, video_path: Path, video_info: Dict[str, Any]) -> List[Tuple[float, float]]:
        """Detect scenes using PySceneDetect."""
        print(f"[*] Running PySceneDetect ({self.detector_type} detector, threshold={self.threshold})...")
        video = open_video(str(video_path))
        scene_manager = SceneManager()

        min_frames = max(1, int(self.min_scene_duration * video_info["fps"]))

        if self.detector_type == "adaptive":
            scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=self.threshold, min_scene_len=min_frames))
        elif self.detector_type == "threshold":
            scene_manager.add_detector(ThresholdDetector(threshold=self.threshold, min_scene_len=min_frames))
        else:  # default content
            scene_manager.add_detector(ContentDetector(threshold=self.threshold, min_scene_len=min_frames))

        scene_manager.detect_scenes(video, show_progress=HAS_TQDM)
        scenes = scene_manager.get_scene_list()

        cut_pairs = []
        for start_tc, end_tc in scenes:
            start_sec = start_tc.get_seconds()
            end_sec = end_tc.get_seconds()
            cut_pairs.append((start_sec, end_sec))

        # Fallback if no cut detected (single whole video scene)
        if not cut_pairs:
            cut_pairs = [(0.0, video_info["duration"])]

        return cut_pairs

    def detect_scenes_ffmpeg(self, video_path: Path, video_info: Dict[str, Any]) -> List[Tuple[float, float]]:
        """Fallback scene detection using FFmpeg's built-in scene filter."""
        print(f"[*] PySceneDetect not available. Running FFmpeg scene change filter (threshold={self.threshold/100:.2f})...")
        norm_thresh = min(max(self.threshold / 100.0, 0.1), 0.9)
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-filter_complex", f"select='gt(scene,{norm_thresh:.2f})',metadata=print:file=-",
            "-f", "null",
            "-"
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

        cut_pairs = []
        for i in range(len(timestamps) - 1):
            cut_pairs.append((timestamps[i], timestamps[i+1]))

        return cut_pairs

    def detect_scenes_cv2(self, video_path: Path, video_info: Dict[str, Any]) -> List[Tuple[float, float]]:
        """Fallback scene detection using OpenCV HSV histogram differences."""
        print("[*] Running OpenCV HSV histogram scene detector...")
        cap = cv2.VideoCapture(str(video_path))
        fps = video_info["fps"]
        total_dur = video_info["duration"]

        timestamps = [0.0]
        prev_hist = None
        frame_idx = 0
        min_frame_gap = int(self.min_scene_duration * fps)
        last_cut_frame = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            if prev_hist is not None and (frame_idx - last_cut_frame) >= min_frame_gap:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                if diff > (self.threshold / 100.0):
                    ts = frame_idx / fps
                    timestamps.append(ts)
                    last_cut_frame = frame_idx

            prev_hist = hist
            frame_idx += 1

        cap.release()

        if timestamps[-1] < total_dur:
            timestamps.append(total_dur)

        cut_pairs = []
        for i in range(len(timestamps) - 1):
            cut_pairs.append((timestamps[i], timestamps[i+1]))

        return cut_pairs

    def detect_scenes(self, video_path: Path, video_info: Dict[str, Any]) -> List[Tuple[float, float]]:
        """Run scene detection using the best available backend."""
        if HAS_SCENEDETECT:
            return self.detect_scenes_scenedetect(video_path, video_info)
        elif shutil.which("ffmpeg"):
            return self.detect_scenes_ffmpeg(video_path, video_info)
        elif HAS_CV2:
            return self.detect_scenes_cv2(video_path, video_info)
        else:
            raise RuntimeError("No scene detection engine available! Install scenedetect or ffmpeg.")

    def apply_max_duration_subsplit(self, scenes: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        If a scene is longer than max_scene_duration (e.g. 5s for LTX-Video limits),
        smoothly sub-split it so no sub-video exceeds max_scene_duration.
        """
        if not self.max_scene_duration or self.max_scene_duration <= 0:
            return scenes

        refined_scenes = []
        for start_t, end_t in scenes:
            scene_dur = end_t - start_t
            if scene_dur <= self.max_scene_duration:
                refined_scenes.append((start_t, end_t))
            else:
                # Sub-divide into <= max_scene_duration chunks
                curr_start = start_t
                while curr_start < end_t:
                    curr_end = min(curr_start + self.max_scene_duration, end_t)
                    refined_scenes.append((curr_start, curr_end))
                    curr_start = curr_end

        return refined_scenes

    def cut_subvideo(
        self,
        video_path: Path,
        start_time: float,
        duration: float,
        output_path: Path
    ):
        """Slice video segment cleanly with FFmpeg."""
        if self.reencode:
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start_time:.3f}",
                "-i", str(video_path),
                "-t", f"{duration:.3f}",
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "fast",
                "-c:a", "aac",
                "-b:a", "192k",
                "-avoid_negative_ts", "make_zero",
                str(output_path)
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start_time:.3f}",
                "-i", str(video_path),
                "-t", f"{duration:.3f}",
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                str(output_path)
            ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

    def extract_first_frame(self, subvideo_path: Path, frame_output_path: Path):
        """Extract the exact first frame of a sub-video."""
        cmd = [
            "ffmpeg", "-y",
            "-ss", "0.0",
            "-i", str(subvideo_path),
            "-vframes", "1",
            "-q:v", "2",
            str(frame_output_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

    def process_video(self, video_path: Path) -> Path:
        """Process a single video file: detect scenes -> cut sub-videos -> extract frames."""
        video_path = Path(video_path)
        print("\n" + "=" * 70)
        print(f"🎬 Analyzing Video: {video_path.name}")
        print("=" * 70)

        # 1. Probe video
        info = self.get_video_info(video_path)
        print(f"[*] Specs: Duration={info['duration']:.2f}s | FPS={info['fps']:.2f} | Resolution={info['width']}x{info['height']}")

        # 2. Detect scenes
        raw_scenes = self.detect_scenes(video_path, info)
        print(f"[✓] Detected {len(raw_scenes)} raw scene cut(s)")

        # 3. Apply max duration cap (e.g. 5.0s for ComfyUI / LTXV)
        scenes = self.apply_max_duration_subsplit(raw_scenes)
        if len(scenes) != len(raw_scenes):
            print(f"[✓] Refined into {len(scenes)} sub-video(s) after capping at <= {self.max_scene_duration}s")

        # 4. Create dedicated subfolder for this video's outputs
        video_out_dir = self.output_dir / video_path.stem
        video_out_dir.mkdir(parents=True, exist_ok=True)

        metadata_records = []
        concat_lines = []

        print(f"[*] Exporting {len(scenes)} sub-videos to: {video_out_dir}")

        iterator = enumerate(scenes)
        if HAS_TQDM:
            iterator = tqdm(list(iterator), desc="Exporting Scenes")

        for idx, (start_t, end_t) in iterator:
            duration = end_t - start_t
            subvid_name = f"scene_{idx:04d}.mp4"
            frame_name = f"scene_{idx:04d}_frame0.{self.frame_format}"

            subvid_path = video_out_dir / subvid_name
            frame_path = video_out_dir / frame_name

            # Cut video
            self.cut_subvideo(video_path, start_t, duration, subvid_path)

            # Extract 1st frame
            if self.extract_first_frames:
                self.extract_first_frame(subvid_path, frame_path)

            metadata_records.append({
                "scene_index": idx,
                "subvideo_file": subvid_name,
                "first_frame_file": frame_name if self.extract_first_frames else None,
                "start_time_seconds": round(start_t, 3),
                "end_time_seconds": round(end_t, 3),
                "duration_seconds": round(duration, 3)
            })

            concat_lines.append(f"file '{subvid_path.resolve()}'\n")

        # 5. Save metadata JSON and concat list
        meta_file = video_out_dir / "scenes_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({
                "source_video": str(video_path.resolve()),
                "total_duration": info["duration"],
                "total_scenes": len(scenes),
                "max_scene_duration_limit": self.max_scene_duration,
                "scenes": metadata_records
            }, f, indent=2)

        concat_file = video_out_dir / "concat_list.txt"
        with open(concat_file, "w", encoding="utf-8") as f:
            f.writelines(concat_lines)

        print(f"[✓] Video Split Complete: {len(scenes)} sub-videos generated.")
        print(f"    - Metadata: {meta_file.name}")
        print(f"    - Re-stitch list: {concat_file.name}")
        return video_out_dir

    def process_folder(
        self,
        input_dir: Path,
        extensions: Tuple[str, ...] = (".mp4", ".mov", ".mkv", ".avi", ".webm")
    ):
        """Process all video files in an input directory."""
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_path}")

        video_files = [
            f for f in sorted(input_path.iterdir())
            if f.is_file() and f.suffix.lower() in extensions and not f.name.startswith(".")
        ]

        if not video_files:
            print(f"[!] No video files found in '{input_path}' with extensions {extensions}")
            return

        print(f"\n📂 Found {len(video_files)} video(s) to process in '{input_path}':")
        for vf in video_files:
            print(f"  - {vf.name}")

        start_time = time.time()
        for idx, vf in enumerate(video_files):
            print(f"\n[{idx+1}/{len(video_files)}] Processing: {vf.name}")
            try:
                self.process_video(vf)
            except Exception as e:
                print(f"[❌] Error processing {vf.name}: {e}")

        total_elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"🎉 BATCH SCENE SPLITTING FINISHED in {total_elapsed:.1f}s ({total_elapsed/60:.2f} min)")
        print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Scene-Based Video Splitter: Splits video whenever a scene change occurs & extracts first frames."
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to an input video file OR directory containing videos"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="./scene_outputs",
        help="Directory where split sub-videos, frames, and metadata will be saved (default: ./scene_outputs)"
    )
    parser.add_argument(
        "-d", "--detector",
        type=str,
        choices=["content", "adaptive", "threshold"],
        default="content",
        help="Scene detection algorithm (default: content)"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=27.0,
        help="Detector sensitivity threshold (default: 27.0 for content, 3.0 for adaptive)"
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.0,
        help="Minimum scene duration in seconds (prevents micro-cuts, default: 1.0)"
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=5.0,
        help="Maximum chunk duration in seconds. Splits long scenes into sub-chunks of <= this duration (default: 5.0). Set to 0 to disable."
    )
    parser.add_argument(
        "--no-frames",
        action="store_true",
        help="Disable 1st frame image extraction"
    )
    parser.add_argument(
        "--frame-format",
        type=str,
        choices=["png", "jpg", "webp"],
        default="png",
        help="Image format for extracted 1st frames (default: png)"
    )
    parser.add_argument(
        "--fast-copy",
        action="store_true",
        help="Use fast stream-copy without re-encoding (Note: cuts might be less accurate at non-keyframes)"
    )

    args = parser.parse_args()

    max_dur = None if args.max_duration <= 0 else args.max_duration

    splitter = SceneSplitter(
        output_dir=args.output_dir,
        detector_type=args.detector,
        threshold=args.threshold,
        min_scene_duration=args.min_duration,
        max_scene_duration=max_dur,
        extract_first_frames=not args.no_frames,
        reencode=not args.fast_copy,
        frame_format=args.frame_format
    )

    input_path = Path(args.input)
    if input_path.is_file():
        splitter.process_video(input_path)
    elif input_path.is_dir():
        splitter.process_folder(input_path)
    else:
        print(f"Error: Invalid input path '{args.input}'")
        sys.exit(1)


if __name__ == "__main__":
    main()
