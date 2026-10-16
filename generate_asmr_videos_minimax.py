#!/usr/bin/env python3
"""
=============================================================================
MiniMax H3 Automated Batch Video Generator with Hermes Agent
=============================================================================

This pipeline generates N distinct, highly detailed cinematic video prompts
using the Hermes Agent (or its underlying OpenRouter model) and automatically
executes them through the ComfyUI MiniMax H3 text-to-video workflow
(`video_minimax_h3_t2v.json`).

Key Features:
1. Inputs & Setup: Prompts for (or accepts via CLI) the number of videos (e.g. 100)
   and the exact duration of each video in seconds (e.g. 5.0) before starting.
2. Hermes Prompt Engine: Uses the local Hermes Agent (`hermes -z`) to craft unique,
   professional, cinematic MiniMax H3 prompts with timestamped storyboards,
   camera movement, and binaural soundscapes (ideal for ASMR).
3. Direct Fallback: Falls back to OpenRouter direct API if Hermes CLI is busy/unavailable.
4. ComfyUI Orchestration: Automatically injects prompts, durations, randomized seeds,
   and output prefixes into `video_minimax_h3_t2v.json`.
5. WebSocket Monitoring: Real-time progress monitoring with ping keep-alives and tqdm.
6. Manifest & Delivery: Automatically downloads generated MP4 videos and writes a
   detailed manifest of all prompts and metadata.
=============================================================================
"""

import os
import sys
import re
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


# Curated pool of 100+ creative ASMR themes & tactile actions
ASMR_THEME_POOL = [
    "Slicing kinetic sand into razor-thin rainbow layers that cascade downward",
    "Carving intricate hexagonal patterns on artisanal lavender soap with a precision blade",
    "Cutting raw golden honeycomb with an antique brass knife, thick amber nectar oozing",
    "Crushing dried autumn maple leaves between soft velvet-gloved fingertips",
    "Gentle rhythmic tapping on crystal glass goblets filled with varying water levels",
    "Pouring thick viscous clear syrup into a shallow glass dish creating slow concentric ripples",
    "Shaving paper-thin curls off cedar wood block using a polished Japanese hand plane",
    "Turning crisp vintage parchment pages in a leather-bound encyclopedia in a quiet library",
    "Whisking vibrant green ceremonial matcha powder in a ceramic bowl with a bamboo chasen",
    "Peeling a translucent dry gelatin mask off a polished smooth obsidian surface",
    "Crushing a spherical fizzy bath bomb under a heavy transparent glass press",
    "Slicing through a chilled block of floral foam soaked in shimmering micellar water",
    "Dropping large viscous droplets of coloured resin onto an ultra-clear acrylic disc",
    "Raking fine white quartz sand in an indoor zen rock garden with miniature bamboo rake",
    "Typing rapidly on a vintage 1940s manual typewriter, satisfying heavy mechanical key clacks",
    "Softly sweeping delicate horsehair cosmetic brushes across a stereo binaural microphone mesh",
    "Cracking thin crispy caramelized sugar crust on a crème brûlée with an antique silver spoon",
    "Shaping wet terracotta clay on a spinning pottery wheel, gentle hand smoothing",
    "Cutting fresh juicy aloe vera leaf into geometric cubes, viscous gel shimmering under light",
    "Popping individual bubbles on heavy-duty packaging bubble wrap with deliberate precision",
    "Slicing chilled butter slab with a hot stainless steel wire cutter",
    "Rolling heavy brass ball bearings slowly over a textured corrugated wooden track",
    "Grinding pastel chalk sticks into ultra-fine velvety powder using a ceramic mortar and pestle",
    "Scraping cold stone slab with stainless steel spatulas while rolling artisanal ice cream",
    "Trimming delicate bonsai branches with razor-sharp micro-shears, crisp snipping sound",
    "Slicing ripe pomegranate open, glistening jewel-like arils tumbling into a marble bowl",
    "Sprinkling coarse sea salt flakes onto a black slate surface from high above",
    "Folding and pressing crisp linen origami paper into precise crease lines",
    "Dripping melted candle wax slowly into chilled ice water creating floating frozen tendrils",
    "Spinning a heavy titanium precision top on a concave glass lens",
    "Pressing memory foam with slow deep finger indents, watching it slowly recover form",
    "Stirring glass marbles slowly in a shallow bowl of clear viscous aloe vera gel",
    "Peeling iridescent blue metallic tape smoothly off a matte black anodized surface",
    "Cutting compressed kinetic glitter dough with a serrated stainless steel dough cutter",
    "Gently tapping acrylic nails in alternating rhythms on a wooden hollow jewelry box",
    "Crushing dry pinecones beneath a heavy smooth river stone",
    "Slicing thick florist foam with a heated serrated bread knife, crisp dry crunch",
    "Pouring hot espresso slowly over a sphere of crystal clear ice in a heavy tumbler",
    "Scraping dried oil paint impasto textures off an artist palette with a flexible knife",
    "Winding an intricate brass skeleton clockwork mechanism, delicate ticking and gear clicks",
    "Dropping dense colored glass marbles into a vessel filled with clear viscous oil",
    "Cutting a freshly baked sourdough baguette with a heavy bread knife, super-crispy crust crackle",
    "Smoothing wet river stones with warm botanical oil between two hands",
    "Crushing hollow frozen water spheres that shatter into delicate crystalline shards",
    "Tapping rhythmically on hollow bamboo chime pipes with a soft felt mallet",
    "Brushing clean dry sand off ancient engraved fossils using a fine sable artist brush",
    "Slicing a block of cold modelling clay into immaculate geometric cubes with a cheese wire",
    "Pouring fine golden sand into a double-bulb antique hourglass, tiny grains piling",
    "Gently pressing dried lavender blossoms between heavy blotting paper sheets",
    "Stirring a cauldron of shimmering pearlescent wax with a polished glass rod",
    "Peeling vinyl decal transfer paper off a brushed aluminum MacBook chassis",
    "Slicing through layers of coloured modeling wax with a scalpel",
    "Stamping red wax seal onto folded parchment envelope with a heavy engraved brass stamp",
    "Carving geometric spiral patterns into a firm melon using a micro-chisel",
    "Crushing dry lavender and rose potpourri petals in a heavy stone bowl",
    "Sliding heavy smooth mahjong tiles across a felt green gaming table",
    "Pouring crystal-clear sparkling water over mint leaves and ice cubes in slow motion",
    "Slicing cold artisanal cheddar cheese into translucent wafers with a wire slicer",
    "Polishing a rough gemstone on a wet rotating lapidary disc, gentle hum and water splash",
    "Gently crinkling iridescent cellophane wrapping paper near binaural microphones",
    "Tapping fingers on a leather-bound notebook cover with embossed foil details",
    "Scraping frost and thin ice crystals off a textured windowpane with a brass scraper",
    "Dropping beads of mercury-like silver liquid onto a hydrophobic black surface",
    "Slowly turning the combination dial on a heavy vintage steel safe, satisfying clicks",
    "Crushing dehydrated berry clusters into tart aromatic powder",
    "Slicing fresh eucalyptus leaves and releasing microscopic aromatic mist bursts",
    "Dipping textured wooden stamps into dark indigo ink and pressing firmly onto textured cotton paper",
    "Shaving dark chocolate block with a vegetable peeler into delicate chocolate curls",
    "Stretching and folding glossy iridescent slime, releasing tiny satisfying air pocket pops",
    "Gently tapping different sized porcelain teacups with a bamboo chopstick",
    "Raking zen garden gravel around dark basalt stones in graceful concentric arcs",
    "Slicing a roll of freshly made sushi with a razor-sharp yanagiba knife",
    "Crushing brittle honeycomb candy toffee under a flat brass tenderizer mallet",
    "Peeling protective plastic film off a brand-new glossy smartphone screen",
    "Pouring liquid dark chocolate into silicone molds, smooth glossy swirl",
    "Shredding crisp fibrous washi paper strips by hand along watermark deckled edges",
    "Tapping vintage glass fountain pen against a crystal inkwell rim",
    "Slicing chilled coconut jelly cubes with an ultra-thin stainless cleaver",
    "Pressing botanical fern leaves into soft porcelain clay tile",
    "Crushing dry sea shells on a wooden cutting board with a marble pestle",
    "Slowly brushing long coarse wool fibers with a wooden carding comb",
    "Cutting multi-layered florist foam with hot knife emitting silent vapor",
    "Pouring cold milk into black cold-brew coffee creating swirling billowy clouds",
    "Snapping dry spaghetti bundles into equal halves with clean sharp snaps",
    "Stirring heavy brass coins in a velvet-lined wooden tray",
    "Slicing ripe dragonfruit revealing stark white flesh with countless crisp black seeds",
    "Polishing an antique brass telescope barrel with micro-fiber cloth and cream",
    "Crushing crystallized rock sugar crystals on an anvil with small jewelers hammer",
    "Peeling dried PVA glue film cleanly off a smooth silicone craft mat",
    "Gently shaking a vintage glass snow globe with heavy swirling white flecks",
    "Cutting layered floral soap cake with wire cutter revealing hidden botanical embeds",
    "Tapping fingernails on textured slate tile under directional warm studio light",
    "Pouring warm amber resin over dried botanical flowers in a silicone sphere mold",
    "Shaving ice block into ultra-fine fluffy snow with a manual Japanese kakigori machine",
    "Folding heavy raw denim fabric into sharp tailored accordion creases",
    "Crushing dry clay pigment nodules on a glass muller board with linseed oil",
    "Turning brass thumbscrews on a vintage microscope adjusting focal distance",
    "Slicing through a dense roll of colorful washi tape with a surgical blade",
    "Gently pressing fingers into chilled pearlized kinetic putty",
    "Dripping water droplets onto a taut acoustic drum skin one by one"
]


class HermesPromptEngine:
    """
    Interfaces with the local Hermes Agent CLI (or OpenRouter direct API fallback)
    to generate cinematic, production-grade MiniMax H3 prompts.
    """

    def __init__(
        self,
        hermes_bin: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 120
    ):
        self.hermes_bin = self._find_hermes_binary(hermes_bin)
        self.model = model
        self.timeout = timeout
        self.openrouter_api_key = api_key or self._load_openrouter_key()
        self.used_themes: List[str] = []

    def _find_hermes_binary(self, custom_path: Optional[str] = None) -> Optional[str]:
        """Auto-detect the Hermes CLI binary."""
        if custom_path and os.path.exists(custom_path) and os.access(custom_path, os.X_OK):
            return custom_path

        candidates = [
            shutil.which("hermes"),
            os.path.expanduser("~/.local/bin/hermes"),
            os.path.expanduser("~/.hermes/hermes-agent/venv/bin/hermes"),
            "/usr/local/bin/hermes"
        ]
        for c in candidates:
            if c and os.path.exists(c) and os.access(c, os.X_OK):
                return c
        return None

    def _load_openrouter_key(self) -> Optional[str]:
        """Try loading OpenRouter API key from ~/.hermes/.env or environment."""
        if "OPENROUTER_API_KEY" in os.environ:
            return os.environ["OPENROUTER_API_KEY"]

        env_path = Path.home() / ".hermes" / ".env"
        if env_path.exists():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("OPENROUTER_API_KEY="):
                            val = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if val:
                                return val
            except Exception:
                pass
        return None

    def clean_output(self, raw_text: str) -> str:
        """Strip markdown fences, conversational preambles, and extra meta text."""
        text = raw_text.strip()

        # Remove markdown code block wrappers
        text = re.sub(r"^```(?:markdown|text|txt)?\r?\n", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\r?\n```$", "", text)

        # Remove typical LLM chat intro lines
        lines = text.splitlines()
        cleaned_lines = []
        skip_header = True

        for line in lines:
            stripped = line.strip()
            if skip_header:
                if stripped.lower().startswith((
                    "here is", "sure", "below is", "certainly", "here's", "prompt:"
                )) and len(stripped) < 80:
                    continue
                if not stripped:
                    continue
                skip_header = False
            cleaned_lines.append(line)

        result = "\n".join(cleaned_lines).strip()
        return result if result else raw_text.strip()

    def generate_prompt_via_hermes_cli(self, system_instruction: str) -> str:
        """Invoke Hermes CLI in one-shot mode."""
        if not self.hermes_bin:
            raise FileNotFoundError("Hermes binary not found on system.")

        cmd = [self.hermes_bin, "-z", system_instruction, "--ignore-rules"]
        if self.model:
            cmd.extend(["-m", self.model])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            encoding="utf-8",
            check=False
        )

        if result.returncode != 0:
            err = result.stderr.strip() or f"Hermes exited with code {result.returncode}"
            raise RuntimeError(f"Hermes execution error: {err}")

        output = result.stdout.strip()
        if not output and result.stderr:
            output = result.stderr.strip()

        if not output:
            raise ValueError("Hermes returned an empty response.")

        return self.clean_output(output)

    def generate_prompt_via_openrouter_api(self, system_instruction: str) -> str:
        """Direct fallback to OpenRouter API."""
        if not self.openrouter_api_key:
            raise ValueError("No OpenRouter API key found.")

        url = "https://openrouter.ai/api/v1/chat/completions"
        model_name = self.model or "nvidia/nemotron-3-ultra-550b-a55b:free"
        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": system_instruction}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/comfyui-minimax-pipeline",
            "X-Title": "MiniMax H3 Video Generator"
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data_bytes, headers=headers)

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            return self.clean_output(content)

    def generate_video_prompt(
        self,
        video_index: int,
        total_videos: int,
        duration: float = 5.0,
        theme_focus: str = "ASMR"
    ) -> Tuple[str, str]:
        """
        Generate a unique, high-detail prompt for video N of total.
        Returns (sub_theme, prompt_text).
        """
        # Pick sub-theme avoiding recent duplicates
        available_themes = [t for t in ASMR_THEME_POOL if t not in self.used_themes[-30:]]
        if not available_themes:
            available_themes = ASMR_THEME_POOL

        sub_theme = random.choice(available_themes)
        self.used_themes.append(sub_theme)

        # Calculate storyboard timing splits based on duration
        half = round(duration / 2.0, 1)
        dur_str = f"{duration:.1f}s" if duration != int(duration) else f"{int(duration)}s"

        system_instruction = f"""You are an elite prompt engineer for MiniMax H3 (a cutting-edge text-to-video diffusion model with native audio synthesis).

Generate a single, highly detailed, production-ready video generation prompt for a {dur_str} video in the '{theme_focus}' genre.

Specific Subject/Action:
{sub_theme}

You MUST follow this exact structure:

Realistic live-action cinematic look, macro lens, shallow depth of field, warm diffused studio lighting, ultra-detailed 4k tactile textures.

Scene overview: [Write 1-2 vivid, sensory-rich sentences describing the exact satisfying action and tactile materials.]

Storyboard:
[0s-{half}s] Shot 1: [Extreme close-up macro action, initial interaction, micro-details]
[{half}s-{dur_str}] Shot 2: [Camera creeps closer or shifts angle, the satisfying culmination of the action]

Camera: extreme close-up macro, subtle creeping forward motion, rock-steady stabilization.

Audio: binaural stereo microphone, crisp satisfying high-fidelity acoustics of the action, isolated sounds, completely zero background music.

Negative: No text, subtitles, logos, watermarks, cartoon rendering, low quality, unnatural blur, artifacts.

CRITICAL INSTRUCTIONS:
- Return ONLY the finished prompt text matching the structure above.
- Do NOT include any intro, commentary, conversational remarks, or markdown quotes.
"""

        # 1. Try Hermes CLI first (Preferred)
        if self.hermes_bin:
            try:
                prompt_text = self.generate_prompt_via_hermes_cli(system_instruction)
                return sub_theme, prompt_text
            except Exception as e:
                print(f"[!] Hermes CLI error: {e}. Trying direct OpenRouter fallback...")

        # 2. Try OpenRouter direct API fallback
        if self.openrouter_api_key:
            try:
                prompt_text = self.generate_prompt_via_openrouter_api(system_instruction)
                return sub_theme, prompt_text
            except Exception as e:
                print(f"[!] OpenRouter direct API error: {e}")

        # 3. Deterministic template fallback if offline
        print("[!] Generating template prompt using built-in engine...")
        fallback_prompt = (
            f"Realistic live-action cinematic look, macro lens, shallow depth of field, warm diffused studio lighting, ultra-detailed 4k tactile textures.\n\n"
            f"Scene overview: {sub_theme}. The motion is slow, deliberate, and deeply satisfying, capturing micro-textures in hyper-focus.\n\n"
            f"Storyboard:\n"
            f"[0s-{half}s] Shot 1: Extreme macro establishing the surface tension and texture as the motion begins, light catching micro-facets.\n"
            f"[{half}s-{dur_str}] Shot 2: Camera creeps closer as the satisfying culmination unfolds in pristine detail.\n\n"
            f"Camera: extreme close-up macro, subtle creeping forward motion, rock-steady stabilization.\n\n"
            f"Audio: binaural stereo microphone, crisp satisfying acoustics, isolated natural soundscape, zero background music.\n\n"
            f"Negative: No text, subtitles, logos, watermarks, cartoon rendering, low quality, unnatural blur, artifacts."
        )
        return sub_theme, fallback_prompt


class MiniMaxVideoGenerator:
    """
    Connects to ComfyUI, parameterizes `video_minimax_h3_t2v.json`,
    queues execution, monitors live WebSocket progress, and downloads videos.
    """

    def __init__(
        self,
        server_address: str = "127.0.0.1:8188",
        workflow_path: str = "video_minimax_h3_t2v.json",
        output_dir: str = "./output_videos",
        aspect_ratio: str = "16:9 (Widescreen)",
        hermes_bin: Optional[str] = None,
        llm_model: Optional[str] = None,
        api_key: Optional[str] = None
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
        self.workflow_path = self._resolve_workflow_path(workflow_path)
        self.output_dir = Path(output_dir)
        self.aspect_ratio = aspect_ratio
        self.client_id = str(uuid.uuid4())

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.prompt_engine = HermesPromptEngine(
            hermes_bin=hermes_bin,
            model=llm_model,
            api_key=api_key
        )
        self._load_workflow()
        self._check_comfyui_health()

    def _resolve_workflow_path(self, path_str: str) -> Path:
        p = Path(path_str)
        if p.exists():
            return p
        script_dir = Path(__file__).resolve().parent
        alt = script_dir / p.name
        if alt.exists():
            return alt
        return p

    def _load_workflow(self):
        """Load the base workflow JSON."""
        if not self.workflow_path.exists():
            raise FileNotFoundError(f"Workflow file not found: {self.workflow_path}")
        with open(self.workflow_path, "r", encoding="utf-8") as f:
            self.base_workflow = json.load(f)
        print(f"[✓] Loaded base workflow: {self.workflow_path.name} ({len(self.base_workflow)} nodes)")

    def _http_get_json(self, endpoint: str, timeout: int = 10) -> Tuple[int, Any]:
        url = f"{self.http_base}{endpoint}"
        if HAS_REQUESTS:
            try:
                r = requests.get(url, timeout=timeout)
                return r.status_code, r.json() if r.status_code == 200 else r.text
            except Exception as e:
                raise ConnectionError(f"HTTP GET failed to {url}: {e}")
        else:
            req = urllib.request.Request(url, headers={"User-Agent": "MiniMaxGenerator/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                status = response.getcode()
                content = response.read().decode("utf-8")
                return status, json.loads(content)

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
                headers={"Content-Type": "application/json", "User-Agent": "MiniMaxGenerator/1.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                status = response.getcode()
                content = response.read().decode("utf-8")
                return status, json.loads(content)

    def _check_comfyui_health(self):
        """Check ComfyUI server connectivity and GPU device info."""
        try:
            status, res = self._http_get_json("/system_stats", timeout=5)
            if status == 200 and isinstance(res, dict):
                devices = res.get("devices", [])
                device_strs = []
                for d in devices:
                    d_name = d.get("name", "Unknown GPU")
                    if "vram_free" in d and "vram_total" in d:
                        free_gb = d["vram_free"] / (1024**3)
                        total_gb = d["vram_total"] / (1024**3)
                        device_strs.append(f"{d_name} ({free_gb:.1f} / {total_gb:.1f} GB VRAM free)")
                    elif "vram_free" in d:
                        device_strs.append(f"{d_name} ({d['vram_free'] / (1024**3):.1f} GB free)")
                    else:
                        device_strs.append(d_name)
                gpu_summary = ", ".join(device_strs) if device_strs else "CPU"
                print(f"[✓] Connected to ComfyUI at {self.http_base}")
                print(f"    Target GPU: {gpu_summary}")
            else:
                print(f"[!] Warning: ComfyUI returned status {status}")
        except Exception as e:
            raise ConnectionError(
                f"Could not connect to ComfyUI at {self.http_base}. "
                f"Please ensure ComfyUI is running and accessible.\nDetails: {e}"
            )

    def prepare_workflow(
        self,
        prompt: str,
        duration: float,
        video_index: int,
        run_tag: str
    ) -> Dict[str, Any]:
        """Clone and customize video_minimax_h3_t2v.json for one generation."""
        wf = json.loads(json.dumps(self.base_workflow))

        # 1. Inject prompt (Node 105:104 - MiniMaxH3ImageToVideo)
        if "105:104" in wf and "inputs" in wf["105:104"]:
            wf["105:104"]["inputs"]["prompt"] = prompt

        # 2. Inject duration in seconds (Node 105:111 - PrimitiveFloat)
        if "105:111" in wf and "inputs" in wf["105:111"]:
            wf["105:111"]["inputs"]["value"] = float(duration)

        # 3. Inject randomized seed (Node 105:15 - RandomNoise)
        new_seed = random.randint(100000000000000, 999999999999999)
        if "105:15" in wf and "inputs" in wf["105:15"]:
            wf["105:15"]["inputs"]["noise_seed"] = new_seed

        # 4. Inject output filename prefix (Node 92 - SaveVideo)
        output_prefix = f"video/{run_tag}_video_{video_index:03d}"
        if "92" in wf and "inputs" in wf["92"]:
            wf["92"]["inputs"]["filename_prefix"] = output_prefix

        # 5. Optionally adjust aspect ratio (Node 115 - ResolutionSelector)
        if "115" in wf and "inputs" in wf["115"]:
            wf["115"]["inputs"]["aspect_ratio"] = self.aspect_ratio

        return wf

    def queue_prompt(self, workflow_dict: Dict[str, Any]) -> str:
        """Submit the workflow to ComfyUI queue and return prompt_id."""
        payload = {"prompt": workflow_dict, "client_id": self.client_id}
        status, data = self._http_post_json("/prompt", payload)
        if status != 200:
            raise RuntimeError(f"Failed to queue prompt (HTTP {status}): {data}")
        return data["prompt_id"]

    def wait_for_execution_ws(self, prompt_id: str, timeout: int = 1800) -> Dict[str, Any]:
        """Wait for prompt execution via WebSocket with keep-alive ping and fallback."""
        if not HAS_WEBSOCKET:
            return self.wait_for_execution_polling(prompt_id, timeout)

        ws_url = f"{self.ws_base}?clientId={self.client_id}"
        ws = websocket.WebSocket()
        try:
            # Keepalive ping frames prevent connection drops during long diffusion generation
            # Doubled intervals (timeout=240, ping_interval=60, ping_timeout=20) for long-running remote GPU execution
            ws.connect(ws_url, timeout=240, ping_interval=60, ping_timeout=20)
        except Exception as e:
            print(f"[!] WebSocket connection failed ({e}), switching to polling fallback...")
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
            time.sleep(3.0)
        raise TimeoutError(f"Polling timed out after {timeout} seconds")

    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """Get history output data for a prompt ID."""
        status, data = self._http_get_json(f"/history/{prompt_id}")
        if status == 200 and isinstance(data, dict):
            return data.get(prompt_id, {})
        return {}

    def download_output_video(self, history: Dict[str, Any], dest_path: Path):
        """Locate and download generated video from ComfyUI output."""
        outputs = history.get("outputs", {})
        video_info = None

        # Check Node 92 (SaveVideo)
        if "92" in outputs and "videos" in outputs["92"] and len(outputs["92"]["videos"]) > 0:
            video_info = outputs["92"]["videos"][0]
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
            req = urllib.request.Request(url, headers={"User-Agent": "MiniMaxGenerator/1.0"})
            with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
                shutil.copyfileobj(resp, f)

        print(f"[✓] Downloaded -> {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")

    def run_batch(
        self,
        num_videos: int,
        duration: float,
        theme_focus: str = "ASMR"
    ):
        """Run complete batch generation loop."""
        run_tag = f"{theme_focus.lower().replace(' ', '_')}_{uuid.uuid4().hex[:6]}"
        manifest_path = self.output_dir / f"manifest_{run_tag}.json"
        markdown_log_path = self.output_dir / f"manifest_{run_tag}.md"

        print("\n" + "═" * 72)
        print(f"🎬 Starting MiniMax H3 Batch Generation")
        print(f"   • Total Videos:    {num_videos}")
        print(f"   • Video Duration:  {duration} seconds each")
        print(f"   • Theme/Style:     {theme_focus}")
        print(f"   • Hermes Engine:   {'Installed (' + self.prompt_engine.hermes_bin + ')' if self.prompt_engine.hermes_bin else 'OpenRouter Direct API'}")
        print(f"   • Output Folder:   {self.output_dir.resolve()}")
        print("═" * 72 + "\n")

        manifest_records = []

        start_total = time.time()

        for idx in range(1, num_videos + 1):
            print(f"\n{'─' * 70}")
            print(f"📹 [Video {idx}/{num_videos}] Generating Concept & Prompt with Hermes...")
            print(f"{'─' * 70}")

            # 1. Generate unique prompt via Hermes
            t_prompt_start = time.time()
            sub_theme, prompt_text = self.prompt_engine.generate_video_prompt(
                video_index=idx,
                total_videos=num_videos,
                duration=duration,
                theme_focus=theme_focus
            )
            t_prompt = time.time() - t_prompt_start
            print(f"[✓] Concept ({t_prompt:.1f}s): {sub_theme}")
            print(f"    Prompt Preview: {prompt_text[:140].replace(chr(10), ' ')}...")

            # 2. Prepare workflow
            custom_wf = self.prepare_workflow(
                prompt=prompt_text,
                duration=duration,
                video_index=idx,
                run_tag=run_tag
            )

            # 3. Queue in ComfyUI
            print(f"[*] Submitting to ComfyUI queue...")
            prompt_id = self.queue_prompt(custom_wf)
            print(f"[✓] Queued (Prompt ID: {prompt_id})")

            # 4. Wait for execution
            t_gen_start = time.time()
            history = self.wait_for_execution_ws(prompt_id)
            t_gen = time.time() - t_gen_start
            print(f"[✓] Generation finished in {t_gen:.1f}s")

            # 5. Download output
            out_filename = f"{run_tag}_{idx:03d}.mp4"
            dest_video = self.output_dir / out_filename
            self.download_output_video(history, dest_video)

            # Record manifest entry
            record = {
                "index": idx,
                "filename": out_filename,
                "path": str(dest_video.resolve()),
                "duration": duration,
                "theme": sub_theme,
                "prompt": prompt_text,
                "prompt_id": prompt_id,
                "generation_time_seconds": round(t_gen, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            manifest_records.append(record)

            # Save manifest incrementally
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_records, f, indent=2)

        total_elapsed = time.time() - start_total
        print("\n" + "═" * 72)
        print(f"🎉 Batch Generation Complete!")
        print(f"   • Completed Videos: {len(manifest_records)}/{num_videos}")
        print(f"   • Total Time:       {total_elapsed / 60:.2f} minutes")
        print(f"   • Videos Saved To:  {self.output_dir.resolve()}")
        print(f"   • Manifest Saved:   {manifest_path.name}")
        print("═" * 72)

        # Write clean markdown report
        self._write_markdown_report(markdown_log_path, manifest_records, run_tag, duration, theme_focus, total_elapsed)

    def _write_markdown_report(self, md_path: Path, records: List[Dict[str, Any]], run_tag: str, duration: float, theme: str, total_time: float):
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# MiniMax H3 Batch Video Report ({run_tag})\n\n")
            f.write(f"- **Theme**: {theme}\n")
            f.write(f"- **Total Videos**: {len(records)}\n")
            f.write(f"- **Duration Per Video**: {duration}s\n")
            f.write(f"- **Total Generation Time**: {total_time/60:.2f} minutes\n\n")
            f.write("| # | Video File | Sub-Theme | Gen Time | Prompt Preview |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            for r in records:
                preview = r['prompt'][:80].replace('\n', ' ') + "..."
                f.write(f"| {r['index']} | `{r['filename']}` | {r['theme'][:30]} | {r['generation_time_seconds']}s | {preview} |\n")


def normalize_aspect_ratio(val: Optional[str]) -> str:
    """Normalize user input or shorthand into exact ComfyUI ResolutionSelector string."""
    if not val:
        return "16:9 (Widescreen)"
    cleaned = str(val).strip().lower()
    if cleaned in ["9:16", "9/16", "vertical", "portrait", "shorts", "reels", "tiktok", "9:16 (vertical)"]:
        return "9:16 (Vertical)"
    if cleaned in ["16:9", "16/9", "horizontal", "landscape", "widescreen", "youtube", "16:9 (widescreen)"]:
        return "16:9 (Widescreen)"
    if cleaned in ["1:1", "1/1", "square", "1:1 (square)"]:
        return "1:1 (Square)"
    if cleaned in ["4:3", "4/3"]:
        return "4:3"
    if cleaned in ["21:9", "21/9", "ultrawide"]:
        return "21:9"
    return val


def prompt_user_inputs() -> Tuple[int, float, str, str]:
    """Interactively prompt user for number of videos, duration, theme, and aspect ratio."""
    print("\n" + "═" * 70)
    print("🎬 MiniMax H3 Batch Video Generator Setup")
    print("═" * 70)

    # Number of videos
    while True:
        num_str = input("👉 Enter number of videos to generate [default: 10]: ").strip()
        if not num_str:
            num_videos = 10
            break
        try:
            num_videos = int(num_str)
            if num_videos > 0:
                break
            print("   Please enter a positive integer.")
        except ValueError:
            print("   Invalid number. Please enter an integer (e.g. 10, 100).")

    # Duration in seconds
    while True:
        dur_str = input("👉 Enter length/duration of each video in seconds [default: 5.0]: ").strip()
        if not dur_str:
            duration = 5.0
            break
        try:
            duration = float(dur_str)
            if duration >= 1.0:
                break
            print("   Duration should be at least 1.0 second.")
        except ValueError:
            print("   Invalid duration. Please enter a float/number (e.g. 5.0).")

    # Theme
    theme = input("👉 Enter video theme/focus [default: ASMR]: ").strip()
    if not theme:
        theme = "ASMR"

    # Aspect Ratio
    print("👉 Choose aspect ratio:")
    print("   [1] 16:9 Landscape / Widescreen (YouTube, Desktop) [default]")
    print("   [2] 9:16 Vertical (Shorts, Reels, TikTok)")
    print("   [3] 1:1 Square (Instagram Feed)")
    ar_choice = input("Select [1-3, default: 1]: ").strip()
    if ar_choice == "2" or "vertical" in ar_choice.lower() or "9:16" in ar_choice:
        aspect_ratio = "9:16 (Vertical)"
    elif ar_choice == "3" or "square" in ar_choice.lower() or "1:1" in ar_choice:
        aspect_ratio = "1:1 (Square)"
    else:
        aspect_ratio = "16:9 (Widescreen)"

    print("═" * 70)
    return num_videos, duration, theme, aspect_ratio


def main():
    parser = argparse.ArgumentParser(
        description="Automated MiniMax H3 Batch Video Generation with Hermes Agent Prompts"
    )
    parser.add_argument(
        "-n", "--num-videos",
        type=int,
        default=None,
        help="Number of videos to generate (e.g. 100). If omitted, prompts interactively."
    )
    parser.add_argument(
        "-l", "--length", "--duration",
        type=float,
        default=None,
        dest="duration",
        help="Length/duration of each video in seconds (e.g. 5.0). If omitted, prompts interactively."
    )
    parser.add_argument(
        "-t", "--theme",
        type=str,
        default=None,
        help="Video theme / style focus (default: ASMR). If omitted, prompts interactively."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="./output_videos",
        help="Directory to save generated MP4 videos (default: ./output_videos)"
    )
    parser.add_argument(
        "-w", "--workflow",
        type=str,
        default="video_minimax_h3_t2v.json",
        help="Path to ComfyUI workflow JSON (default: video_minimax_h3_t2v.json)"
    )
    parser.add_argument(
        "-s", "--server",
        type=str,
        default="127.0.0.1:8188",
        help="ComfyUI server address (default: 127.0.0.1:8188)"
    )
    parser.add_argument(
        "-a", "--aspect-ratio",
        type=str,
        default=None,
        help="Video aspect ratio: '9:16', 'vertical', '16:9', '1:1', etc. (default: '16:9 (Widescreen)')"
    )
    parser.add_argument(
        "--hermes-bin",
        type=str,
        default=None,
        help="Explicit path to hermes binary (auto-detected by default)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="Override LLM model used by Hermes/OpenRouter"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenRouter API key for LLM prompting (if Hermes CLI is not present on this machine)"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run without interactive confirmation prompts"
    )

    args = parser.parse_args()

    # Determine parameters (interactive if not provided via CLI)
    if args.num_videos is not None and args.duration is not None:
        num_videos = args.num_videos
        duration = args.duration
        theme = args.theme or "ASMR"
        aspect_ratio = normalize_aspect_ratio(args.aspect_ratio or "16:9 (Widescreen)")
    else:
        num_videos, duration, theme, auto_aspect = prompt_user_inputs()
        aspect_ratio = normalize_aspect_ratio(args.aspect_ratio or auto_aspect)

    generator = MiniMaxVideoGenerator(
        server_address=args.server,
        workflow_path=args.workflow,
        output_dir=args.output_dir,
        aspect_ratio=aspect_ratio,
        hermes_bin=args.hermes_bin,
        llm_model=args.model,
        api_key=args.api_key
    )

    # Confirmation summary before starting
    if not args.non_interactive and sys.stdin.isatty():
        print(f"\nReady to generate {num_videos} '{theme}' videos ({duration}s each, format: {aspect_ratio}).")
        proceed = input("Start generation now? [Y/n]: ").strip().lower()
        if proceed in ["n", "no"]:
            print("Aborted by user.")
            sys.exit(0)

    generator.run_batch(
        num_videos=num_videos,
        duration=duration,
        theme_focus=theme
    )


if __name__ == "__main__":
    main()
