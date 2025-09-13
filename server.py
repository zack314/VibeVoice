import os
import uuid
import shutil
import pathlib
from urllib.parse import urlparse
from urllib.request import urlopen

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import uvicorn
import threading
import time


APP_ROOT = pathlib.Path(__file__).resolve().parent
OUTPUT_DIR = APP_ROOT / "outputs"
SPEAKER_DIR = OUTPUT_DIR / "speakers"
SPEAKER_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(title="VibeVoice TTS Server", version="0.1")


def _safe_ext_from_url(url: str) -> str:
    parsed = urlparse(url)
    ext = pathlib.Path(parsed.path).suffix.lower()
    # Default to .wav if missing/unsupported extension
    if ext in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
        return ext
    return ".wav"


def _store_from_url(url: str) -> str:
    ext = _safe_ext_from_url(url)
    dest = SPEAKER_DIR / f"speaker-{uuid.uuid4().hex}{ext}"
    try:
        with urlopen(url) as resp, open(dest, "wb") as out:
            shutil.copyfileobj(resp, out)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download speaker audio: {e}")
    return str(dest)


def _store_from_local(path: str) -> str:
    src = pathlib.Path(path)
    if not src.exists() or not src.is_file():
        raise HTTPException(status_code=400, detail=f"Speaker audio not found: {path}")
    ext = src.suffix.lower() or ".wav"
    dest = SPEAKER_DIR / f"speaker-{uuid.uuid4().hex}{ext}"
    try:
        shutil.copy2(src, dest)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store speaker audio: {e}")
    return str(dest)


# --- Persistent TTS service (lazy-loaded) ---
class VibeVoiceService:
    def __init__(self):
        self._init_lock = threading.Lock()
        self._gen_lock = threading.Lock()
        self._loaded = False
        self.model = None
        self.processor = None
        self.device = None
        self.cfg_scale = float(os.environ.get("VIBEVOICE_CFG_SCALE", 1.3))
        self.steps = int(os.environ.get("VIBEVOICE_INFER_STEPS", 10))
        self.model_path = os.environ.get("VIBEVOICE_MODEL", "microsoft/VibeVoice-1.5b")
        self.attn_override = os.environ.get("VIBEVOICE_ATTENTION")  # optional
        self.device_override = os.environ.get("VIBEVOICE_DEVICE")    # optional: cuda|mps|cpu

    def _pick_device_dtype_attn(self):
        import torch
        if self.device_override in ("cuda", "mps", "cpu"):
            device = self.device_override
        else:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        if device == "mps":
            load_dtype = torch.float32
            attn_impl = "sdpa"
        elif device == "cuda":
            load_dtype = torch.bfloat16
            attn_impl = "flash_attention_2"
        else:
            load_dtype = torch.float32
            attn_impl = "sdpa"
        if self.attn_override in ("flash_attention_2", "sdpa"):
            attn_impl = self.attn_override
        return device, load_dtype, attn_impl

    def ensure_loaded(self):
        if self._loaded:
            return
        with self._init_lock:
            if self._loaded:
                return
            # Lazy import heavy deps
            from vibevoice.modular.modeling_vibevoice_inference import (
                VibeVoiceForConditionalGenerationInference,
            )
            from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
            import torch

            self.device, load_dtype, attn_impl = self._pick_device_dtype_attn()
            self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)

            try:
                if self.device == "cuda":
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        self.model_path, torch_dtype=load_dtype, device_map="cuda", attn_implementation=attn_impl
                    )
                elif self.device == "mps":
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        self.model_path, torch_dtype=load_dtype, attn_implementation=attn_impl, device_map=None
                    )
                    self.model.to("mps")
                else:
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        self.model_path, torch_dtype=load_dtype, device_map="cpu", attn_implementation=attn_impl
                    )
            except Exception:
                # Fallback to SDPA if flash attention fails
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map=(self.device if self.device in ("cuda", "cpu") else None),
                    attn_implementation="sdpa",
                )
                if self.device == "mps":
                    self.model.to("mps")

            self.model.eval()
            try:
                self.model.set_ddpm_inference_steps(num_steps=self.steps)
            except Exception:
                pass
            self._loaded = True

    def generate(self, script_line: str, speaker_samples: list[str]) -> str:
        self.ensure_loaded()
        # Prepare inputs
        inputs = self.processor(
            text=[script_line],
            voice_samples=[speaker_samples],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        # Move tensors to target device
        target_device = self.device if self.device != "cpu" else "cpu"
        for k, v in inputs.items():
            if hasattr(v, "to"):
                try:
                    inputs[k] = v.to(target_device)
                except Exception:
                    pass

        # Serialize generation to avoid GPU contention
        with self._gen_lock:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=self.cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
            )

        # Save output
        out_dir = OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{int(time.time())}_generated.wav"
        self.processor.save_audio(outputs.speech_outputs[0], output_path=str(out_path))
        return str(out_path)


# Singleton instance
tts_service = VibeVoiceService()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tts")
def tts(
    text: str = Query(..., description="Plain text to synthesize; single speaker only"),
    speaker_url: str | None = Query(None, description="URL to a speaker reference audio file"),
    speaker_path: str | None = Query(None, description="Local path to a speaker reference audio file"),
):
    """
    Minimal endpoint for single-speaker TTS workflow.

    For now, it stores the provided speaker audio (by URL or local path) under
    VibeVoice/outputs/speakers and returns the stored path. It then attempts
    to perform TTS using the installed VibeVoice package, assuming the
    environment has the model available and adequate resources.
    """

    if not (speaker_url or speaker_path):
        raise HTTPException(status_code=400, detail="Provide either speaker_url or speaker_path")

    # Store the speaker audio
    if speaker_url:
        stored_path = _store_from_url(speaker_url)
    else:
        stored_path = _store_from_local(speaker_path or "")

    # Build the single-speaker script format expected by VibeVoice
    script_line = f"Speaker 1: {text.strip()}"

    # Build base response
    resp: dict[str, str | bool] = {
        "ok": True,
        "stored_speaker_path": stored_path,
        "script": script_line,
        "note": "Stored speaker audio. Attempting TTS using single-speaker 'Speaker 1: ...' format.",
    }

    # Always attempt TTS generation
    try:
        audio_path = tts_service.generate(script_line, [stored_path])
        resp["generated_audio_path"] = audio_path
    except Exception as e:
        # If generation fails, still return the stored speaker path
        resp["ok"] = False
        resp["error"] = f"TTS generation failed: {e}"

    return JSONResponse(resp)


def _run_vibevoice_tts(script_line: str, speaker_samples: list[str]) -> str:
    # Backwards-compat wrapper; uses the persistent service
    return tts_service.generate(script_line, speaker_samples)


if __name__ == "__main__":
    # Run with: python VibeVoice/server.py
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8002)), reload=False)
