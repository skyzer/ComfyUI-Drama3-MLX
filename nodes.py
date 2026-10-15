"""Nodes: Drama3MLXLoader -> Drama3MLXGenerate -> AUDIO; Drama3MLXUnload.
Original wrapper code (not derived from the GPLv3 ComfyUI-Drama3). Model + mlx-audio port: Fish Audio Community License / MIT."""
import os, gc, tempfile, platform
import numpy as np

MODELS = {
    "mlx-community/fish-audio-drama-3-4bit": 26,   # approx peak GB
    "mlx-community/fish-audio-drama-3-8bit": 49,
}
_CACHE = {}

def _mem_gb():
    try:
        import subprocess
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 2**30
    except Exception:
        return 0.0

def _local_dirs():
    roots = [os.environ.get("DRAMA3_MLX_MODELS", ""), os.path.expanduser("~/AI/drama3-mlx/models")]
    try:
        import folder_paths
        roots.append(os.path.join(folder_paths.models_dir, "drama3_mlx"))
    except Exception:
        pass
    out = []
    for r in roots:
        if r and os.path.isdir(r):
            out += [os.path.join(r, d) for d in sorted(os.listdir(r)) if os.path.isfile(os.path.join(r, d, "config.json"))]
    return out

def _unload_all():
    _CACHE.clear(); gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass

class Drama3MLXLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": (_local_dirs() + list(MODELS), )}}
    RETURN_TYPES = ("DRAMA3_MLX",); FUNCTION = "load"; CATEGORY = "audio/Drama3 MLX"

    def load(self, model):
        if platform.machine() != "arm64" or platform.system() != "Darwin":
            raise RuntimeError("Drama3 MLX needs Apple Silicon macOS")
        need = MODELS.get(model, 26 if "4bit" in model else 49)
        have = _mem_gb()
        if have and have < need:
            print(f"[Drama3-MLX] warning: {model} peaks ~{need} GB, machine has {have:.0f} GB")
        if model not in _CACHE:
            _unload_all()
            from mlx_audio.tts.utils import load_model
            _CACHE[model] = load_model(model)
        return ({"name": model, "model": _CACHE[model]},)

class Drama3MLXGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "drama3": ("DRAMA3_MLX",),
                    "text": ("STRING", {"multiline": True, "default": "Hello there. [laughs] I can't believe it worked!"}),
                    "seed": ("INT", {"default": 11, "min": 0, "max": 2**31 - 1}),
                    "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05}),
                    "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.01}),
                    "top_k": ("INT", {"default": 30, "min": 1, "max": 200}),
                    "max_frames": ("INT", {"default": 600, "min": 16, "max": 4096})},
                "optional": {
                    "ref_audio": ("AUDIO",),
                    "ref_text": ("STRING", {"multiline": True, "default": ""})}}
    RETURN_TYPES = ("AUDIO",); FUNCTION = "generate"; CATEGORY = "audio/Drama3 MLX"
    DESCRIPTION = "Delivery cues go inline, e.g. [whispers], [laughs], [screams]. MLX seeds differ from CUDA seeds."

    def generate(self, drama3, text, seed, temperature, top_p, top_k, max_frames, ref_audio=None, ref_text=""):
        import mlx.core as mx, torch, soundfile as sf
        model = drama3["model"]; ref_path = None
        if ref_audio is not None:
            if not ref_text.strip():
                raise ValueError("ref_text (transcript of ref_audio) is required with ref_audio")
            w = ref_audio["waveform"][0].float().mean(0).cpu().numpy()
            fd, ref_path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
            sf.write(ref_path, w, int(ref_audio["sample_rate"]))
        pbar = None
        try:
            import comfy.utils
            pbar = comfy.utils.ProgressBar(max_frames)
        except Exception:
            pass
        interrupt = None
        try:
            import comfy.model_management as mm
            interrupt = mm.throw_exception_if_processing_interrupted
        except Exception:
            pass
        mx.random.seed(seed); chunks = []; sr = 44100; frames = 0
        try:
            for r in model.generate(text=text, ref_audio=ref_path, ref_text=ref_text.strip() or None,
                                    temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_frames,
                                    stream=True, streaming_interval=0.5):
                a = np.asarray(r.audio).reshape(-1)
                if a.size: chunks.append(a)
                sr = r.sample_rate or sr
                frames = min(max_frames, frames + int(round(a.size / sr * 21.533)))
                if pbar: pbar.update_absolute(frames, max_frames)
                if interrupt: interrupt()
        finally:
            if ref_path: os.unlink(ref_path)
        if not chunks:
            raise RuntimeError("Drama 3 produced no audio")
        wav = torch.from_numpy(np.concatenate(chunks).astype(np.float32)).view(1, 1, -1)
        return ({"waveform": wav, "sample_rate": sr},)

class Drama3MLXUnload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",)}}
    RETURN_TYPES = ("AUDIO",); FUNCTION = "unload"; CATEGORY = "audio/Drama3 MLX"
    def unload(self, audio):
        _unload_all(); return (audio,)

NODE_CLASS_MAPPINGS = {"Drama3MLXLoader": Drama3MLXLoader, "Drama3MLXGenerate": Drama3MLXGenerate, "Drama3MLXUnload": Drama3MLXUnload}
NODE_DISPLAY_NAME_MAPPINGS = {"Drama3MLXLoader": "Drama 3 MLX Loader", "Drama3MLXGenerate": "Drama 3 MLX Generate", "Drama3MLXUnload": "Drama 3 MLX Unload (free memory)"}
