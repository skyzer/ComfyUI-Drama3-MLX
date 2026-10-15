"""CLI smoke: runs the node classes without ComfyUI. usage: python tests/smoke.py MODEL REF_WAV REF_TXT OUT_WAV"""
import sys, os, time, soundfile as sf, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib; n = importlib.import_module("nodes")
m, rw, rt, out = sys.argv[1:5]
w, sr = sf.read(rw, dtype="float32"); w = torch.from_numpy(w).view(1, 1, -1) if w.ndim == 1 else torch.from_numpy(w.T).unsqueeze(0)
t = time.time(); (d,) = n.Drama3MLXLoader().load(m); print("load", round(time.time()-t, 1))
t = time.time()
(a,) = n.Drama3MLXGenerate().generate(d, "Well, that went better than expected. [laughs] Let's do it again.", 11, 1.0, 0.95, 30, 600,
                                      {"waveform": w, "sample_rate": sr}, open(rt).read())
g = time.time()-t; dur = a["waveform"].shape[-1]/a["sample_rate"]
sf.write(out, a["waveform"].view(-1).numpy(), a["sample_rate"]); print("gen", round(g, 1), "dur", round(dur, 2), "rtf", round(g/dur, 2), out)
