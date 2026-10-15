# ComfyUI-Drama3-MLX (private draft, not published)
Runs Fish Audio Drama 3 TTS on Apple Silicon through mlx-audio.
- Nodes: **Loader** (4-bit, ~26 GB peak; 8-bit, ~49 GB peak), **Generate** (text with inline cues like `[whispers]`, optional reference AUDIO with its transcript, seed/temperature/top_p/top_k/max_frames, progress bar, interruptible), **Unload**.
- Install: clone the repo into `ComfyUI/custom_nodes/`, then `pip install -r requirements.txt`.
- Models: an HF id (downloaded automatically) or a local folder in `$DRAMA3_MLX_MODELS`, `~/AI/drama3-mlx/models` or `ComfyUI/models/drama3_mlx`.
- Note: MLX runs outside ComfyUI's torch/MPS memory manager. Use Unload before you run heavy MPS models.
- Example: `example_workflows/drama3_mlx_basic.json`. Built with Fish Audio (see NOTICE).
