# ComfyUI-Drama3-MLX: Fish Audio Drama 3 on Apple Silicon

ComfyUI nodes that run [Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3) natively on Apple Silicon through [mlx-audio](https://github.com/Blaizzy/mlx-audio). No CUDA needed.

Sample, generated on an M5 Max with the 4-bit model and the LJ Speech public-domain reference voice: [▶ play in browser](https://skyzer.github.io/ComfyUI-Drama3-MLX/) · [mp4](samples/mp4/mlx_public_s11.mp4) · [wav](samples/mlx_public_s11.wav)

## Install
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/skyzer/ComfyUI-Drama3-MLX
pip install -r ComfyUI-Drama3-MLX/requirements.txt
```
Load `example_workflows/drama3_mlx_basic.json`.

## Nodes
- **Drama 3 MLX Loader:** `mlx-community/fish-audio-drama-3-4bit` (~26 GB peak), `-8bit` (~49 GB), or a local folder. Local folders are searched in `$DRAMA3_MLX_MODELS`, `~/AI/drama3-mlx/models` and `ComfyUI/models/drama3_mlx`.
- **Drama 3 MLX Generate:**
  - text with inline delivery cues (`[whispers]`, `[laughs]`, `[sobbing]`, `[screaming]`)
  - optional reference AUDIO plus its transcript, for voice cloning
  - seed, temperature, top_p, top_k and max_frames
  - a progress bar, and it can be interrupted
- **Drama 3 MLX Unload:** frees the MLX weights (MLX lives outside ComfyUI's torch memory manager).

## Benchmark
Measured on an M5 Max (128 GB) with the 4-bit model: <!--MLX_BENCH-->. The full method and comparison against CUDA builds are in [drama3-quant-bench](https://github.com/skyzer/drama3-quant-bench).

## Notes
- MLX and PyTorch-MPS share unified memory, so unload before running heavy MPS models.
- MLX seeds are not the same as CUDA seeds.

## License
The wrapper code is MIT (see LICENSE). The model weights are not included and are under the Fish Audio Community License: built with Fish Audio (see NOTICE).
