# ComfyUI Workflow Templates

This folder holds the ComfyUI API-format workflow JSONs used by
`test_comfyUI_WAN2.1.py`, `test_comfyUI_WAN2.2.py`, and `test_comfyUI_ZIT.py`.

## How to obtain

### Wan 2.1 / Wan 2.2 (export from ComfyUI)

1. Start ComfyUI:
   ```bash
   cd ~/ComfyUI && source venv/bin/activate && python main.py --normalvram --fp16-vae
   ```
2. Open http://127.0.0.1:8188
3. For each model, load the template:
   - Wan 2.1: Workflow → Browse Templates → Video → **Wan 2.1 Image to Video**
   - Wan 2.2: Workflow → Browse Templates → Video → **Wan2.2 5B video generation**
4. Export as API format: top-right menu → **Export (API format)**
5. Save here:
   - `workflows/comfyui_wan2.1.json`
   - `workflows/comfyui_wan2.2.json`

### Z-Image-Turbo (download from Hugging Face)

The official workflow is bundled in the model repo. Download it once:

```bash
huggingface-cli download SeeSee21/Z-Image-Turbo-AIO \
  workflows/ZIT-AIO-v1.0.json \
  --local-dir ~/ComfyUI/workflows/
```

`test_comfyUI_ZIT.py` looks for the file at `~/ComfyUI/workflows/ZIT-AIO-v1.0.json`
first, then falls back to `workflows/ZIT-AIO-v1.0.json` in this folder.

The scripts load these files, patch prompt/steps/seed/resolution at runtime,
and submit via the ComfyUI HTTP API. No manual editing of the JSON is needed.

## Files

| File | Model | Notes |
|------|-------|-------|
| `comfyui_wan2.1.json` | Wan 2.1 14B I2V 480p | Export from ComfyUI template |
| `comfyui_wan2.2.json` | Wan 2.2 TI2V-5B 720p | Export from ComfyUI template |
| `ZIT-AIO-v1.0.json` | Z-Image-Turbo FP8 AIO 1024×1024 | Download via `huggingface-cli` (see above) |
