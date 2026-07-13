# AGENTS.md

## Project purpose

This repository provides a web demo for single-view Balinese mask
3D reconstruction using InstantMesh.

The project must reuse the existing pretrained and fine-tuned models.
Do not train a new model and do not replace the InstantMesh architecture.

## Source of truth

The working reference implementations are:

- references/notebooks/instantmesh_working.ipynb
- references/notebooks/instantmesh_finetuned_working.ipynb

Preserve their:

- model initialization,
- preprocessing,
- camera configuration,
- diffusion configuration,
- reconstruction configuration,
- checkpoint loading behavior,
- mesh extraction,
- video generation,
- OBJ/MTL/texture export.

## Non-negotiable constraints

- Do not rewrite InstantMesh from scratch.
- Do not silently change model parameters.
- Do not add training code to the web application.
- Do not commit checkpoints or credentials.
- Do not hard-code Google Drive paths in reusable modules.
- Use environment variables or CLI arguments for paths.
- Never load all model variants into GPU memory simultaneously.
- GPU inference concurrency must be one.
- Every request must have an isolated UUID output directory.
- GLB is the primary format for the interactive viewer.
- OBJ, MTL and texture files must be packaged as ZIP.
- Fine-tuned checkpoints may contain partial state dictionaries.
- Log missing and unexpected checkpoint keys.
- Fail clearly if a checkpoint is incompatible.

## Default inference settings

- seed: 42
- diffusion_steps: 75
- view_count: 6
- image_size: 320
- fp16: true
- export_texmap: true
- save_video: true

Do not alter these defaults without an explicit request.

## Planned repository layout

- app.py
- webapp/
- configs/
- scripts/
- tools/
- tests/
- references/
- outputs/

## Required web outputs

The inference function must return:

1. processed input image,
2. six individual multiview images,
3. rotating MP4,
4. GLB mesh,
5. ZIP containing OBJ, MTL and texture,
6. inference metadata and logs.

## Verification

Before claiming completion:

- run static import checks,
- run unit tests,
- run pretrained GPU smoke inference,
- run fine-tuned GPU smoke inference,
- verify six multiview images,
- verify playable MP4,
- verify non-empty GLB,
- verify ZIP contains OBJ and related assets,
- verify repeated requests use different output directories.