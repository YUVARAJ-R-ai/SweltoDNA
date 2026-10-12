# Environment

- **Machine:** Omarchy (Arch) Linux, RTX 5060 Laptop GPU (8 GB, compute 12.0 / Blackwell), 22 GB RAM.
- **Python:** 3.11 in `.venv` (uv). PyTorch 2.14.1+cu130 (Blackwell needs CUDA ≥ 12.8 builds), transformers 5.19. Extras: `dev`, `serve` (fastapi, uvicorn); matplotlib for research figures.
- **Node:** 22; `web/` is Next.js 15.5 + React 19 + Tailwind v4 + shadcn. Headless verification: `/usr/bin/chromium` (`CHROMIUM_PATH`), swiftshader for screenshots; Vulkan for real-GPU FPS.
- **Data (git-ignored, `data/`):** GENCODE v46 basic GTF, Ensembl GRCh38 chromosomes 1–22, ClinVar GRCh38 VCF; processed Parquet in `data/processed/`. Trained head: `checkpoints/splice_head.pt` (reproduce: `scripts/train_splice_head.py`, 14 min).
- **SpliceAI fixture generator** needs a separate env: `tensorflow-cpu==2.15`, `spliceai`, `setuptools<70` (pkg_resources).
