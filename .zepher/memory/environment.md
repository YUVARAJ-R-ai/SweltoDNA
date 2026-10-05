# Environment

- **OS:** NixOS (x86_64, Linux 7.0.10-zen1)
- **Package Manager:** `uv` (`0.9.30`)
- **Python Version:** 3.11.15 in `.venv`
- **Key Libraries:** `torch==2.14.1+cpu`, `transformers==5.18.0`, `einops==0.8.2`, `pydantic==2.13.5`, `psutil==7.2.2`, `pytest==9.1.1`
- **Device Support:** Auto-detection for CUDA and CPU, automatic precision fallback (`torch.float32` on CPU, `float16`/`bfloat16` on CUDA)
