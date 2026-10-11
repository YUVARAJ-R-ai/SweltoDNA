#!/usr/bin/env python3
"""
Run the splice scoring server for the web app (issue #6).

  python scripts/serve.py --checkpoint checkpoints/splice_head.pt                      # synthetic region (matches the web app)
  python scripts/serve.py --checkpoint checkpoints/splice_head.pt --gene MAPT --exon 10 \
      --gtf data/raw/gencode.v46.basic.annotation.gtf.gz --genome-dir data/raw/genome   # real GRCh38 region

Then start the web app with NEXT_PUBLIC_SPLICE_WS=ws://localhost:8000/ws/splice-session
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None, help="Trained splice head (scripts/train_splice_head.py)")
    ap.add_argument("--mock", action="store_true", help="Use mock backbone instead of HyenaDNA")
    ap.add_argument("--gene", default=None); ap.add_argument("--exon", type=int, default=10)
    ap.add_argument("--gtf", default=None); ap.add_argument("--genome-dir", default=None)
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()

    import uvicorn
    from svelto_dna.core.backbone import SveltoBackbone
    from svelto_dna.model.splice_head import SpliceHead, SplicePredictor
    from svelto_dna.serve.app import create_app
    from svelto_dna.serve.regions import genome_region, synthetic_region

    if a.checkpoint and Path(a.checkpoint).is_file():
        predictor = SplicePredictor.load(a.checkpoint)
        predictor.engine = f"{predictor.backbone.model_name.split('/')[-1]} + splice head"
    else:
        model_name = "mock" if a.mock else "LongSafari/hyenadna-small-32k-seqlen-hf"
        try:
            bb = SveltoBackbone(model_name=model_name, fallback_to_mock=True)
        except Exception:
            bb = SveltoBackbone(model_name="mock")
        head = SpliceHead(hidden_dim=bb.hidden_dim)
        predictor = SplicePredictor(bb, head)
        predictor.engine = f"{predictor.backbone.model_name.split('/')[-1]} + head (untrained)"

    region = genome_region(a.gtf, a.genome_dir, a.gene, a.exon) if a.gene else synthetic_region()
    print(f"Serving {region.description} with {predictor.engine} on ws://{a.host}:{a.port}/ws/splice-session")
    uvicorn.run(create_app(predictor, region), host=a.host, port=a.port, ws_ping_interval=20, ws_ping_timeout=20)
    return 0


if __name__ == "__main__":
    sys.exit(main())
