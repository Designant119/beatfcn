"""Actually execute the previously prepared10M attention control."""
import json

from beatfcn.model.local_typed_canvas import LocalTypedCanvas
from scripts import run_mert_context_pair as engine


def main():
    engine.IDENT='local-attention-control-quality-v1'
    engine.PLAN=engine.ROOT/f'bench/rhythm-v2/{engine.IDENT}.json'
    engine.REPORT=engine.ROOT/'reports/local_attention_control_quality_v1.json'
    engine.ART=engine.ROOT/f'artifacts/{engine.IDENT}'
    engine.MODEL_CLASS=LocalTypedCanvas
    engine.MODEL_CONFIG_OVERRIDE=json.loads((engine.ROOT/'bench/rhythm-v2/local-portfolio-scale-v1.json').read_text())['models']['capacity64']
    engine.SEED_OVERRIDE=20261066;engine.SINGLE_FRONTEND=True
    engine.EXTRA_SOURCES=('scripts/run_attention_control_quality.py',)
    engine.main()


if __name__=='__main__':main()
