"""Same initial SSD weights/supervision as direct-song/full, old routing."""
import json

from beatfcn.model.local_mamba_canvas import MambaCanvas
from scripts import run_mert_context_pair as engine


def main():
    engine.IDENT='local-concat-control-quality-v1'
    engine.PLAN=engine.ROOT/f'bench/rhythm-v2/{engine.IDENT}.json'
    engine.REPORT=engine.ROOT/'reports/local_concat_control_quality_v1.json'
    engine.ART=engine.ROOT/f'artifacts/{engine.IDENT}'
    engine.MODEL_CLASS=MambaCanvas
    engine.MODEL_CONFIG_OVERRIDE=json.loads((engine.ROOT/'bench/rhythm-v2/local-portfolio-mamba-quality-v1.json').read_text())['model']
    engine.SEED_OVERRIDE=20261064;engine.SINGLE_FRONTEND=True
    engine.SINGLE_HYPOTHESIS='Exact same initialization/data/batch4/optimizer/decision as direct-song full-context arm; preserve original concatenation-only routing to test the one added physical global read. New input-dependence evidence motivates this matched routing control; not a width experiment.'
    engine.EXTRA_SOURCES=('scripts/run_concat_control_quality.py',)
    engine.main()


if __name__=='__main__':main()
