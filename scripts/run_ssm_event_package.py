"""Run the complete SSD+hurdle/GRU alternative through the common harness."""
import json

from beatfcn.model.ssm_event_package import SSMEventPackage
from scripts import run_canonical_package_experiment as run


def main():
    run.IDENT='local-ssm-event-package-v1'
    run.PLAN=run.ROOT/f'bench/rhythm-v2/{run.IDENT}.json'
    run.REPORT=run.ROOT/'reports/local_ssm_event_package_v1.json'
    run.ART=run.ROOT/f'artifacts/{run.IDENT}'
    run.CanonicalPackageCanvas=SSMEventPackage
    run.MODEL_OVERRIDE=json.loads((run.ROOT/'bench/rhythm-v2/local-portfolio-mamba-quality-v1.json').read_text())['model']
    run.EXTRA_SOURCES=('scripts/run_ssm_event_package.py','beatfcn/model/ssm_event_package.py','beatfcn/model/event_code_flow.py','beatfcn/model/local_mamba_canvas.py','beatfcn/model/mamba2_reference.py')
    run.COUNT_HEALTH_GATES=False
    run.main()


if __name__=='__main__':main()
