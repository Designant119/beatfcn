"""Closed D88: relative-head candidate failed the registered250/500 stop gate.

Original source and all failed checkpoints remain available for reproduction.
This entry cannot start or resume training after a terminal quality decision.
"""


def run(*args, **kwargs):
    raise RuntimeError(
        'D88 is closed after all four active cores directly END at250 and500; '
        'training/resume is disabled. See reports/d88_candidate_disposition_v1.json '
        'and inputs/code_baselines/d88-closed-v1/scripts/run_local_relative_quality.py.'
    )


if __name__ == '__main__':
    run()
