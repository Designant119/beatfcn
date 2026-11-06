"""Closed D90: matched 1000-update horizon failed whole-song quality."""


def run(*args, **kwargs):
    raise RuntimeError(
        'D90 is closed: 19/20 complete, all-tap collapse and density nonresponse. '
        'No training/resume is permitted through this entry. See '
        'reports/d90_candidate_disposition_v1.json and the archived original '
        'inputs/code_baselines/d90-closed-v1/scripts/run_local_relative_horizon.py.'
    )


if __name__ == '__main__':
    run()
