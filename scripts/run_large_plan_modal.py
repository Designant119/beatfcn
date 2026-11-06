"""Run a large-training plan on H100 and copy outputs to the Volume."""
import argparse
import json

from beatfcn.cloud import train_stage0_app as training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    with training.app.run(detach=True):
        result = training.train_large_plan.remote(args.plan)
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
