"""Run requirement benchmarks from a large checkpoint on H100."""
import argparse
import json

from beatfcn.cloud import train_stage0_app as training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int, default=4)
    args = parser.parse_args()
    with training.app.run(detach=True):
        result = training.benchmark_large.remote(args.plan, args.checkpoint, args.out, args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
