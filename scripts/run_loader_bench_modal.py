"""Run the CPU-only exposure-loader throughput benchmark in a Modal CPU container."""
import argparse
import json

from beatfcn.cloud import train_stage0_app as training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', default='bench/rhythm-v2/stage0-mert-int-v1.json')
    parser.add_argument('--workers', default='2,4,8,16')
    parser.add_argument('--batches', type=int, default=200)
    args = parser.parse_args()
    with training.app.run():
        result = training.bench_loader.remote(args.plan, args.workers, args.batches)
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
