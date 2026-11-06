"""Run the full-database native tape builder in a Modal CPU container."""
import argparse
import json

from beatfcn.cloud import train_stage0_app as training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    with training.app.run(detach=True):
        result = training.build_full_native_tape.remote(args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
