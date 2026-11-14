"""python -m platesim.sample3d plan|run spec.json [--out DIR]."""
import argparse
import json

from platesim.spec3d import validate_spec
from . import plan_tile, run_sample


def main():
    parser = argparse.ArgumentParser(description='Uncalibrated L1 sample pipeline')
    parser.add_argument('command', choices=['plan', 'run'])
    parser.add_argument('spec')
    parser.add_argument('--out', default='sample3d_run', help='new result directory (run only)')
    args = parser.parse_args()
    try:
        spec, errors, _ = validate_spec(args.spec)
        if errors:
            raise ValueError(json.dumps(errors))
        result = plan_tile(spec) if args.command == 'plan' else run_sample(spec, args.out)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'{exc}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
