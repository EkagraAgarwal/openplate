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
    parser.add_argument('--quality', choices=['preview', 'standard', 'fine'], help='override spec quality')
    parser.add_argument('--field-margin-um', type=float, help='override isolated-feature field margin')
    args = parser.parse_args()
    try:
        spec, errors, _ = validate_spec(args.spec)
        if errors:
            raise ValueError(json.dumps(errors))
        if args.quality:
            spec.numerics.quality = args.quality
        result = (plan_tile(spec, field_margin_um=args.field_margin_um) if args.command == 'plan'
                  else run_sample(spec, args.out, field_margin_um=args.field_margin_um))
    except (ValueError, OSError) as exc:
        parser.exit(2, f'{exc}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
