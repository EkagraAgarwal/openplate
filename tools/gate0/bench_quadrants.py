"""Quadrants Vulkan entrypoint; QD_VISIBLE_DEVICE selects loader index."""
from common import parser, failure

if __name__ == '__main__':
    args = parser().parse_args()
    try:
        from taichi_backend import run
        raise SystemExit(run('quadrants', args))
    except Exception as exc:
        failure('quadrants', args.device, exc)
        raise SystemExit(1)
