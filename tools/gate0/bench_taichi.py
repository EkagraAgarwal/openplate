"""Taichi 1.7.4 Vulkan entrypoint; no CPU fallback."""
from common import parser, failure

if __name__ == '__main__':
    args = parser().parse_args()
    try:
        from taichi_backend import run
        raise SystemExit(run('taichi', args))
    except Exception as exc:
        failure('taichi', args.device, exc)
        raise SystemExit(1)
