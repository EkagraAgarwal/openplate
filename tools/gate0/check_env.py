"""Enumerate installed backends; off-screen smoke test, no personal paths."""
import importlib
from common import RESULTS, clean, failure, parser, save, version


def main():
    parser(False).parse_args()  # --list accepted; environment checks always run
    rows = []
    for name in ('numpy', 'numba', 'taichi', 'quadrants', 'pyopencl', 'tinygrad',
                 'wgpu', 'pyvista', 'gdstk', 'ezdxf', 'mcp', 'onnxruntime'):
        row: dict = dict(framework=name, version=version(name), device='environment',
                   test='import', status='passed')
        try:
            module = importlib.import_module(name)
            if name == 'pyopencl':
                from bench_pyopencl import devices
                for i, d in enumerate(devices()):
                    print(f'OpenCL {i}: {d.name.strip()}; fp64={bool(d.double_fp_config)}')
            elif name == 'wgpu':
                for i, a in enumerate(module.gpu.enumerate_adapters_sync()):
                    print(clean(f'wgpu {i}: {dict(a.info)}'))
            elif name == 'pyvista':
                RESULTS.mkdir(parents=True, exist_ok=True)
                plotter = module.Plotter(off_screen=True, window_size=(256, 256))
                plotter.add_mesh(module.Sphere())
                image = plotter.screenshot(str(RESULTS/'pyvista_sphere.png'))
                plotter.close()
                if image is None or image.max() == image.min():
                    raise RuntimeError('off-screen sphere produced an empty image')
                row['test'] = 'offscreen sphere PNG'
            elif name in ('taichi', 'quadrants'):
                from vulkan_devices import devices
                found = devices()
                for d in found:
                    print(f'{name} Vulkan loader {d.index}: {d.name}; hardware fp64={d.fp64}')
                row['test'] = 'import + Vulkan loader enumeration'
                row['devices'] = [d.name for d in found]
            elif name == 'tinygrad':
                from bench_tinygrad import devices
                found = devices()
                for index, label in found:
                    print(f'tinygrad CL {index}: {label}')
                row['devices'] = [label for _, label in found]
            print(f'{name}: {row["version"]}: OK')
        except Exception as exc:
            row.update(status='failed', error=clean(f'{type(exc).__name__}: {exc}'))
            print(f'{name}: {row["error"]}')
        rows.append(row)
    save('environment', 'installed', rows)


if __name__ == '__main__':
    main()
