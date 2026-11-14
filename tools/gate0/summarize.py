"""Merge every results/*.json; never infer success from missing results."""
import json
from common import RESULTS, clean, table, write_chunks


def main():
    rows = []
    for path in sorted(RESULTS.glob('*.json')):
        data = json.loads(path.read_text(encoding='utf8'))
        if not isinstance(data, list):
            raise ValueError(f'{path.name}: expected a list of result records')
        if 'initial' in path.stem:
            for row in data:
                row['test'] = 'historical: ' + row.get('test', 'initial failure')
        rows.extend(data)
    cpu = [r for r in rows if r.get('framework') == 'numba' and r.get('test') == 'mgpcg'
           and r.get('N') == 160 and r.get('status') == 'passed']
    decisions = []
    for framework in ('taichi', 'quadrants', 'pyopencl', 'tinygrad', 'wgpu'):
        good = [r for r in rows if r.get('framework') == framework and r.get('status') == 'passed']
        dgpu = [r for r in good if '7700' in r.get('device', '') and r.get('test') == 'mgpcg'
                and r.get('N') == 160]
        igpu = [r for r in good if r.get('test') == 'mgpcg' and r.get('N') == 160
                and any(s in r.get('device', '').lower() for s in ('680', 'gfx1035', 'radeon(tm)'))]
        if cpu and dgpu and igpu:
            speedup = min(r['seconds'] for r in cpu)/min(r['seconds'] for r in dgpu)
            decisions.append(f'- {framework}: {speedup:.3f}x; '
                             + ('QUALIFIES' if speedup >= 5 else 'does not qualify (<5x)'))
        else:
            decisions.append(f'- {framework}: NOT QUALIFIED (missing both-GPU N=160 solve evidence'
                             + ('; sweep-only candidate' if framework in ('tinygrad', 'wgpu') else '') + ').')
    text = '# Gate 0 results\n\n' + '\n'.join(decisions) + '\n\n' + table(rows) + '\n'
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_chunks(RESULTS/'summary.md', clean(text))
    print(clean(text))


if __name__ == '__main__':
    main()
