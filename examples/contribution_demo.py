"""Small portfolio demo added during preparation; original team model unchanged."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from macaque_sim.config.scenarios import build_grooming_matrix
from macaque_sim.config.sim_config import default_config
from macaque_sim.initializer.mri_init import initialize
from macaque_sim.main import run_simulation


def write_svg(matrices: dict[str, np.ndarray], path: Path) -> None:
    """Render input matrices with one shared scale, without browser dependencies."""
    maximum = max(float(matrix.max()) for matrix in matrices.values())
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1020" height="430" viewBox="0 0 1020 430" role="img" aria-label="Three grooming strategy matrices with a shared scale">',
             '<rect width="1020" height="430" fill="#f8fafc"/>',
             '<g font-family="Arial, sans-serif" fill="#0f172a">',
             '<text x="30" y="34" font-size="22" font-weight="bold">Grooming assumptions → model inputs</text>',
             '<text x="30" y="59" font-size="14">30 agents · rows: groomers · columns: recipients · common color scale</text>']
    for panel, (name, matrix) in enumerate(matrices.items()):
        left = 30 + 335 * panel
        parts.append(f'<text x="{left}" y="93" font-size="17">{name.replace("_", " ")}</text>')
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = float(matrix[row, col]) / maximum if maximum else 0.0
                rgb = tuple(round(a + value * (b - a)) for a, b in zip((241, 245, 249), (15, 118, 110)))
                parts.append(f'<rect x="{left + col * 10}" y="{108 + row * 10}" width="10" height="10" fill="rgb{rgb}"/>')
    parts.append('</g></svg>')
    path.write_text('\n'.join(parts) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('output/contribution-demo'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrices, summaries = {}, {}
    for strategy in ('uniform', 'upward_biased', 'coalition'):
        cfg = default_config()
        cfg.grooming_strategy = strategy
        cfg.T = 10.0
        cfg.validate()
        matrix = build_grooming_matrix(cfg)
        alliances, power, ranks = initialize(cfg, matrix)
        assert matrix.shape == (cfg.N, cfg.N)
        assert np.isfinite(alliances).all() and np.all(alliances >= 0)
        assert np.allclose(np.diag(alliances), 0)
        assert np.isclose(power.sum(), 1.0)
        assert set(ranks) == set(range(1, cfg.N + 1))
        log = run_simulation(cfg)
        history = log.get_power_history()
        assert np.isfinite(history).all()
        assert np.allclose(history.sum(axis=1), 1.0)
        matrices[strategy] = matrix
        summaries[strategy] = {
            'config': cfg.to_dict(),
            'initial_ranks': ranks.tolist(),
            'initial_power_sum': float(power.sum()),
            'logged_steps': int(log.step_idx),
            'rank_change_events': len(log.get_events()),
            'final_power': history[-1].tolist(),
        }
    (args.output_dir / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n', encoding='utf-8')
    write_svg(matrices, args.output_dir / 'grooming-strategies.svg')
    print('PASS: three strategies, valid initial states and finite normalized power histories.')
    print(f'Wrote summary.json and grooming-strategies.svg to {args.output_dir}')
    print('Illustrative seeded runs only; not a Monte Carlo or biological validation.')


if __name__ == '__main__':
    main()
