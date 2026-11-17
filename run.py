#!/usr/bin/env python3
"""Build, calculate and analyze two-body diffusion in position or velocity space."""
import argparse
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'diffu_r_simple_each'
EXECUTABLE = SOURCE / 'out_revision.exe'
sys.path.insert(0, str(ROOT / 'data_process'))


def build(args):
    candidates = [args.eigen_include, os.environ.get('MFRT_EIGEN_INCLUDE'),
                  ROOT / 'dependencies/eigen3', '/usr/include/eigen3',
                  '/usr/local/include/eigen3']
    eigen = next((Path(p).expanduser().resolve() for p in candidates
                  if p and (Path(p).expanduser() / 'Eigen/Dense').is_file()), None)
    if eigen is None:
        raise ValueError('Eigen3 headers not found; set MFRT_EIGEN_INCLUDE '
                         'or pass --eigen-include PATH')
    command = ['make', '-C', str(SOURCE), '-j', str(args.build_jobs),
               'CXX=' + os.environ.get('CXX', 'g++'),
               'CPPFLAGS=' + shlex.join(['-I', str(eigen)]), 'out_revision.exe']
    subprocess.run(command, check=True)


def analyze(root, output, no_plots=False):
    os.environ.setdefault('MPLBACKEND', 'Agg')
    if 'MPLCONFIGDIR' not in os.environ:
        cache = output / '.matplotlib'
        cache.mkdir(parents=True, exist_ok=True)
        os.environ['MPLCONFIGDIR'] = str(cache)
    from revision_analysis import analyze_root
    report = analyze_root(root, output, make_plots=not no_plots)
    for case in report['cases']:
        print(f"{case['case_id']}: median enhancement={case['zeta_median']}, "
              f"mean enhancement={case['zeta_mean']}")
    print(f'Analysis: {output}')


def calculate(args):
    from physical_scales import PhysicalScale
    reference = 'uniform' if args.space == 'pos' else 'iso'
    kind = args.kind or ('perlin' if args.space == 'pos' else 'composite')
    allowed = {'pos': ('uniform', 'perlin', 'radial'),
               'vel': ('iso', 'aniso', 'tail', 'noised', 'composite')}
    if kind not in allowed[args.space]:
        raise ValueError(f'{args.space} sample must be one of {allowed[args.space]}')
    if not 64 <= args.n <= 100_000_000:
        raise ValueError('N must be between 64 and 100000000')
    if any(not math.isfinite(x) or x <= 0 for x in (args.rm_pc, args.mass_msun)):
        raise ValueError('Mass and mean radius must be finite and positive')
    scale = PhysicalScale(args.n, args.rm_pc, args.mass_msun)
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**64 for s in args.seeds):
        raise ValueError('Particle seeds must be distinct unsigned 64-bit integers')
    if any(x < 0 for x in (args.direct_targets, args.count_targets, args.iterations)):
        raise ValueError('Target counts and iteration count must be nonnegative')
    theta = args.theta if args.theta is not None else (.7 if args.space == 'pos' else .5)
    if not math.isfinite(theta) or theta < 0:
        raise ValueError('Tree opening parameter must be finite and nonnegative')
    output = args.output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Output directory must be new or empty')
    build(args)
    options = scale.revision_options(args.space)
    options.update(space=args.space, threads=args.threads, theta=theta,
                   axes=args.axes, seed=0, iterations=args.iterations,
                   **{'count-targets': args.count_targets,
                      'direct-targets': args.direct_targets})
    for key in ('noise_scale', 'noise_step'):
        value = getattr(args, key)
        if value is not None:
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')
            options[key.replace('_', '-')] = value
    output.mkdir(parents=True, exist_ok=True)
    print(f'M={args.mass_msun:g} Msun, Rm={args.rm_pc:g} pc, '
          f'v0={scale.v0:.6g} km/s', flush=True)
    for seed in args.seeds:
        for sample in dict.fromkeys((reference, kind)):
            folder = output / f'{args.space}_{sample}_N{args.n}_seed{seed}'
            options.update(seed=seed, kind=sample, output=str(folder))
            command = [str(EXECUTABLE)]
            for key, value in options.items():
                command.extend(['--' + key, str(value)])
            print(f'Calculating {folder.name}', flush=True)
            with folder.with_suffix('.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
    analyze(output, output / 'analysis', args.no_plots)


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        'Example: python3 run.py run --space pos --n 40000 '
        '--output runs/position_N40000'))
    sub = parser.add_subparsers(dest='command', required=True)
    compile_parser = sub.add_parser('build', help='Compile the C++ diffusion executable')
    run_parser = sub.add_parser('run', help='Calculate a model and its reference, then analyze')
    analyze_parser = sub.add_parser('analyze', help='Analyze existing output directories')
    for p in (compile_parser, run_parser):
        p.add_argument('--eigen-include', type=Path, help='Directory containing Eigen/Dense')
        p.add_argument('--build-jobs', type=int, default=2)
    run_parser.add_argument('--space', choices=('pos', 'vel'), default='pos')
    run_parser.add_argument('--kind', help='pos: uniform, perlin, radial; '
                            'vel: iso, aniso, tail, noised, composite')
    run_parser.add_argument('--n', type=int, default=40000, help='Particle count')
    run_parser.add_argument('--output', type=Path, required=True)
    run_parser.add_argument('--threads', type=int, default=2)
    run_parser.add_argument('--seeds', type=int, nargs='+', default=[20261030],
                            help='Particle seeds; Perlin field seed stays fixed')
    run_parser.add_argument('--mass-msun', type=float, default=1.e7, help='Fixed total mass in Msun')
    run_parser.add_argument('--rm-pc', type=float, default=2000., help='Fixed mean radius in pc')
    run_parser.add_argument('--axes', default='33,28,23', help='Cartesian velocity dispersion ratios')
    run_parser.add_argument('--theta', type=float, help='Tree opening parameter (pos: 0.7; vel: 0.5)')
    run_parser.add_argument('--iterations', type=int, default=14)
    run_parser.add_argument('--noise-scale', type=float, help='Perlin inverse-coordinate scale '
                            '(pos: kpc^-1; vel: (km/s)^-1)')
    run_parser.add_argument('--noise-step', type=float, help='Perlin displacement '
                            '(pos: kpc; vel: km/s)')
    run_parser.add_argument('--direct-targets', type=int, default=0,
                            help='Number of targets checked by direct summation')
    run_parser.add_argument('--count-targets', type=int, default=10000,
                            help='Number of targets for neighbour counts (0 disables counts)')
    run_parser.add_argument('--no-plots', action='store_true')
    analyze_parser.add_argument('--root', type=Path, required=True)
    analyze_parser.add_argument('--output', type=Path, required=True)
    analyze_parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    if getattr(args, 'build_jobs', 1) < 1 or getattr(args, 'threads', 1) < 1:
        parser.error('Thread counts must be positive')
    try:
        if args.command == 'build':
            build(args)
        elif args.command == 'run':
            calculate(args)
        else:
            analyze(args.root.expanduser().resolve(), args.output.expanduser().resolve(), args.no_plots)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
