"""Start an experiment with driver libraries matching the loaded kernel module.

Only the child environment and a private symlink directory change. The system
driver, ldconfig cache, CUDA runtime packages, and running processes are untouched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time


def native_cuda_environment():
    kernel = Path('/proc/driver/nvidia/version').read_text()
    match = re.search(r'\b(\d{3}\.\d+\.\d+)\b', kernel)
    if match is None:
        raise RuntimeError('Cannot identify the loaded NVIDIA kernel version')
    version = match.group(1)
    native = Path('/usr/lib64')
    bindings = {}
    for family in ['libcuda', 'libnvidia-ptxjitcompiler', 'libnvidia-nvvm',
                   'libcudadebugger', 'libnvidia-ml']:
        for candidate in native.glob(f'{family}.so*'):
            target = candidate.resolve(strict=True)
            if target.name == f'{family}.so.{version}':
                bindings[candidate.name] = target
    for required in ['libcuda.so.1', 'libnvidia-ptxjitcompiler.so.1']:
        if required not in bindings:
            raise RuntimeError(f'No native {required} matching kernel {version}')
    directory = Path('/tmp') / f'loopfill-native-driver-{version}'
    directory.mkdir(exist_ok=True)
    if directory.is_symlink():
        raise RuntimeError(f'Unexpected symlink for private driver directory: {directory}')
    for name, target in bindings.items():
        link = directory / name
        if os.path.lexists(link):
            if not link.is_symlink() or link.resolve(strict=True) != target:
                raise RuntimeError(f'Unexpected existing driver binding: {link}')
        else:
            link.symlink_to(target)
    if {p.name for p in directory.iterdir()} != set(bindings):
        raise RuntimeError('Unexpected files in private driver directory')
    env = os.environ.copy()
    previous = env.get('LD_LIBRARY_PATH', '')
    env['LD_LIBRARY_PATH'] = str(directory) + (':' + previous if previous else '')
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(set(bindings.values()))}
    evidence = dict(
        created_unix=time.time(), kernel_version=version, kernel_description=kernel,
        directory=str(directory), bindings={n: str(p) for n, p in sorted(bindings.items())},
        native_library_sha256=hashes, previous_ld_library_path=previous,
        ld_library_path=env['LD_LIBRARY_PATH'],
        ld_cache_mtime=Path('/etc/ld.so.cache').stat().st_mtime,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='child process environment only; no system driver/cache modification')
    return env, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A child command is required after --')
    if args.record.exists():
        raise FileExistsError(args.record)
    env, evidence = native_cuda_environment()
    evidence['command'] = command
    with args.record.open('x') as stream:
        stream.write(json.dumps(evidence, indent=2) + '\n')
    os.execvpe(command[0], command, env)


if __name__ == '__main__':
    main()
