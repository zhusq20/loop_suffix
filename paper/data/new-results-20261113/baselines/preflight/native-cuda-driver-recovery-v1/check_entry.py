"""CPU checks for re-exec of DFlash stages spawned by an adopted old worker."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from dflash_source_identity import sha, without_main, validate_evaluation_sources
from native_cuda_env import native_cuda_environment
from provenance import freeze_sources

env, evidence = native_cuda_environment()
resolver = "import ctypes,json;from pathlib import Path;ctypes.CDLL('libcuda.so.1');print(json.dumps(sorted({l.split()[-1] for l in Path('/proc/self/maps').read_text().splitlines() if 'libcuda.so.' in l})))"
libraries = json.loads(subprocess.check_output([sys.executable, '-c', resolver], env=env, text=True))
assert libraries == ['/usr/lib64/libcuda.so.' + evidence['kernel_version']]
assert Path('/etc/ld.so.cache').stat().st_mtime == evidence['ld_cache_mtime']

spec = importlib.util.spec_from_file_location('entry_under_test', BASE/'evaluate_dflash.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)

class Reexecuted(Exception):
    pass

with tempfile.TemporaryDirectory(prefix='loopfill-driver-entry-check-') as tmp:
    temp = Path(tmp)
    (temp/'logs').mkdir()
    out = temp/'uncreated-run'
    argv = [str(BASE/'evaluate_dflash.py'), '--model', 'ouro-2.6b', '--method', 'ar',
            '--dataset', 'math500', '--temperature', '1', '--gpu', '4',
            '--cpus', '48-51', '--out', str(out)]
    stale_env = dict(os.environ, LD_LIBRARY_PATH='/usr/local/cuda/compat')
    with patch.object(entry, 'BASE', temp), patch.object(sys, 'argv', argv), \
         patch.dict(os.environ, stale_env, clear=True), \
         patch('native_cuda_env.native_cuda_environment', return_value=(env, dict(evidence))), \
         patch.object(os, 'execve', side_effect=Reexecuted) as execute:
        try:
            entry.main()
        except Reexecuted:
            pass
        else:
            raise AssertionError('Old environment did not re-exec')
        execute.assert_called_once_with(sys.executable, [sys.executable, '-u', *argv], env)
        assert not out.exists()
        record = temp/'logs/uncreated-run-native-driver.json'
        assert json.loads(record.read_text())['command'] == [sys.executable, '-u', *argv]
        try:
            entry.main()
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing restart evidence was overwritten')
        assert execute.call_count == 1
    # A corrected parent does not re-exec. Stop at the existing-output guard,
    # before nvidia-smi, CUDA imports, or a GPU context can be reached.
    out.mkdir()
    with patch.object(entry, 'BASE', temp), patch.object(sys, 'argv', argv), \
         patch.dict(os.environ, env, clear=True), \
         patch('native_cuda_env.native_cuda_environment', return_value=(env, dict(evidence))), \
         patch.object(os, 'execve') as execute:
        try:
            entry.main()
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing output was not rejected')
        execute.assert_not_called()

checked = []
for manifest in sorted((BASE/'runs').glob('dflash-*/source-manifest.json')):
    folder = manifest.parent
    if not (folder/'metadata.json').exists():
        continue
    validate_evaluation_sources(folder)
    assert without_main(folder/'source/evaluate_dflash.py') == without_main(BASE/'evaluate_dflash.py')
    checked.append(str(folder.relative_to(BASE)))
assert checked

# Reject a changed archived implementation without weakening historical checks.
with tempfile.TemporaryDirectory(prefix='loopfill-driver-source-check-') as tmp:
    import shutil
    copy = Path(tmp)/'run'
    source = BASE/checked[0]
    shutil.copytree(source/'source', copy/'source')
    for name in ['source-manifest.json', 'metadata.json']:
        shutil.copy2(source/name, copy/name)
    target = copy/'source/dflash_impl/decode.py'
    target.write_text(target.read_text() + '\n# simulated modification\n')
    try:
        validate_evaluation_sources(copy)
    except AssertionError as error:
        assert 'Archived evaluation source changed' in str(error)
    else:
        raise AssertionError('Changed archived decoder was accepted')

out = Path(__file__).parent/'entry-check'
out.mkdir(exist_ok=False)
freeze_sources(out, [BASE/'evaluate_dflash.py', BASE/'native_cuda_env.py',
                    BASE/'dflash_source_identity.py', Path(__file__)])
report = dict(status='cpu_entry_checks_passed', checked_unix=time.time(),
              entry_sha256=sha(BASE/'evaluate_dflash.py'), native_environment=evidence,
              native_loader_libraries=libraries, checked_runs=checked,
              non_main_ast_unchanged=True, existing_record_and_output_rejected=True,
              tampered_decoder_rejected=True,
              limitations='Re-exec branch is mocked; actual loader resolves the native library without acquiring CUDA. Earlier GPU probes cover kernels; next full MATH stage is still pending.')
(out/'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(status=report['status'], checked_runs=len(checked), entry_sha256=report['entry_sha256'])))
