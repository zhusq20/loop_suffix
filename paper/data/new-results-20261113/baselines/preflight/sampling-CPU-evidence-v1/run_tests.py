import hashlib,json,sys,time,unittest
from pathlib import Path
import torch
b=Path(sys.argv[1]);out=Path(sys.argv[2]);sys.path.insert(0,str(b))
manifest=json.loads((out/'source-manifest.json').read_text())
for name,digest in manifest.items():
 assert hashlib.sha256((b/name).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((out/'source'/name).read_bytes()).hexdigest()==digest
torch.set_num_threads(1)
suite=unittest.defaultTestLoader.loadTestsFromNames(['dflash_impl.test_decode','loopahead.test_sampling','loopahead.test_decode','loopahead.test_admission'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
assert result.testsRun==17 and result.wasSuccessful()
assert not torch.cuda.is_initialized()
for name,digest in manifest.items():
 assert hashlib.sha256((b/name).read_bytes()).hexdigest()==digest
(out/'tests.json').write_text(json.dumps(dict(status='passed_existing_sampling_and_scheduler_tests',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),torch=str(torch.__version__),cuda_initialized=torch.cuda.is_initialized(),source_sha256=manifest,finished_unix=time.time(),scope='Small-vocabulary exact conditional enumeration and Monte Carlo rejection correction, scheduler/EOS/cap/admission oracles. No new real-model statistical benchmark.'),indent=2)+'\n')
