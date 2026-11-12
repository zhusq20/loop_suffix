"""Load the real final narrow checkpoint in the deployment environment on CPU."""
import json
import os
from pathlib import Path
import sys
import time

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE/'third_party/dflash'))
from dflash_training_registry import sha
from provenance import freeze_sources

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert os.sched_getaffinity(0) == {80, 81, 82, 83}
import torch
import transformers
from safetensors import safe_open
from dflash_impl.projected_draft import inference_draft_class

torch.set_num_threads(4)
assert torch.__version__ == '2.6.0+cu126' and transformers.__version__ == '4.57.1'
assert not torch.cuda.is_initialized()
out = Path(__file__).parent
assert not (out/'report.json').exists()
audit_path = BASE/'training/dflash-huginn-narrow-formal-audit-v1/report.json'
audit = json.loads(audit_path.read_text())
assert audit['scope'] == 'formal' and audit['metrics']['updates'] == 8886 and audit['actual_exit_code'] == 0
checkpoint = Path(audit['checkpoint'])
weights_path = checkpoint/'model.safetensors'
assert sha(weights_path) == audit['weights_sha256'] == 'fde0e64955405f5f38de8566d09459db89a8547d367ab321edf38586324b3d48'
assert sha(checkpoint/'config.json') == audit['checkpoint_config_sha256']
started = time.time()
draft, loading = inference_draft_class().from_pretrained(checkpoint, torch_dtype=torch.bfloat16,
    attn_implementation='sdpa', output_loading_info=True, local_files_only=True)
draft.eval()
assert not any(loading.values()), loading
assert draft.config.hidden_size == 2048 and draft.target_hidden_size == 5280
assert draft.config.intermediate_size == 5632 and draft.config.num_hidden_layers == 5
assert draft.config.num_attention_heads == draft.config.num_key_value_heads == 16
assert draft.block_size == 16 and draft.target_layer_ids == [1, 126, 128, 130, 131]
assert sum(p.numel() for p in draft.parameters()) == 332621056
state = draft.state_dict()
with safe_open(str(weights_path), framework='pt', device='cpu') as saved:
    assert set(saved.keys()) == set(state) and len(state) == 60
    for name, tensor in state.items():
        assert tensor.device.type == 'cpu' and tensor.dtype == torch.bfloat16
        assert torch.equal(tensor, saved.get_tensor(name)), name
assert not torch.cuda.is_initialized()
freeze_sources(out, [Path(__file__), BASE/'dflash_impl/projected_draft.py',
    BASE/'third_party/dflash/dflash/model.py', BASE/'evaluate_projected_dflash.py',
    BASE/'configs/dflash-huginn-narrow-2048-v1.json'])
report = dict(status='final_checkpoint_deployment_CPU_load_passed_not_GPU_benchmark',
              pid=os.getpid(), started_unix=started, finished_unix=time.time(),
              python=sys.executable, torch=str(torch.__version__), transformers=transformers.__version__,
              checkpoint=str(checkpoint), weights_sha256=audit['weights_sha256'],
              training_audit_sha256=sha(audit_path), config_sha256=audit['checkpoint_config_sha256'],
              parameters=332621056, tensors=60, loading_info=loading,
              all_loaded_tensors_equal_saved_weights=True, all_tensors_on_CPU=True,
              cuda_initialized=torch.cuda.is_initialized(), source_sha256=sha(Path(__file__)),
              limitations=['CPU serialization/deployment compatibility of the final 8886-step checkpoint only.',
                           'No forward pass, CUDA model load, acceptance, speed or numerical-decoding equivalence measured.'])
(out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
