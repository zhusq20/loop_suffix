from pathlib import Path
import sys,json,torch
root=Path(__file__).resolve().parents[1]
protocol=json.loads((root/'trace-protocol.json').read_text())
source=Path(protocol['teacher_source'])
sys.path[:0]=[str(source),str(source/'src'),str(source/'scripts')]
from baselines.dflash.target import LoopTarget
from baselines.dflash.packed_target import PackedDFlashTarget
from transformers import AutoTokenizer
name=sys.argv[1]
family='huginn' if name.startswith('huginn') else 'ouro'
torch.set_num_threads(1)
torch.manual_seed(0)
native=LoopTarget.load(protocol['model_paths'][name],family,4096)
packed=PackedDFlashTarget(native)
tokenizer=AutoTokenizer.from_pretrained(protocol['model_paths'][name],trust_remote_code=True,local_files_only=True)
trace=json.loads((root/'traces/spider-dev-0129.json').read_text())
records=[]
with torch.inference_mode():
 for call in trace['calls'][:2]:
  ids=torch.tensor([call['tokenizations'][name]['prompt_token_ids']],device='cuda')
  expected=native.forward(ids,features=True)
  actual=packed.forward(ids,features=True)
  a,b=expected.logits[:,-1].float(),actual.logits[:,-1].float()
  record=dict(turn=call['assistant_turn'],prompt_tokens=ids.shape[1],native_top1=a.argmax().item(),packed_top1=b.argmax().item(),logit_max_abs=(a-b).abs().max().item(),logit_mean_abs=(a-b).abs().mean().item(),feature_mean_abs=(expected.features.float()-actual.features.float()).abs().mean().item())
  records.append(record)
  del expected,actual,a,b
 packed.close()
 native.close()
(root/'checks'/(name+'-native-audit.json')).write_text(json.dumps(dict(model=name,records=records),indent=2))
print(json.dumps(records))
