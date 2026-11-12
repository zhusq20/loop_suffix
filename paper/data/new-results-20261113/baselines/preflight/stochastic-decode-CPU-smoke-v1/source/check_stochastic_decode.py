"""CPU stochastic integration check against an independently enumerated tiny LM.

Exercises the real DFlash generation loop and LoopAhead tree scheduler. Synthetic
sequences and CPU runtimes are test evidence, never paper benchmark measurements.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1]/'results/loopfill-latency512-20261107/source/src'
sys.path.insert(0,str(ENGINE))
from dflash_impl.decode import generate
from loopahead.admission import AdmissionPolicy
from loopahead.decode import LoopAheadConfig, decode_steps
from packed_dflash_evidence import save, sha
from provenance import freeze_sources

PROMPT = [0,1]
CAP = 5
EOS = 2
TARGET = [(0.12,0.73,0.15),(0.64,0.24,0.12),(0.27,0.48,0.25)]
Q1 = (0.80,0.15,0.05)
Q2 = (0.08,0.82,0.10)


def target_probs(prefix):
    return TARGET[(sum(prefix)+len(prefix)) % len(TARGET)]


def target_logits(prefix):
    return torch.tensor([math.log(p) for p in target_probs(prefix)],dtype=torch.float32)


def enumerate_target():
    result = {}
    def visit(tokens, probability):
        if len(tokens) == CAP or (tokens and tokens[-1] == EOS):
            result[tuple(tokens)] = probability
            return
        for token,p in enumerate(target_probs(PROMPT+tokens)):
            visit(tokens+[token],probability*p)
    visit([],1.)
    assert math.isclose(sum(result.values()),1.,abs_tol=1e-14)
    return result


class Cache:
    def __init__(self):
        self.tokens = []
    def get_seq_length(self):
        return len(self.tokens)


class MarkovTarget:
    device = torch.device('cpu')
    max_seq = 100
    depth = 4
    def new_cache(self):
        return Cache()
    def crop(self, cache, length):
        assert 0 <= length <= len(cache.tokens)
        del cache.tokens[length:]
    def embedding(self, tokens):
        return torch.nn.functional.one_hot(tokens,3).float()
    def head(self, hidden):
        return hidden
    def forward(self, tokens, cache, features=True):
        logits = []
        for token in tokens[0].tolist():
            cache.tokens.append(token)
            logits.append(target_logits(cache.tokens))
        return SimpleNamespace(logits=torch.stack(logits).unsqueeze(0),
                               features=torch.zeros(1,len(logits),3),cache=cache)


class AdverseBlockDraft:
    block_size = 16
    mask_token_id = 0
    def __call__(self, *, target_hidden, noise_embedding, position_ids, past_key_values, **kwargs):
        n = noise_embedding.shape[1]
        size = target_hidden.shape[1]+n
        assert position_ids.shape[1] == size
        assert int(position_ids[0,0]) == past_key_values.get_seq_length()
        k = torch.zeros(1,1,size,1)
        past_key_values.update(k,k.clone(),0)
        # Parallel draft positions deliberately disagree with the target's
        # prefix-conditioned probabilities; the actual decoder supplies q.
        return torch.tensor([math.log(p) for p in Q1]).repeat(1,n,1)


class StochasticTreeOracle:
    def __init__(self, cfg, generator, admission):
        self.cfg,self.generator,self.admission = cfg,generator,admission
        self.backbone = SimpleNamespace(length=0)
        self.ready = {}
        self.committed = []

    def execute(self, call):
        name,args = call.name,call.args
        if name == 'synchronize':
            return None
        if name == 'prefill':
            assert args[0] == PROMPT
            self.backbone.length = len(PROMPT)
            token = int(torch.multinomial(torch.tensor(target_probs(PROMPT)),1,generator=self.generator))
            self.committed = [token]
            return token
        tree = args[0]
        if name == 'embed':
            for node in args[1]:
                self.ready[node.idx] = 0
            return None
        if name == 'advance':
            nodes = args[1]
            assert len(nodes) <= self.cfg.execute
            selected = {n.idx for n in nodes}
            for node in nodes:
                assert self.ready[node.idx] == node.step
                parent = node.parent
                while parent >= 0:
                    ancestor = tree.nodes[parent]
                    assert ancestor.alive
                    assert self.ready[parent] > node.step or (self.ready[parent] == node.step and parent in selected)
                    parent = ancestor.parent
            for node in nodes:
                self.ready[node.idx] += 1
            return None
        if name == 'lad_readout':
            _,groups,full = args
            reads,finals = {},{}
            for depth,group in groups:
                for node in group:
                    q = Q1 if depth == self.cfg.first else Q2
                    if depth == self.cfg.first:
                        node.shallow_state = torch.tensor([len(node.path)])
                    reads[node.idx] = (depth,torch.tensor([math.log(p) for p in q]))
            for node in full:
                finals[node.idx] = target_logits(PROMPT+list(node.path))
            return reads,finals
        if name == 'lad_propose':
            # Misleading draft paths predict EOS repeatedly. Only verification
            # may stop a sequence, including when this path is retained early.
            return [EOS]*args[3],[.95]*args[3]
        if name == 'lad_commit':
            _,node,token = args
            assert node.step == self.cfg.total
            assert tuple(self.committed) == node.path
            self.committed.append(token)
            self.backbone.length += 1
            return None
        raise AssertionError(name)

    def run(self):
        steps = decode_steps(self.backbone,PROMPT,self.cfg,CAP,{EOS},generator=self.generator,
            admission=self.admission,quality=None if self.admission is None else self.admission.quality)
        value = None
        while True:
            try:
                call = steps.send(value)
            except StopIteration as done:
                tokens,stats = done.value
                assert tokens == self.committed
                assert stats['tokens'] == len(tokens)-1
                assert stats['max_rows'] <= self.cfg.execute and stats['peak_live_nodes'] <= self.cfg.retain
                return tokens,stats
            value = self.execute(call)


def statistical_check(counts, expected, samples, families):
    assert sum(counts.values()) == samples and set(counts) <= set(expected)
    # A fixed family-wise Bernstein bound, selected before seeing observations.
    alpha = 1e-6
    logterm = math.log(2*len(expected)*families/alpha)
    details = []
    for sequence,p in expected.items():
        empirical = counts[sequence]/samples
        bound = math.sqrt(2*p*(1-p)*logterm/samples)+2*logterm/(3*samples)
        details.append(dict(sequence=list(sequence),expected_probability=p,count=counts[sequence],
                            empirical_probability=empirical,absolute_error=abs(empirical-p),bound=bound))
    assert all(r['absolute_error'] <= r['bound'] for r in details), 'Observed output frequencies exceeded the predeclared bound.'
    return dict(samples=samples,outcomes=len(expected),familywise_alpha=alpha,
        max_absolute_error=max(r['absolute_error'] for r in details),
        total_variation=0.5*sum(r['absolute_error'] for r in details),outcomes_detail=details)


def run(out, samples):
    assert samples >= 1 and not out.exists()
    torch.set_num_threads(1)
    out.mkdir(parents=True)
    sources = freeze_sources(out,[Path(__file__),BASE/'dflash_impl/decode.py',BASE/'loopahead/decode.py',
        BASE/'loopahead/sampling.py',BASE/'loopahead/admission.py',BASE/'packed_dflash_evidence.py',BASE/'provenance.py'])
    engine_files = ['loop_budget_decode.py','loop_prob_tree.py','loop_decode_steps.py']
    engine_hashes = {n:sha(ENGINE/'llm_memory_editability'/n) for n in engine_files}
    policy = AdmissionPolicy(dict(split='development_only',costs=dict(t_save_s=1.,t_head_s=.55,t_node_s=.1),
        quality=dict(offset={},distance_offset={},distance_offset_confidence={},native_distance={})))
    configurations = [('dflash-block16',None,None),
        ('loopahead-ouro-default',LoopAheadConfig(temperature=1.),policy),
        ('loopahead-ouro-tight',LoopAheadConfig(temperature=1.,retain=3,execute=2),policy),
        ('loopahead-huginn',LoopAheadConfig(total=32,first=4,second=8,future=2,guard=1,stride=2,family='huginn',temperature=1.),None)]
    expected = enumerate_target()
    results = {}
    save(out/'plan.json',dict(samples_per_configuration=samples,prompt=PROMPT,max_new_tokens=CAP,eos=EOS,
        target=TARGET,q1=Q1,q2=Q2,configurations={name:None if cfg is None else asdict(cfg) for name,cfg,_ in configurations},
        admission_fixture=policy.record,familywise_alpha=1e-6,engine_sha256=engine_hashes,
        source_sha256=sources,scope='Synthetic CPU integration only; no benchmark measurements'))
    started = time.time()
    for config_index,(name,cfg,admission) in enumerate(configurations):
        counts,counters = Counter(),Counter()
        for seed in range(samples):
            actual_seed = 812800+config_index*1000000+seed
            if cfg is None:
                result = generate(MarkovTarget(),torch.tensor([PROMPT]),CAP,[EOS],
                                  draft=AdverseBlockDraft(),temperature=1.,seed=actual_seed)
                tokens,stats = result.tokens,result.stats
                counters['accepted_draft_tokens'] += sum(stats['accepted_draft_tokens'])
                counters['target_verifications'] += stats['target_verifications']
            else:
                tokens,stats = StochasticTreeOracle(cfg,torch.Generator().manual_seed(actual_seed),admission).run()
                for field in ['lad_calls','lad_tokens','first_proposals','second_proposals','accepted_first',
                              'accepted_second','fallback_tokens','cascade_commits','native_budget_rejections','retention_evictions']:
                    counters[field] += stats[field]
            assert 1 <= len(tokens) <= CAP and (EOS not in tokens or tokens[-1] == EOS)
            counts[tuple(tokens)] += 1
        result = statistical_check(counts,expected,samples,len(configurations))
        result['execution_counters'] = dict(counters)
        results[name] = result
        save(out/'progress.json',results)
        print(name,dict(samples=samples,max_absolute_error=result['max_absolute_error'],
                        total_variation=result['total_variation'],counters=dict(counters)),flush=True)
    assert not torch.cuda.is_initialized()
    for name,digest in sources.items():
        assert sha(BASE/name) == digest
    for name,digest in engine_hashes.items():
        assert sha(ENGINE/'llm_memory_editability'/name) == digest
    save(out/'report.json',dict(status='passed_stochastic_CPU_integration',samples_per_configuration=samples,
        configurations=results,source_sha256=sources,engine_sha256=engine_hashes,torch=str(torch.__version__),
        cuda_initialized=False,started_unix=started,finished_unix=time.time(),new_gpu_runs=0,
        limitations=['Synthetic vocabulary of three tokens and generation cap5, not real-model quality or speed.',
                     'Finite Monte Carlo with fixed seeds and a predeclared simultaneous bound, not a universal distribution proof.',
                     'Full target compute is replaced by a prefix-conditioned oracle; actual model numerical differences are checked separately.']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=8000)
    args = parser.parse_args()
    run(args.out.resolve(),args.samples)
