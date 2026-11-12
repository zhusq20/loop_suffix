import pytest
from llm_memory_editability.loop_prob_tree import Calibration, ProbConfig
from llm_memory_editability.loop_score_ablation import RawScores, SimpleScores
from llm_memory_editability.loop_prob_decode import prob_tree_decode
from test_loop_prob_decode import ProbFakeLoop, ProbFakeSuffix, calibration
from test_loop_tree_decode import greedy


def test_raw_uses_latest_probability_and_missing_top8_is_zero():
    c=RawScores()
    assert c.shallow_prob(2,0,.3,True)==.3
    assert c.joint_prob(2,7,0,.3,False)==.3
    assert c.joint_prob(2,0,8,0.,True)==0.
    assert c.suffix_prob_insitu(.8,20,True,8)==.8


def test_simple_pools_features_and_uses_smoothing_without_minimum_count():
    base=Calibration(shallow={(1,0,6,False):[2,3],(1,1,6,True):[0,2],(1,8,0,True):[9,10]},
                     insitu={(6,2,False,0):[1,2],(6,8,True,3):[1,3]})
    c=SimpleScores(base)
    assert c.shallow_prob(1,0,.35,False)==pytest.approx(3/7)
    assert c.shallow_prob(1,7,.35,True)==pytest.approx(3/7)
    assert c.shallow_prob(1,0,.95,False)==pytest.approx(3/7)
    assert c.shallow_prob(1,8,0.,True)==0
    assert c.suffix_prob_insitu(.35,2,False,0)==pytest.approx(3/7)
    assert c.suffix_prob_insitu(.35,90,True,9)==pytest.approx(3/7)
    assert c.suffix_prob(.95,20)==pytest.approx(3/7)


@pytest.mark.parametrize('scoring',['raw','simple','four'])
@pytest.mark.parametrize('tau',[.003,.01,.03,.1,.3])
def test_scoring_changes_tree_not_greedy_output(scoring,tau):
    base=calibration((1,2))
    c={'raw':RawScores(),'simple':SimpleScores(base),'four':base}[scoring]
    prompt=[1,2,3,4,5,6,1,2]
    config=ProbConfig('ablation',4,1,(1,2),tau=tau,joint=True,prune=True,suffix_calibration='insitu')
    out,_=prob_tree_decode(ProbFakeLoop(4,1),prompt,config,c,40,set(),ProbFakeSuffix(),0)
    assert out==greedy(prompt,40,set())
