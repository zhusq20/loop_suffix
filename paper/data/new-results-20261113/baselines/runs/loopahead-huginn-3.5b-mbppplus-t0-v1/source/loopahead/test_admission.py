import unittest
from .admission import AdmissionPolicy,confidence_bin,quality_counts


class AdmissionChecks(unittest.TestCase):
    def test_compact_suffix_matches_full_path_counts(self):
        events=[dict(type='lad',path=[1,2,3],distance=1,offset=1,confidence=.8),
                dict(type='lad',path=[1,7,3],distance=1,offset=2,confidence=.9),
                dict(type='native',path=[1,2,3,4],distance=3)]
        compact=[]
        for e in events:
            c=dict(e);path=c.pop('path');c.update(path_offset=1,path_suffix=path[1:]);compact.append(c)
        self.assertEqual(quality_counts([dict(tokens=[1,2,3],candidates=events)]),
                         quality_counts([dict(tokens=[1,2,3],candidates=compact)]))

    def test_path_matches_require_entire_prefix(self):
        tables=quality_counts([dict(tokens=[1,2,3],candidates=[
            dict(type='lad',path=[1,2],distance=1,offset=1,confidence=.8),
            dict(type='lad',path=[9,2,3],distance=1,offset=2,confidence=.9),
            dict(type='native',path=[1,2,3,4],distance=3)])])
        self.assertEqual(tables['offset']['1'],dict(success=1,count=1))
        self.assertEqual(tables['offset']['2'],dict(success=0,count=1))
        self.assertEqual(tables['native_distance']['3']['success'],0)

    def test_utility_and_empty_bin_backoff(self):
        record=dict(split='development_only',costs=dict(t_save_s=1.,t_head_s=.6,t_node_s=.5),
            quality=dict(offset={},distance_offset={},distance_offset_confidence={},native_distance={}))
        p=AdmissionPolicy(record)
        self.assertEqual(p.probability(8,2,.99),.5)
        self.assertEqual(p.call_length(8,3),0)
        self.assertEqual(p.keep_length(8,[.1,.8,1.]),0)
        record['costs'].update(t_head_s=.1,t_node_s=.1)
        p=AdmissionPolicy(record)
        self.assertEqual(p.call_length(8,3),3)
        self.assertEqual(p.keep_length(8,[.1,.8,1.]),3)
        self.assertEqual([confidence_bin(x) for x in [0,.249,.25,.5,.75,1]], [0,0,1,2,3,3])


if __name__=='__main__':
    unittest.main()
