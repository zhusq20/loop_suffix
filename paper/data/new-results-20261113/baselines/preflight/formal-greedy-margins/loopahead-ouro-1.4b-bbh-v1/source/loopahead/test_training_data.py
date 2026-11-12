import unittest
import torch
from .training_data import FeatureStore


class AnchorChecks(unittest.TestCase):
    def test_state_label_alignment_and_end_mask(self):
        store = FeatureStore.__new__(FeatureStore)
        store.rows = [dict(source_id='synthetic/train/0', pool='math', tokens=list(range(100)),
            prompt_tokens=list(range(70)), output_tokens=list(range(70,100)),
            states=torch.arange(100,dtype=torch.bfloat16)[:,None].repeat(1,2),
            anchor_start=70,anchor_stop=99)]
        store.eligible = [0]
        context, anchor, labels, _ = store.batch([(0,70),(0,98)], 'cpu')
        self.assertEqual(context[0,0,0],6)
        self.assertEqual(context[0,-1,0],69)
        self.assertEqual(anchor.tolist(),[70,98])
        self.assertEqual(labels[0].tolist(),list(range(71,78)))
        self.assertEqual(labels[1].tolist(),[99]+[-100]*6)
        # Short trajectories sample with replacement, without admitting prompt
        # labels or the terminal position with no future label.
        first = store.anchors(32,0,True)
        self.assertEqual(first,store.anchors(32,0,True))
        self.assertEqual(len(first),32)
        self.assertTrue(all(70 <= t < 99 for _,t in first))
        self.assertNotEqual(first,store.anchors(32,1,True))


if __name__ == '__main__':
    unittest.main()
