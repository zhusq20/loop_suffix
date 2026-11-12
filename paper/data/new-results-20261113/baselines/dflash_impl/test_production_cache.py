import unittest
import torch
from .production_cache import HuginnProductionCache


class CacheChecks(unittest.TestCase):
    def test_owned_storage_negative_slots_and_crop(self):
        cache = HuginnProductionCache(12)
        reference = {}
        gen = torch.Generator().manual_seed(17)
        for width in [4,1,3]:
            for slot in [0,1,2,3,-1,-2]:
                k = torch.randn(2,3,width,4,generator=gen)
                v = torch.randn(2,3,width,4,generator=gen)
                old = reference.get(slot)
                reference[slot] = (k.clone(),v.clone()) if old is None else (torch.cat([old[0],k],-2),torch.cat([old[1],v],-2))
                actual = cache.update(k,v,slot)
                k.zero_();v.zero_()  # Storing a view would corrupt the cache.
                for a,r in zip(actual,reference[slot]):
                    torch.testing.assert_close(a,r,atol=0,rtol=0)
        self.assertEqual(cache.get_seq_length(),8)
        cache.clear_last_k_entries(3)
        for slot in [0,1,2,3,-1,-2]:
            k = torch.randn(2,3,2,4,generator=gen)
            v = torch.randn(2,3,2,4,generator=gen)
            a,b = cache.update(k,v,slot)
            torch.testing.assert_close(a,torch.cat([reference[slot][0][...,:5,:],k],-2))
            torch.testing.assert_close(b,torch.cat([reference[slot][1][...,:5,:],v],-2))
        self.assertEqual(cache.get_seq_length(),7)

    def test_reject_missing_loop_cache(self):
        cache = HuginnProductionCache(12)
        x = torch.zeros(1,2,1,4)
        cache.update(x,x,0)
        cache.update(x,x,0)
        with self.assertRaises(RuntimeError):
            cache.update(x,x,1)


if __name__ == '__main__':
    unittest.main()
