"""Regression tests for exact half-open boundaries, independent of stored pools."""
import unittest
import numpy as np
from core import interval_frame_indices, pool_words


class TimeIntervals(unittest.TestCase):
    def test_ctc_boundaries_against_integer_oracle(self):
        centers=160*np.arange(6000)+200
        times=centers/16000
        # Include the formerly failing decimal representation 3.3225000000000002.
        for first in range(1,2500):
            start=round((first*.02+.0025)*16000)
            end=round(((first+6)*.02+.0225)*16000)
            result=interval_frame_indices(times,start,end)
            np.testing.assert_array_equal(result,np.arange(2*first-1,2*(first+6)+1))

    def test_pool_includes_left_excludes_right(self):
        times=(160*np.arange(500)+200)/16000
        audio=np.arange(500,dtype=np.float32)[:,None]
        vt=np.arange(125)/25
        words=[dict(word='your',s=3.3225000000000002,e=3.4625,confidence=1.)]
        result=pool_words(words,audio,times,np.ones((125,35)),vt,np.ones(125,bool))
        self.assertEqual(words[0]['a_frames'],list(range(331,345)))
        self.assertEqual(float(result['audio'][0,0]),337.5)

    def test_empty_window_fallback_and_no_face(self):
        at=np.array([200,360])/16000
        vt=np.array([0.,.04])
        words=[dict(word='short',s=.019,e=.021,confidence=1.)]
        result=pool_words(words,np.array([[1.],[3.]]),at,np.ones((2,35)),vt,np.zeros(2,bool))
        self.assertEqual(words[0]['a_frames'],[1])
        self.assertEqual(words[0]['v_frames'],[0])  # exact midpoint tie
        self.assertTrue(result['fallback'].all())
        self.assertFalse(result['availability_mask'][0,2])
        np.testing.assert_array_equal(result['vision'],np.zeros((1,35)))


if __name__=='__main__':unittest.main(verbosity=2)
