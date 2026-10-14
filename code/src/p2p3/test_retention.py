"""Regression: a plotted retention curve must integrate to the reported AURC."""
import unittest
import numpy as np
import pandas as pd
from final_tables import retention_curve


class RetentionAggregationTest(unittest.TestCase):
    def test_ratio_before_model_and_type_average(self):
        rows=[]
        for seed,base,values in [(17,1.,[1.,3.]),(29,2.,[4.,8.])]:
            rows.append(dict(model_seed=seed,family='clean',kind='clean',rate=0.,mae=base))
            for kind,value in zip(['T','A'],values):
                # Two placements must be averaged before dividing by their mean.
                for placement in [.5,1.5]:
                    rows.append(dict(model_seed=seed,family='rate',kind=kind,rate=1.,mae=value*placement))
        curve=retention_curve(pd.DataFrame(rows),'mae')
        expected_endpoint=np.mean([1.,1/3,1/2,1/4])
        self.assertAlmostEqual(curve.loc[0.],1.)
        self.assertAlmostEqual(curve.loc[1.],expected_endpoint)
        self.assertAlmostEqual(np.trapezoid(curve,curve.index),np.mean([(1+x)/2 for x in [1.,1/3,1/2,1/4]]))
        self.assertNotAlmostEqual(curve.loc[1.],1.5/4.)


if __name__=='__main__':unittest.main()
