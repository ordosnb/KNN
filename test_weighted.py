# Homework 3 solution
# INFO 4871/5871, Spring 2019
# Robin Burke
# University of Colorado, Boulder

from Weighted_Hybrid import WeightedHybrid
import unittest

import pandas as pd
from lenskit import batch
from Fallbacks import UserUserFallback, ItemItemFallback

class test_WeightedHybrid(unittest.TestCase):

    _algo = None
    _ratings = None
    _pred_tests = pd.DataFrame([(1,5), (2,1), (3,4), (3,5), (4,1), (4,2), (5,2), (5,4)],
                    columns=['user', 'item'])

    def setUp(self):
        user_comp = UserUserFallback(2, min_nbrs=1)
        item_comp = ItemItemFallback(2, min_nbrs=1)
        self._algo = WeightedHybrid([user_comp, item_comp], [1,1])

        self._ratings = pd.read_csv('test.csv')
        self._ratings.columns = ['user','item','rating']

        self._algo.fit(self._ratings)

    def test_normalize_weights(self):
        weights = self._algo.weights
        self.assertAlmostEqual(weights[0], 0.5, 3, 'Weights not normalized.')

    # Tests that all of the missing values in the ratings matrix are predicted correctly
    def test_predict(self):
        comp1 = self._algo.algorithms[0]
        comp2 = self._algo.algorithms[1]
        hybrid = self._algo

        pred1 = batch.predict(comp1, self._pred_tests)
        pred2 = batch.predict(comp2, self._pred_tests)
        hybrid = batch.predict(hybrid, self._pred_tests)

        pred_lst = 0.5 * pred1['prediction'] + 0.5 * pred2['prediction']
        algo_lst = hybrid['prediction']

        preds = zip(pred_lst, algo_lst)

        for pred, actual in preds:
            self.assertAlmostEqual(pred, actual, 3, 'Prediction does not match components.')

    def test_more_components(self):
        user_comp = UserUserFallback(2, min_nbrs=1)
        test_algo = WeightedHybrid([user_comp, user_comp, user_comp, user_comp, user_comp], [1,2,3,4,5])
        self.assertEqual(len(test_algo.algorithms), 5, "Multiple algorithms not supported")
        self.assertAlmostEqual(test_algo.weights[2], 0.2, 5, "Weights not normalized correctly")
