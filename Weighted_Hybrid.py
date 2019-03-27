# Homework 3
# INFO 4871/5871, Spring 2019
# Robin Burke
# University of Colorado, Boulder

import logging
from lenskit.algorithms import Predictor
from lenskit.algorithms.basic import UnratedItemCandidateSelector

_logger = logging.getLogger(__name__)


class WeightedHybrid(Predictor):
    """

    """

    # HOMEWORK 3 TODO: Follow the constructor for Fallback, which can be found at
    # https: // github.com / lenskit / lkpy / blob / master / lenskit / algorithms / basic.py
    # Note that you will need to
    # -- Check for agreement between the set of weights and the number of algorithms supplied.
    # -- You should clone the algorithms with hwk3_util.my_clone() and store the cloned version.
    # -- You should normalize the weights so they sum to 1.
    # -- Keep the line that set the `selector` function.

    algorithms = []
    weights = []

    def __init__(self, algorithms, weights):
        """
        Args:
            algorithms: a list of component algorithms.  Each one will be trained.
            weights: weights for each component to combine predictions.
        """
        # HWK 3: Code here
        self.algorithms = algorithms
        
        self.weights = weights
        self.selector = UnratedItemCandidateSelector()
        list3 = []
        for i in self.weights:
            a = (i/sum(self.weights))*1
            list3.append(a)
        self.weights = list3
        
    def clone(self):
        return WeightedHybrid(self.algorithms, self.weights)

    # HOMEWORK 3 TODO: Complete this implementation
    # Will be similar to Fallback. Must also call self.selector.fit()
    def fit(self, ratings, *args, **kwargs):

        self.selector.fit(ratings)
        # HWK 3: Code here
        for algo in self.algorithms:
            algo.fit(ratings, *args, **kwargs)
        return self

    def candidates(self, user, ratings):
        return self.selector.candidates(user, ratings)

    # HOMEWORK 3 TODO: Complete this implementation
    # Computes the weighted average of the predictions from the component algorithms
    def predict_for_user(self, user, items, ratings=None):
        preds = None
        predall = []
        # HWK 3: Code here

        #preds = None

        for algo in self.algorithms:
            #_logger.debug('predicting for %d items for user %s', len(remaining), user)
            aps = algo.predict_for_user(user, items, ratings=ratings)
            predall.append(aps)
        
        preds = predall[0]*self.weights[0] + predall[1]*self.weights[1]

        return preds

    def __str__(self):
        return 'Weighted([{}])'.format(', '.join(self.algorithms))
