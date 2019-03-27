import logging
from lenskit.algorithms import Predictor
from lenskit.algorithms.basic import Fallback, Bias
from lenskit import util
from lenskit.algorithms.user_knn import UserUser
from lenskit.algorithms.item_knn import ItemItem
from hwk3_util import my_clone

class UserUserFallback(Fallback):

    def __init__(self, nnbrs, min_nbrs=1, min_sim=0, center=True, aggregate='weighted-average'):
        algo = UserUser(nnbrs, min_nbrs, min_sim, center, aggregate)
        fallback = Bias()
        Fallback.__init__(self, [algo, fallback])

    def clone(self):
        uualg = self.algorithms[0]
        return UserUserFallback(uualg.nnbrs, uualg.min_nbrs, uualg.min_sim, uualg.center, uualg.aggregate)


class ItemItemFallback(Fallback):

    def __init__(self, nnbrs, min_nbrs=1, min_sim=1e-06, save_nbrs=None, center=True, aggregate='weighted-average'):
        algo = ItemItem(nnbrs, min_nbrs, min_sim, save_nbrs, center, aggregate)
        fallback = Bias()
        Fallback.__init__(self, [algo, fallback])

    def clone(self):
        iialg = self.algorithms[0]
        return ItemItemFallback(iialg.nnbrs, iialg.min_nbrs, iialg.min_sim, iialg.save_nbrs,
                                iialg.center, iialg.aggregate)

