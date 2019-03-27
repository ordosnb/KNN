#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Homnework 2
# INFO 4871/5871, Spring 2019
# _Ben Niu__
# University of Colorado, Boulder

from typing import Dict

import pandas as pd
import numpy as np
import logging
from heapq import nlargest
#from __future__ import division
_logger = logging.getLogger(__name__)

class User_KNN:
    """
    User-user nearest-neighbor collaborative filtering with ratings. Not a very efficient implementation
    using data frames and tables instead of numpy arrays, which would be _much_ faster.

    Attributes:
        _ratings (pandas.DataFrame): Ratings with user, item, ratings
        _sim_cache (Dict of Dicts): a multi-level dictionary with user/user similarities pre-calculated
        _profile_means (Dict of float): a dictionary of user mean ratings
        _profile_lenghts (Dict of float): a dictionary of user profile vector lengths
        _item_means (Dict of float): a dictionary of item mean ratings
        _nhood_size (int): number of peers in each prediction
        _sim_threshold (float): minimum similarity for a neighbor
    """
    _ratings = None
    _sim_cache: Dict[int, Dict] = {}
    _profile_means: Dict[int, float] = {}
    _profile_lengths: Dict[int, float] = {}
    _item_means: Dict[int, float] = {}
    _nhood_size = 2
    _sim_threshold = 0

    def __init__(self, nhood_size, sim_threshold=0):
        """
        Args:
        :param nhood_size: number of peers in each prediction
        :param sim_threshold: minimum similarity for a neighbor
        """
        self._nhood_size = nhood_size
        self._sim_threshold = sim_threshold

    def get_users(self): return list(self._ratings.index.levels[0])

    def get_items(self): return list(self._ratings.index.levels[1])

    def get_profile(self, u): return self._ratings.loc[u]

    def get_profile_length(self, u): return self._profile_lengths[u]

    def get_profile_mean(self, u): return self._profile_means[u]

    def get_similarities(self, u): return self._sim_cache[u]

    def get_rating(self, u, i):
        """
        Args:
        :param u: user
        :param i: item
        :return: user's rating for item or None
        Issues a warning if the user has more than one rating for the same item. This indicates a problem
        with the data.
        """
        if (u,i) in self._ratings.index:
            maybe_rating = self._ratings.loc[u,i]
            if len(maybe_rating) == 1:
                return float(maybe_rating.iloc[0])
            else:  # More than one rating for the same item, shouldn't happen
                _logger.warning('Multiple ratings for an item - User %d Item %d', u, i)
                return None
        else: # User, item pair doesn't exist in index
            return None

    
    def compute_profile_length(self, u):
        """
        Computes the geometric length of a user's profile vector.
        :param u: user
        :return: length
        """
        distance =0
        for x in self.get_profile(u)['rating']:
            distance += x*x
            
        return np.sqrt(distance)


    
    def compute_profile_lengths(self):
        """
        Computes the profile length table `_profile_lengths`
        :return: None
        """
        for x in self.get_users():
            self._profile_lengths[x] =self.compute_profile_length(x)
        
        return None
        
    
    def compute_profile_means(self):
        """
        Computes the user mean rating table `_user_means`
        :return: None
        """
        for x in self.get_users():
            self._profile_means[x] = self.get_profile(x)['rating'].mean()
        
        return None
    
    def compute_item_means(self):
        """
        Computes the item means table `_item_means`
        :return: None
        """
        mean = []
        for i in self.get_items():
            for u in self.get_users():
                if self.get_rating(u,i) != None:
                    mean.append(self.get_rating(u,i))
                    self._item_means[i] = np.mean(mean,dtype=np.float64)
            mean = []
            
        return None
    
    def compute_similarity_cache(self):
        """
        Computes the similarity cache table `_sim_cache`
        :return: None
        """
        count = 1
        for u in self.get_users():
            # TODO Rest of code here
            self._sim_cache[u] = {}
            for i in self.get_users():
                
                self._sim_cache[u][i] = self.cosine(u,i)/(np.sqrt(self.get_profile_length(u)**2)*np.sqrt(self.get_profile_length(i)**2))
                
                
                if count % 10 == 0:
                    print ("Processed user {} ({})".format(u, count))
                count += 1

    
    def get_overlap(self, u, v):
        """
        Computes the items in common between profiles. Hint: use set operations
        :param u: user1
        :param v: user2
        :return: set intersection
        """
        a = set(self.get_profile(u)['rating'].index)

        b = set(self.get_profile(v)['rating'].index)
            
        return a&b 

    
    def cosine(self, u, v):
        """
        Computes the cosine between u and v vectors
        :param u:
        :param v:
        :return: cosine value
        """
        import math

        dot_prod = 0
        sumxx, sumyy = 0, 0
        overlap = self.get_overlap(u, v)
        
        for movieId in overlap:
            # TODO Rest of implementation
            x = self.get_profile(u)['rating'][movieId]
            y = self.get_profile(v)['rating'][movieId]
            if x or y > 0:
                #sumxx += np.dot(x,x)
                #sumyy += np.dot(y,y)
                dot_prod += np.dot(x,y)
          # np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))
        return dot_prod#/np.sqrt(sumxx*sumyy)
            #dot_prod = 0
            
       # return 0

    def fit(self, ratings):
        """
        Trains the model by computing the various cached elements. Technically, there isn't any training
        for a memory-based model.
        :param ratings:
        :return: None
        """
        self._ratings = ratings.set_index(['userId', 'movieId'])
        self.compute_profile_lengths()
        self.compute_profile_means()
        self.compute_similarity_cache()

    
    def neighbors(self, u, i):
        """
        Computes the user neighborhood
        :param u: user
        :param i: item
        :return: largest two simlarity
        """
        ud = [] # user who rated same item
        for x in self.get_users():
            for r in self.get_profile(x).index:
                if r == i:
                    ud.append(x)
        
        maxu=[] # find the two largest similarity of user u who rated item i
        for a in ud:
    
            fm =list((self.get_similarities(a).values()))[self.get_users().index(u)] #similarity of user u who rated item i
            maxu.append([fm,a])
        sim = []
        for s in sorted(maxu)[-self._nhood_size:]:
            sim.append(s[0])
                

        
        return sim

     
    def predict(self, u, i):
        """
        Predicts the rating of user for item
        :param u: user
        :param i: item
        :return: predicted rating
        """
        peers = self.neighbors(u, i)
        
        ud = [] # user who rated same item
        for x in self.get_users():
            for r in self.get_profile(x).index:
                if r == i:
                    ud.append(x)
        
        maxu=[] # find the k largest similarity of user u who rated item i
        for a in ud:
    
            fm =list((self.get_similarities(a).values()))[self.get_users().index(u)] #similarity of user u who rated item i
            maxu.append([fm,a])
        n = [] # store user's key
        for k in sorted(maxu)[-self._nhood_size:]:
    
            n.append(k[1])        

        
       
        c = 0
        dot,sumy = 0,0
        for r in n:
            
            x=  self.get_rating(r,i) # user r's rating
            y =  peers[c] # similarity of user r
            z = self.get_profile_mean(r) # user r's mean
            print(x,y,z)
            dot += (x-z)*y #prediction function
            sumy += y
            c +=1
            
        return (dot/sumy)+self.get_profile_mean(u) # result

    def predict_for_user(self, user, items, ratings=None):
        """
        Predicts the ratings for a list of items. This is used to calculate ranked lists.
        Note that the `ratings` parameters is required by the LKPy interface, but is not
        used by this algorithm.
        :param user:
        :param items:
        :param ratings:
        :return (pandas.Series): A score Series indexed by item.
        """
        scores = [self.predict(user, i) for i in items]

        return pd.Series(scores, index=pd.Index(items))

