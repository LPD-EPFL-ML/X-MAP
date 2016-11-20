# -*- coding: utf-8 -*-
"""calculate extended similarity."""

from itertools import combinations

import numpy as np


class CrossSim:
    def __init__(self, method, num_atleast):
        """Initialize parameter"""
        self.method = method
        self.num_atleast = num_atleast

    def mapping_item(self, line, mapping_dict):
        """Use traini.
        Args:
            line: in the form of (uid, iid, rating, rating time).
            mapping_dict: {source item: target item}.
            mapping_key: it contains all keys of mapping.
        """
        return (line[0], mapping_dict[line[1]], line[2], line[3]) \
            if line[1] in mapping_dict else None

    def build_alterEgo(self, trainRDD, mapping_dict):
        """use this function to build alterEgo profile.
        Args:
            rdd: training dataset, contains source/target domain item,
                in the form of `iid, rating, rating time.`
            mapping_dict: `{target item: source item}`.
        Returns:
            alterEgoProfile in target domain.
                It has structure as follows: (uid, iid, rating, time)*
        """
        dataRDD = trainRDD.flatMap(
            lambda line: [(line[0], l[0], l[1], l[2]) for l in line[1]])
        alterEgo_profile = dataRDD.map(
            lambda line: self.mapping_item(line, mapping_dict)).filter(
            lambda line: line is not None)
        return dataRDD.filter(
            lambda line: "T:" in line[1]).union(alterEgo_profile)

    def build_sthbased_profile(self, rdd, profile):
        """build an item-based or a user-based profile.
        Args:
            rdd: (uid, iid, rating, time)*
        """
        if "user" in profile:
            return rdd.map(
                lambda l: (l[0], [(l[1], l[2], l[3])])).reduceByKey(
                lambda a, b: a + b)
        elif "item" in profile:
            return rdd.map(
                lambda l: (l[1], [(l[0], l[2], l[3])])).reduceByKey(
                lambda a, b: a + b)

    def get_info(self, dataRDD):
        """get the information of RDD, either item or user.
        Args:
            dataRDD could either be userRDD or itemRDD.
            userRDD: (uid, (iid, rating, rating time)*)
            itemRDD: (iid, (uid, rating, rating time)*)
        Returns:
            info of the input RDD:
                (uid, (average, norm2, count))* or
                (iid, (average, norm2, count))* or
        """
        def norm2(ratings):
            """calculate the norm 2 of input ratings.
            Args:
                ratings: (iid, rating, rating time)*
            Returns:
                norm of ratings.
            """
            return np.sqrt(np.sum([rating[1] ** 2 for rating in ratings]))

        def average(ratings):
            """calculate the average of the ratings.
            Args:
                ratings: (iid, rating, rating time)*
            Returns:
                average of ratings.
            """
            return 1.0 * np.average([rating[1] for rating in ratings])

        def helper(line):
            """a helper function."""
            uid, ratings = line
            return uid, (average(ratings), norm2(ratings), len(ratings))
        return dataRDD.map(helper)

    def produce_pairwise(self, dataRDD):
        """produce pairwise."""
        def helper(iters):
            """find item pairs."""
            for uid, ratings in iters:
                for item1, item2 in combinations(ratings, 2):
                    yield (item1[0], item2[0]), [(item1[1], item2[1], uid)]
        return dataRDD.filter(
            lambda line: len(line[1]) >= 2).mapPartitions(
            helper).reduceByKey(
            lambda a, b: a + b)

    def significance_weighting(self, sim, count):
        """Deal with on the issues raise by: users with few rated items
            in common will have a high similarity.
        We have sim * min(count, num_atleast) / num_atleast.
        """
        return 1.0 * sim * min(count, self.num_atleast) / self.num_atleast

    def cosine(self, dot_product, norm2_product):
        """The cosine between two vectors A, B by the formula illustrated below
            dotProduct(A, B) / (norm(A) * norm(B))
        """
        return 1.0 * dot_product / (norm2_product) if norm2_product else 0.0

    def cosine_sim(self, line, info):
        """return cosine similarity measure, along with co_raters_count
        Args:
            line: (id1, id2), (rating1, rating2, id)*
            info: (id, (average, norm2, count))*
        Returns:
            [cosine similarity, local sensitivity] of pair (id1, id2)
        """
        def get_local_sensitivity(sim, rating, inner_product, norms, count):
            """calculate the local sensitivity of pair (id1, id2).
                Please check the formula of local sensitivity for more details.
            Args:
                rating: [r_xi * r_xj, ...]. Here, x \in all co-rated user.
                inner_product: inner product of original item rating pair.
                norms: [norm2_x, norm2_y].
            """
            result = []
            [norm_x, norm_y] = norms
            for r in rating:
                m_inner_product = inner_product - r[0] * r[1]
                m_norms_1 = np.sqrt((norm_x ** 2 - r[0] ** 2) * (norm_y ** 2))
                m_norms_2 = np.sqrt((norm_x ** 2) * (norm_y ** 2 - r[1] ** 2))
                result += [self.significance_weighting(
                    self.cosine(m_inner_product, m_norms_1), count - 1)]
                result += [self.significance_weighting(
                    self.cosine(m_inner_product, m_norms_2), count - 1)]
            return max(abs(np.array(result) - sim))

        (id1, id2), rating_pairs = line
        num_overlap_rating = len(rating_pairs)

        ratings = [
            (rating_pair[0], rating_pair[1], rating_pair[0] * rating_pair[1])
            for rating_pair in rating_pairs]
        inner_product = sum(map(lambda line: line[2], ratings))
        norm_x = info.value[id1][1]
        norm_y = info.value[id2][1]
        cos_sim = self.significance_weighting(
            self.cosine(inner_product, norm_x * norm_y), num_overlap_rating)
        local_sensitivity = get_local_sensitivity(
            cos_sim, ratings, inner_product,
            [norm_x, norm_y], num_overlap_rating)
        return (id1, id2), [cos_sim, local_sensitivity]

    def adjusted_cosine_sim(self, line, info):
        """return ajusted cosine similarity measure, along with co_raters_count
            If item-based, then pair is items' id,
            rating pairs is (pair1, pair2, uid). vice verse.
        Args:
            line: (id1, id2), (rating1, rating2, id)*
            info: (id, (average, norm2, count))*
        Returns:
            adjusted cosine similarity of pair (id1, id2)
        """
        def get_local_sensitivity(sim, rating_x, rating_y,
                                  inner_product, average, num_overlap_rating):
            """calculate the local sensitivity of pair (id1, id2).
                However, it will encounter different case:
                    * item-based sim: we need to hide the information of user.
                        * set the corresponding rating and user average to 0.
                        * no need to consider which user we hide, since both of
                        them disappear from co-user at the same time.
                    * user-based sim: we need to hide a rating of an item.
                        * However, the scheme is still simply the same,
                        since no matter which item's rating that we hide,
                        it disappear from the co-item at the same time.
            """
            result = []
            for i in range(len(rating_x)):
                m_rating_x, m_rating_y, m_average = rating_x, rating_y, average
                m_average[i], m_rating_x[i], m_rating_y[i] = 0, 0, 0
                m_norm_x = np.sqrt(np.sum((m_rating_x - m_average) ** 2))
                m_norm_y = np.sqrt(np.sum((m_rating_y - m_average) ** 2))
                result += [self.significance_weighting(
                    self.cosine(
                        sum((m_rating_x - average) * (m_rating_y - average)),
                        m_norm_x * m_norm_y), num_overlap_rating - 1)]
            return max(abs(np.array(result) - sim))

        (id1, id2), rating_pairs = line
        num_overlap_rating = len(rating_pairs)

        rating_x = np.array([rating_pair[0] for rating_pair in rating_pairs])
        rating_y = np.array([rating_pair[1] for rating_pair in rating_pairs])
        average = np.array(
            [info.value[rating_pair[2]][0] for rating_pair in rating_pairs])
        inner_product = np.sum((rating_x - average) * (rating_y - average))
        norm_x = np.sqrt(np.sum((rating_x - average) ** 2))
        norm_y = np.sqrt(np.sum((rating_y - average) ** 2))
        ad_cos_sim = self.significance_weighting(
            self.cosine(inner_product, norm_x * norm_y), num_overlap_rating)
        local_sensitivity = get_local_sensitivity(
            ad_cos_sim, rating_x, rating_y,
            inner_product, average, num_overlap_rating)
        return (id1, id2), [ad_cos_sim, local_sensitivity]

    def calculate_sim(self, item_profile, user_profile, item_info, user_info):
        """use specified method to calculate pair-pair similarity."""
        if "cosine_item" in self.method:
            pair_wise = self.produce_pairwise(user_profile)
            return pair_wise.map(
                lambda line: self.cosine_sim(line, item_info))
        elif "cosine_user" in self.method:
            pair_wise = self.produce_pairwise(item_profile)
            return pair_wise.map(
                lambda line: self.cosine_sim(line, user_info))
        elif "adjust_cosine_item" in self.method:
            pair_wise = self.produce_pairwise(user_profile)
            return pair_wise.map(
                lambda line: self.adjusted_cosine_sim(line, user_info))
        elif "adjust_cosine_user" in self.method:
            pair_wise = self.produce_pairwise(item_profile)
            return pair_wise.map(
                lambda line: self.adjusted_cosine_sim(line, item_info))
