# -*- coding: utf-8 -*-
"""A class to calculate similarity between items."""
from math import sqrt
from itertools import combinations

import numpy as np

from pyspark.sql import Row


class BaselinerSim:
    def __init__(self, method, num_atleast):
        """Initialize parameter."""
        self.method = method
        self.num_atleast = num_atleast

    def get_universal_user_info(self, dataRDD):
        """get the user information of RDD.
        Args:
            dataRDD: A RDD in the form of (uid, [(iid, rating, time)*])*
        Returns:
            a pair: (uid, (average, norm2))*
        """
        def helper(iters):
            """A helper function to get norm and average
            """
            def get_average(pairs):
                """A function used to get the average rating of each user.
                """
                return float(sum([pair[1] for pair in pairs]) / len(pairs))

            def get_norm2(pairs):
                """A function used to get the norm rating of each user.
                """
                return float(sqrt(sum([pair[1] * pair[1] for pair in pairs])))
            for uid, line in iters:
                yield uid, (get_average(line), get_norm2(line))
        return dataRDD.mapPartitions(helper)

    def get_universal_item_info(self, dataRDD, user_info):
        """get universal item information.
        Args:
            dataRDD: A RDD in the form of (uid, [(iid, rating, time)*])*
            user_info: (uid, (average, norm2))*
        Returns:
            (iid, (average, norm2, adjusted norm2, count))*
        """
        def to_item_based(info):
            """transform user-based to item-based.
            Returns:
                a pair: (iid, (uid, rating))*
            """
            uid, lines = info
            return [(iid, (uid, rating)) for (iid, rating, time) in lines]

        def cal_item_info(line):
            """calculate the info for each item."""
            iid, info = line
            return iid, (
                1.0 * info[0] / info[3],
                sqrt(info[1]),
                sqrt(info[2]),
                1.0 * info[3])

        return dataRDD.flatMap(
            lambda line: to_item_based(line)).combineByKey(
            lambda value: (
                value[1],
                value[1] ** 2,
                (value[1] - user_info.value[value[0]][0]) ** 2,
                1),
            lambda x, value: (
                x[0] + value[1],
                x[1] + value[1] ** 2,
                x[2] + (value[1] - user_info.value[value[0]][0]) ** 2,
                x[3] + 1),
            lambda x, y: (
                x[0] + y[0],
                x[1] + y[1],
                x[2] + y[2],
                x[3] + y[3])).map(
            lambda line: cal_item_info(line))

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

    def retrieve_path_info(self, item_pair, rating_pairs, item_info):
        """For each item-item pair, retrieve its path information
            based on mutuality and fractional mutuality.
        Args:
            item_pair: (item1, item2)
            rating_pairs: (rating1, rating2, uid)*
            item_info: {iid: (average, norm2, adjusted norm2, count)}
        """
        mutuality_l = [rating_pair for rating_pair in rating_pairs
                       if rating_pair[0] >= item_info.value[item_pair[0]][0]
                       and rating_pair[1] >= item_info.value[item_pair[1]][0]
                       ]
        mutuality_s = [rating_pair for rating_pair in rating_pairs
                       if rating_pair[0] < item_info.value[item_pair[0]][0]
                       and rating_pair[1] < item_info.value[item_pair[1]][0]
                       ]
        return float(len(mutuality_l) + len(mutuality_s))

    def calculate_cosine_sim(self, item_pair, rating_pairs, item_info):
        """For each item-item pair, return cosine sim with # co_raters.
        Args:
            item_pair: (item1, item2)
            rating_pairs: (rating1, rating2, uid)*
            item_info: {iid: (average, norm2, adjusted norm2, count)}
        Returns:
            (item1, item2), (cosine_sim, mutu, frac_mutu)
        """
        iid1, iid2 = item_pair
        pair_xy = [
            1.0 * rating_pair[0] * rating_pair[1]
            for rating_pair in rating_pairs]

        num_overlap_rating = len(pair_xy)
        inner_product = sum(pair_xy)
        norm_x = item_info.value[iid1][1]
        norm_y = item_info.value[iid2][1]

        cos_sim = self.significance_weighting(
            self.cosine(inner_product, norm_x * norm_y), num_overlap_rating)

        mutu = self.retrieve_path_info(
            item_pair, rating_pairs, item_info)
        frac_mutu = 1.0 * mutu / (
            item_info.value[iid1][3] +
            item_info.value[iid2][3] - num_overlap_rating)
        return item_pair, (cos_sim, mutu, frac_mutu)

    def calculate_adjusted_cosine_sim(
            self, item_pair, rating_pairs, item_info, user_info):
        """For each item-item pair, return adjusted cosine sim with # co_raters.
        Args:
            item_pair: (item1, item2)
            rating_pairs: (rating1, rating2, uid)*
            item_info: {iid: (average, norm2, adjusted norm2, count)}
            user_info: {uid: (average, norm2)}
        Returns:
            (item1, item2), (cosine_sim, mutu, frac_mutu)
        """
        iid1, iid2 = item_pair
        rating_x = np.array([rating_pair[0] for rating_pair in rating_pairs])
        rating_y = np.array([rating_pair[1] for rating_pair in rating_pairs])
        average = np.array([user_info.value[rating_pair[2]][0]
                            for rating_pair in rating_pairs])

        num_overlap_rating = len(average)
        inner_product = np.sum((rating_x - average) * (rating_y - average))
        norm_x = item_info.value[iid1][2]
        norm_y = item_info.value[iid2][2]

        ad_cos_sim = self.significance_weighting(
            self.cosine(inner_product, norm_x * norm_y), num_overlap_rating)

        mutu = self.retrieve_path_info(item_pair, rating_pairs, item_info)
        frac_mutu = 1.0 * mutu / (
            item_info.value[iid1][3] +
            item_info.value[iid2][3] - num_overlap_rating)
        return item_pair, (ad_cos_sim, mutu, frac_mutu)

    def produce_pairwise_items(self, dataRDD):
        """produce pairwise items."""
        def helper(iterations):
            """A helper function to find item pairs."""
            for uid, pairs in iterations:
                for item1, item2 in combinations(pairs, 2):
                    yield ((item1[0], item2[0]), [(item1[1], item2[1], uid)])
        return dataRDD.filter(
            lambda line: len(line[1]) >= 2).mapPartitions(helper)

    def calculate_item2item_sim(self, dataRDD, item_info, user_info):
        """use specified method to calculate item-item similarity."""
        def detect_domain(item_pair):
            """detect if there exist a cross domain item or not."""
            return 1 if (item_pair[0][:2] != item_pair[1][:2]) else 0

        def cosine_helper(iters):
            """A helper function used to assist the process of pairwiseItemsRDD.
            """
            for id_pair, lists in iters:
                sim = self.calculate_cosine_sim(id_pair, lists, item_info)
                if 0.0 not in sim[1]:
                    yield sim[0], sim[1] + (detect_domain(id_pair), )

        def adjusted_cosine_helper(iters):
            """A helper function used to assist the process of pairwiseItemsRDD.
            """
            for id_pair, lists in iters:
                sim = self.calculate_adjusted_cosine_sim(
                    id_pair, lists, item_info, user_info)
                if 0.0 not in sim[1]:
                    yield sim[0], sim[1] + (detect_domain(id_pair), )

        pairwiseItemsRDD = self.produce_pairwise_items(
            dataRDD).reduceByKey(lambda a, b: a + b)

        if "cosine" in self.method:
            return pairwiseItemsRDD.mapPartitions(cosine_helper)
        elif "ad_cos" in self.method:
            return pairwiseItemsRDD.mapPartitions(adjusted_cosine_helper)

    def get_item_sim(self, dataRDD):
        """For the RDD of sim pairs, groupby pair's first item as its key.
        Args:
            dataRDD: in the form of (item_pair, (sim, mutu, frac_mutu))*
        """
        def key_on_first_item(iters):
            """For each item-item pair, make the first item's id the key
            Args:
                item_pair:        (iid1, iid2)
                item_sim_data:    (sim, mutu, frac_mutu)
            """
            for (iid1, iid2), (sim, mutu, frac_mutu, label) in iters:
                yield iid1, [(iid2, sim, mutu, frac_mutu)]
                yield iid2, [(iid1, sim, mutu, frac_mutu)]
        return dataRDD.mapPartitions(
            key_on_first_item).reduceByKey(lambda a, b: a + b)

    def build_sim_DF(self, sim_pairsRDD):
        """convert sim RDD to DataFrame."""
        def helper(line):
            """helper function."""
            iid, info = line
            return [Row(
                        id1=iid[0], id2=iid[1],
                        sim=float(info[0]), mutu=info[1],
                        frac_mutu=float(info[2]), label=info[3]),
                    Row(id1=iid[1], id2=iid[0],
                        sim=float(info[0]), mutu=info[1],
                        frac_mutu=float(info[2]), label=info[3])]
        return sim_pairsRDD.flatMap(helper).toDF()
