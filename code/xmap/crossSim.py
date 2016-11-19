# -*- coding: utf-8 -*-
"""calculate extended similarity."""


class CrossSim:
    def __init__(self, method, num_atleast):
        """Initialize parameter"""
        self.method = method
        self.num_atleast = num_atleast

    def convert_tolist(self, items, chunk):
        return zip(*[iter(items)]*chunk)

    def mapping_item(self, line, mapping_dict, mapping_keys):
        """Use training dataset RDD as well as mapping dictionary to build an alterEgo profile.
        Args:
            line:               it has following structure (uid, iid, rating)
            mapping_dict:       it is dictionary that mapps source domain item to target domain item.
            mapping_key:        it contains all keys of mapping.
        """
        return (line[0], mapping_dict[line[1]], line[2], line[3]) if line[1] in mapping_keys else None

    def build_basic_profile(self, rdd, profile):
        """Use this function to build an item-based or a user-based profile.
        """
        if "user" in profile:
            return rdd.map(lambda line: (line[0], [(line[1], line[2], line[3])])).reduceByKey(lambda a, b: a + b)
        else:
            return rdd.map(lambda line: (line[1], [(line[0], line[2], line[3])])).reduceByKey(lambda a, b: a + b)

    def build_alterEgo(self, rdd, mapping_dict):
        """use this function to build alterEgo profile.
        Args:
            rdd:         rdd is the training dataset, it contains source domain item as well as target domain item.
                         The data structure of rdd is
            mapping:     it is a rdd that contain the information of mapping from source domain to target domain.
        Returns:
            alterEgoProfile in target domain.
                            It has structure as follows:
                            [(uid1, iid1, rating1, time1), (uid2, iid2, rating2, time2), ...]
        """
        mapping_keys = mapping_dict.keys()
        flatRDD  = rdd.flatMap(lambda (uid, pairs): [(uid, pair[0], pair[1], pair[2]) for pair in pairs])
        alterEgo_profile = flatRDD.filter(
            lambda line: "S:" in line[1]).map(
            lambda line: self.mappingItem(line, mapping_dict, mapping_keys)).filter(
            lambda line: line is not None)
        return flatRDD.filter(lambda line: "T:" in line[1]).union(alterEgo_profile)

    def get_info(self, formatedRDD):
        """get the information of RDD. It can be either information of item/user. Below is an example of user information.
        Args:
            formatedRDD:    A formated RDD.
                            [(uid, ((iid1, rating1), (iid2, rating2), ..., )), ...]
        Returns:
            Information of every user:
                            ((uid, (average, norm2, count)), )
        """
        def norm(pairs):
            """Calculate the norm 2 of the pairs
            Args:
                pairs:    [rating1, rating2, rating3, ...]
            Returns:
                norm of pairs' rating.
            """
            return float(sqrt(sum([pair[1] * pair[1] for pair in pairs])))

        def average(pairs):
            """Calculate the average of the pairs
            Args:
                pairs:    [rating1, rating2, rating3, ...]
            Returns:
                average of pairs' rating.
            """
            return float(sum([pair[1] for pair in pairs]) / len(pairs))
        return formatedRDD.map(lambda (uid, pairs): (uid, (average(pairs), norm(pairs), len(pairs))))

    def produce_pairwise(self, mergedRDD):
        """produce pairwise
        """
        def helper(iters):
            """A helper function to find item pairs
            """
            for uid, pairs in iters:
                for item1, item2 in combinations(pairs, 2):
                    yield ((item1[0], item2[0]), [(item1[1], item2[1], uid)])
        return mergedRDD.filter(lambda (uid, pairs): len(pairs) >= 2).mapPartitions(helper).reduceByKey(lambda a, b: a + b)

    def significance_weighting(self, sim, count):
        """Focus on the problem: users with few rated items in common will have very high similarities.
            min(count, numOfAtLeast) / numOfAtLeast
        Args:
            sim:    similarity of cosine similarity
            count:  number of co-rated items
        Returns:
            weighedSim:
        """
        return sim * min(count, self.numOfAtLeast) / self.numOfAtLeast

    def cosine(self, dot_product, norm2_product):
        """The cosine between two vectors A, B: dotProduct(A, B) / (norm(A) * norm(B))
        """
        numerator   = dot_product
        denominator = norm2_product
        return (numerator / (float(denominator))) if denominator else 0.0

    def cosine_Sim(self, pair, rating_pairs, info):
        """For each pair, return cosine similarity measure, along with co_raters_count
        Args:
            pair: (id1, id2)
            rating_pairs: ((rating1, rating2, id), ...)    Here, id can be either iid or uid.
        Returns:
            cosine similarity of pair (id1, id2)
        """
        def get_local_sensitivity(sim, rating, inner_product, norms, count):
            """Use this function to calculate the local sensitivity of pair (id1, id2). Please check the formula of local sensitivity.
            Args:
                rating:                 [r_xi * r_xj, ...].    Here, x \in all co-rated user.
                inner_product:          inner product of original item rating pair.
                norms:                  [norm2_x, norm2_y].    Here,
            """
            result = []
            for r in rating:
                m_inner_product = inner_product - r[0] * r[1]
                m_norms_1 = sqrt((norms[0] ** 2 - r[0] ** 2) * (norms[1] ** 2))
                m_norms_2 = sqrt((norms[0] ** 2 ) * (norms[1] ** 2 - r[1] ** 2))
                result += [self.significance_weighting(self.cosine(m_inner_product, m_norms_1), count - 1)]
                result += [self.significance_weighting(self.cosine(m_inner_product, m_norms_2), count - 1)]
            return max(abs(np.array(result) - sim))

        rating = [(float(rating_pair[0]), float(rating_pair[1]), float(rating_pair[0]) * float(rating_pair[1])) for rating_pair in rating_pairs]
        inner_product = sum(map(lambda line: line[2], rating))
        count = len(rating)
        norm_x = info.value[pair[0]][1]
        norm_y = info.value[pair[1]][1]
        cos_sim = self.significance_weighting(self.cosine(inner_product, norm_x * norm_y), count)
        LS = get_local_sensitivity(cos_sim, rating, inner_product, [norm_x, norm_y], count)
        return pair, [cos_sim, LS]

    def adjusted_cosine_sim(self, pair, rating_pairs, info):
        """For each pair, return ajusted cosine similarity measure, along with co_raters_count.
            If it is item-based, then pair is items' id, rating pairs is (pair1, pair2, uid). vice verse.
        Args:
            pair: (id1, id2)
            rating_pairs: ((rating1, rating2, id), ...)
            sizeOfAtLeast:
            info:        It is item-based, then info is the info of user, i.e., ((uid, (average, norm2, count)), ...)
            info:        It is user-based, then info is the info of item, i.e., ((iid, (average, norm2, count)), ...)
        Returns:
            adjusted cosine similarity of pair (id1, id2)
        """
        #
        def get_local_sensitivity(sim, rating_x, rating_y, inner_product, average, count):
            """Use this function to calculate the local sensitivity of pair (id1, id2). Please check the formula of local sensitivity.
                However, it will encounter different cases when using different method, details are as follow:
                    * When the similarity is item-based, then we need to hide the information of user.
                        * Thus, we can simply set the value of corresponding rating and user average as 0.
                        * We don't need to consider which user we hide, since both of them disappear from co-user at the same time.
                    * When the similarity is user-based, then we need to hide a rating of an item.
                        * However, the scheme is still simply since no matter which item's rating that we hide, it disappear from the co-item at the same time.
            Args:
                rating_x:
                rating_y:
                inner_product:          inner product of original item rating pair.
                average:
                count:
            """
            result = []
            for i in xrange(len(rating_x)):
                m_r_x, m_r_y, m_average = (rating_x, rating_y, average)
                m_average[i] = 0; m_r_x[i] = 0; m_r_y[i] = 0
                m_norm_x = sqrt(sum((m_r_x - m_average) ** 2))
                m_norm_y = sqrt(sum((m_r_y - m_average) ** 2))
                result += [self.significance_weighting(self.cosine(sum((m_r_x - average) * (m_r_y - average)), m_norm_x * m_norm_y), count - 1)]
            return max(abs(np.array(result) - sim))
        #
        rating_x = np.array([rating_pair[0] for rating_pair in rating_pairs])
        rating_y = np.array([rating_pair[1] for rating_pair in rating_pairs])
        average = np.array([info.value[rating_pair[2]][0] for rating_pair in rating_pairs])
        count = len(average)
        inner_product = sum((rating_x - average) * (rating_y - average))
        norm_x = sqrt(sum((rating_x - average) ** 2))
        norm_y = sqrt(sum((rating_y - average) ** 2))
        ad_cos_sim = self.significance_weighting(self.cosine(inner_product, norm_x * norm_y), count)
        LS = get_local_sensitivity(ad_cos_sim, rating_x, rating_y, inner_product, average, count)
        return pair, [ad_cos_sim, LS]

    def calculate_sim(self, item_profile, user_profile, item_info, user_info):
        """use specified method to calculate pair-pair similarity
        Args:
            mergedRDD:      merged RDD of test data that used to calculate pair-pair similarity
            method:         choose one method (e.g., "cosine", "adjust_cosine")
        Returns:
        """
        if "cosine_item" in self.method:
            pair_wise = self.produce_pairwise(user_profile)
            return pair_wise.map(
                lambda (id_pair, rating_pairs): self.cosineSim(
                    id_pair, rating_pairs, item_info)).cache()
        elif "cosine_user" in self.method:
            pair_wise = self.produce_pairwise(item_profile)
            return pair_wise.map(
                lambda (id_pair, rating_pairs): self.cosineSim(
                    id_pair, rating_pairs, user_info)).cache()
        elif "adjust_cosine_item" in self.method:
            pair_wise = self.produce_pairwise(user_profile)
            return pair_wise.map(
                lambda (id_pair, rating_pairs): self.adjustedCosineSim(
                    id_pair, rating_pairs, user_info)).cache()
        elif "adjust_cosine_user" in self.method:
            pair_wise = self.produce_pairwise(item_profile)
            return pair_wise.map(
                lambda (id_pair, rating_pairs): self.adjustedCosineSim(
                    id_pair, rating_pairs, item_info)).cache()
