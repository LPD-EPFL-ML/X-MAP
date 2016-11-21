# -*- coding: utf-8 -*-
"""a final recommender."""
import numpy as np


class Recommendation:
    def __init__(self, alpha, method):
        """Initialize the parameters
        Args:
            alpha: parameter that used to define temporal Relevance,
                control the decaying rate.
            method: the method used for recommendation,
                it will decide if we should add decay.
        """
        self.alpha = alpha
        self.method = method

    def bound_rating(self, rating):
        """bound the value of rating.
            If the predicted rating is out of range,
            i.e., < 0 or > 5, then, adjust it to the closest bound.
        """
        return 1.0 * max(0, min(int(rating + 0.5), 5))

    def user_based_prediction(self, line, rating_bd, sim_bd, user_bd):
        """Use this function to predict the rating of item for this user.
        Args:
            line: (uid, pairs)
                where pairs in the format of (iid, rating, time)*
            rating_bd: broadcast of {uid: [(iid1, rating1, time1)*]}*
            sim_bd: broadcast of {uid: [(uid, sim)*]}*
            user_bd: broadcast pf {uid: (average, norm, count)}*
        """
        def prediction(pair, uid_allneighbor_info, user_bd, uid):
            """do the prediction. It can either add decay rate or not,
                which is decided by `method`.
            Args:
                pair: (iid, rating, time)
                uid_allneighbor_info: (uid, sim, rating_record)*
                average_uid: average rating of current uid.
            """
            iid, real_rating, time = pair
            average_uid_rating = user_bd.value[uid][0]
            sim_rating = []
            for info in uid_allneighbor_info:
                uid, sim, ratings = info
                sim_rating += [
                    (rating[0], sim, rating[1] - average_uid_rating)
                    for rating in ratings if iid in rating[0]]

            if len(sim_rating) != 0:
                sim_rating = [
                    (line[0], line[1] * line[2], abs(line[1]))
                    for line in sim_rating]
                predicted_rating = average_uid_rating + sum(
                    map(lambda line: line[1], sim_rating)) / sum(
                    map(lambda line: line[2], sim_rating))
            else:
                predicted_rating = average_uid_rating
            return iid, real_rating, self.bound_rating(predicted_rating)

        uid, pairs = line
        uid_allneighbor_info = [
            (u[0], u[1], rating_bd.value[u[0]]) for u in sim_bd.value[uid]]
        return uid, [prediction(
            pair, uid_allneighbor_info, user_bd, uid) for pair in pairs]

    def user_based_recommendation(
            self, test_dataRDD,
            user_based_dict_bd, userbased_sim_pair_dict_bd, user_info_bd):
        """user-based recommendation.
            For privacy protection purpose, no decay rate is allowed.
        """
        sim_pair_dict_keys = set(userbased_sim_pair_dict_bd.value.keys())
        return test_dataRDD.filter(
            lambda line: line[0] in sim_pair_dict_keys).map(
            lambda line: self.user_based_prediction(
                line, user_based_dict_bd,
                userbased_sim_pair_dict_bd, user_info_bd))

    def item_based_prediction(self, line, rating_bd, sim_bd, item_bd):
        """predict the rating of item for a specific user.
        Args:
            line: (uid, pairs),
                where pairs: in the format of (iid, rating, time)*
            rating_bd: broadcast of {uid: [(iid1, rating1, time1)*]}
            sim_bd: broadcast of {iid: [(iid, sim)*]}
            item_bd: broadcast of {iid: (average, norm, length)}
        """
        def sort_by_time(pairs):
            """For each user, sort its rating records based on its datetime.
                More specifically, if time_a > time_b,
                    then: time_a <- x, time_b <- x + 1.
            """
            pairs = sorted(pairs, key=lambda line: line[2], reverse=False)
            order = 0
            out = []
            for i in range(len(pairs)):
                if i != 0 and pairs[i][2] == pairs[i - 1][2]:
                    out += [(pairs[i][0], pairs[i][1], order)]
                else:
                    order += 1
                    out += [(pairs[i][0], pairs[i][1], order)]
            return out

        def f_decay(cur, t_ui):
            return np.exp(- self.alpha * (cur - t_ui))

        def add_decay(pairs):
            """add decay rate to the pairs.
            Args:
                pairs:    sim * rating, sim, time
            """
            new_pairs = sort_by_time(pairs)
            current_time = max(map(lambda line: line[2], new_pairs)) + 1
            final_pairs = [
                (pair[0] * f_decay(current_time, pair[2]),
                 pair[1] * f_decay(current_time, pair[2]))
                for pair in new_pairs]
            return sum(map(lambda line: line[0], final_pairs)) / sum(
                map(lambda line: line[1], final_pairs))

        def prediction(uid, pair, rating_bd, sim_bd, item_bd):
            """do the prediction. It can either add decay rate or not,
                which is decided by `method`.
            """
            iid, real_rating = pair[0], pair[1]
            if iid not in sim_bd.value.keys():
                return ()
            iid_neighbors = [
                (i[0], i[1], rating_bd.value[i[0]]) for i in sim_bd.value[iid]]
            average_iid_rating = item_bd.value[iid][0]
            sim_rating = []
            for info in iid_neighbors:
                sim_rating += [
                    (iid, info[1], i[1] - item_bd.value[info[0]][0], i[2])
                    for i in info[2] if uid in i[0]]
            if len(sim_rating) != 0:
                sim_ratings = [
                    (line[1] * line[2], abs(line[1]), line[3])
                    for line in sim_rating]
                predicted_rating_no_decay = average_iid_rating + sum(
                    map(lambda line: line[0], sim_ratings)) / sum(
                    map(lambda line: line[1], sim_ratings))
                predicted_rating_decay = \
                    average_iid_rating + add_decay(sim_ratings)
            else:
                predicted_rating_no_decay = average_iid_rating
                predicted_rating_decay = average_iid_rating
            return iid, real_rating, \
                self.bound_rating(predicted_rating_no_decay), \
                self.bound_rating(predicted_rating_decay)

        uid, pairs = line
        return uid, [
            prediction(uid, pair, rating_bd, sim_bd, item_bd)
            for pair in pairs]

    def item_based_recommendation(
            self, test_dataRDD,
            item_based_dict_bd, itembased_sim_pair_dict_bd, item_info_bd):
        """item-based Recommendation.
            it calculate rating with decay rate as well as without decay rate.
        """
        return test_dataRDD.map(
            lambda line: self.item_based_prediction(
                line, item_based_dict_bd,
                itembased_sim_pair_dict_bd, item_info_bd))

    def calculate_mae(self, rdd):
        """calculate MAE."""
        def helper(line, index=2):
            uid, pairs = line
            return uid, [
                abs(pair[1] - pair[index]) for pair in pairs if pair is not ()]

        if "user" in self.method:
            result = rdd.map(helper).map(
                lambda line: np.array([sum(line[1]), len(line[1])])).reduce(
                lambda a, b: a + b)
            to_return = str(result[0] / result[1])
        else:
            nodelay_result = rdd.map(helper).map(
                lambda line: np.array([sum(line[1]), len(line[1])])).reduce(
                lambda a, b: a + b)
            delay_result = rdd.map(lambda line: helper(line, index=3)).map(
                lambda line: np.array([sum(line[1]), len(line[1])])).reduce(
                lambda a, b: a + b)
            to_return = str(1.0 * nodelay_result[0] / nodelay_result[1]) \
                + '; ' + str(delay_result[0] / delay_result[1])
        return to_return
