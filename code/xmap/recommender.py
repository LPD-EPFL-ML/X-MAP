# -*- coding: utf-8 -*-
"""a final recommender."""


class Recommendation:
    def __init__(self, alpha, method):
        """Initialize the parameters
        Args:
            k:          The size of private neighborhood
            alpha:      Parameter used to define temporal Relevance, controls the decaying rate.
        """
        self.alpha = alpha
        self.method = method

    def modify_rating(self, rating):
        """If the predicted rating is out of range, i.e., < 0 or > 5, then, adjust it to the closest bound.
        """
        return max(0, min(int(rating + 0.5), 5))

    def user_based_prediction(self, uid, pairs, rating_bd, sim_bd, user_bd):
        """Use this function to predict the rating of item for this user.
        Args:
            uid:
            pairs:             in the format of [(iid, rating, time), ...]
            rating_bd:         broadcast [uid: [(iid1, rating1, time1), ...]
            sim_bd:            broadcast [uid: [(uid1, sim1), (uid2, sim2), ...]]
            user_bd:           broadcast [uid: (average, norm, length)]
        """
        def prediction(pair, all_uid_neigh_info, user_bd, uid):
            """Use this function to do the prediction. It can either add decay rate or not, which is decided by "method".
            Args:
                pair:                  It has format as follows:
                                        (iid1, rating1, time1)
                all_uid_neigh_info:    It has format as follows:
                                        [(uid1, sim1, rating_record1), ...]
                average_uid:           Average rating of current uid.
            """
            iid = pair[0]
            real_rating = pair[1]
            average_uid = user_bd.value[uid][0]
            sim_rating = []
            for info in all_uid_neigh_info:
                sim_rating += [(iid[0], info[1], i[1] - user_bd.value[info[0]][0]) for i in info[2] if iid in i[0]]
            if len(sim_rating) != 0:
                sim_rating = [(line[0], line[1] * line[2], abs(line[1]))for line in sim_rating]
                predicted_rating = average_uid + sum(map(lambda line: line[1], sim_rating)) / sum(map(lambda line: line[2], sim_rating))
            else:
                predicted_rating = average_uid
            return iid, real_rating, self.modif_rating(predicted_rating)
        all_uid_neigh_info  = [(u[0], u[1], rating_bd.value[u[0]]) for u in sim_bd.value[uid]]
        return uid, [prediction(pair, all_uid_neigh_info, user_bd, uid) for pair in pairs]

    def user_based_recommendation(self, testRDD, sim_pair_dict, user_based_dict_bd, user_info):
        """Recommendation -> user-based, i.e., normal recommendation. For privacy protection purpose, no decay rate is allowed.
        """
        return (testRDD.filter(lambda (uid, pairs): uid in sim_pair_dict.value.keys())
                        .map(lambda (uid, pairs): self.userBasedPredict(uid, pairs, user_based_dict_bd, sim_pair_dict, user_info))
                )

    def item_based_predict_withoutdecay(self, uid, pairs, rating_bd, sim_bd, item_bd):
        """Use this function to predict the rating of item for this user.
        Args:
            iid:
            pairs:             in the format of [(iid, rating, time), ...]
            rating_bd:         broadcast [uid: [(iid1, rating1, time1), ...]
            sim_bd:            broadcast [iid: [(iid1, sim1), (iid2, sim2), ...]]
            item_bd:           broadcast [iid: (average, norm, length)]
        """
        def sort_by_time(pairs):
            """For each user, sort its rating records based on its datetime.
                More specifically, if time_a > time_b, then: time_a <- x, time_b <- x + 1.
            """
            pairs = sorted(pairs, key=lambda line: line[2], reverse=False)
            order = 0
            out = []
            for i in xrange(len(pairs)):
                if i != 0 and pairs[i][2] == pairs[i - 1][2]:
                    out += [(pairs[i][0], pairs[i][1], order)]
                else:
                    order += 1
                    out += [(pairs[i][0], pairs[i][1], order)]
            return out

        def fDecay(cur, t_ui):
            return exp(- self.alpha * (cur - t_ui))

        def add_decay(pairs):
            """add decay rate to the pairs.
            Args:
                pairs:    sim * rating, sim, time
            """
            new_pairs = sort_by_time(pairs)
            current_time = max(map(lambda line: line[2], new_pairs)) + 1
            final_pairs = [(pair[0] * fDecay(current_time, pair[2]), pair[1] * fDecay(current_time, pair[2])) for pair in new_pairs]
            return sum(map(lambda line: line[0], final_pairs)) / sum(map(lambda line: line[1], final_pairs))

        def prediction(uid, pair, rating_bd, sim_bd, item_bd):
            """Use this function to do the prediction. It can either add decay rate or not, which is decided by "method".
            """
            iid = pair[0]
            if iid not in sim_bd.value.keys():
                return ()
            iid_neigh = [(i[0], i[1], rating_bd.value[i[0]]) for i in sim_bd.value[iid]]
            real_rating = pair[1]
            iid_average = item_bd.value[iid][0]
            sim_rating = []
            for info in iid_neigh:
                sim_rating += [(iid, info[1], i[1] - item_bd.value[info[0]][0], i[2]) for i in info[2] if uid in i[0]]
            if len(sim_rating) != 0:
                sim_ratings = [(line[1] * line[2], abs(line[1]), line[3])for line in sim_rating]
                predicted_rating_no_decay = iid_average + sum(map(lambda line: line[0], sim_ratings)) / sum(map(lambda line: line[1], sim_ratings))
                predicted_rating_decay = iid_average + addDecay(sim_ratings)
            else:
                predicted_rating_no_decay = iid_average
                predicted_rating_decay = iid_average
            return iid, real_rating, self.modif_rating(predicted_rating_no_decay), self.modif_rating(predicted_rating_decay)
        return uid, [prediction(uid, pair, rating_bd, sim_bd, item_bd) for pair in pairs]

    def item_based_recommendation(self, testRDD, sim_pair_dict, item_based_dict_bd, item_info):
        """item-based Recommendation. It calculate rating with decay rate as well as without decay rate.
        """
        return testRDD.map(lambda (uid, pairs): self.itemBasedPredictNoDecay(uid, pairs, item_based_dict_bd, sim_pair_dict, item_info))

    def calculate_mae(self, rdd):
        """A function used to calculate MAE
        """
        if "user" in self.method:
            result = (rdd.map(lambda (uid, pairs): (uid, [abs(pair[1] - pair[2]) for pair in pairs if pair is not ()]))
                        .map(lambda (uid, pairs): np.array([sum(pairs), len(pairs)]))
                        .reduce(lambda a, b: a + b))
            return str(result[0] / result[1])
        else:
            nodelayresult = (rdd.map(lambda (uid, pairs): (uid, [abs(pair[1] - pair[2]) for pair in pairs if pair is not ()]))
                        .map(lambda (uid, pairs): np.array([sum(pairs), len(pairs)]))
                        .reduce(lambda a, b: a + b))
            delayresult = (rdd.map(lambda (uid, pairs): (uid, [abs(pair[1] - pair[3]) for pair in pairs if pair is not ()]))
                        .map(lambda (uid, pairs): np.array([sum(pairs), len(pairs)]))
                        .reduce(lambda a, b: a + b))
            return str(nodelayresult[0] / nodelayresult[1])  + "; " + str(delayresult[0] / delayresult[1])
