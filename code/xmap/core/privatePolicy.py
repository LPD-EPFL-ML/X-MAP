# -*- coding: utf-8 -*-
"""use private policy on alterEgo profile."""


class PrivatePolicy:
    def __init__(self, k, epsilon, p):
        """Initialize the parameters
        Args:
            k:          The size of private neighborhood
            epsilon:    The level of privacy
        """
        self.k = k
        self.epsilon = epsilon / 2
        self.p = p

    def find_neighbor(self, sim_pairs):
        """For the rating pair of sim_pairs, we switch the position of id, and do a groupBy operation to obtain each item's neighbor.\n",
        Args:
            sim_pairs:    for one line, it has the format of [((iid1, iid2), [sim, RS])]
        """
        def key_on_first_item(iters):
            for id_pair, info_pair in iters:
                id1, id2 = id_pair
                yield id1, [(id2, info_pair)]
                yield id2, [(id1, info_pair)]
        return sim_pairs.mapPartitions(
            key_on_first_item).reduceByKey(lambda a, b: a + b)

    def preparePS(self, pairs, M_RS):
        """Prepare the procedure of mapping.
            This method involves several sub functions, e.g.,
                * getKSim:              get k'th sim.
                * splitToSets:          split original list to two sets.
                * getW:                 use the definition of w to get proper w.
        Args:
            pairs:              a pair shows as follows:    ((iid, sim), (iid, sim), ...)
            M_RS:               Maximal RS value among all pairs
        """
        def get_k_sim(lists):
            """Get k'th similarity
            """
            return lists[self.k - 1][1][0] if len(lists) >= self.k else lists[-1][1][0]

        def split_to_sets(pairs, k_sim, w):
            """split original list to two sets based on its similarity
            """
            C1 = [pair for pair in pairs if pair[1][0] >= k_sim - w]
            C0 = [pair for pair in pairs if pair not in C1]
            return [C1, C0]

        def get_w(pairs, M_RS, k_sim):
            """Calculate w.
                Be careful that there exists a condition where len(pairs) < k
            """
            v = len(pairs)
            return min(kSim, (2 * self.k * M_RS / self.epsilon) * log(self.k * (v - self.k) / self.p, e)) if v > self.k else k_sim

        k_sim = get_k_sim(pairs)
        w = get_w(pairs, M_RS, k_sim)
        splites = split_to_sets(pairs, k_sim, w)
        modified_pairs = [(pair[0], [max(pair[1][0], pair[1][0] - w), pair[1][1]]) for pair in pairs]
        return splites, modified_pairs

    def get_private_neighbor(self, pairs):
        """Use this function to get current items' priavte neighbor user. (private version)
        This method contains several sub functions to help us achieve the goal:
            * getMaxRS:                     Get max RS (global). Needed by mathematics method
            * decideNumberOfSelection:      decide exact number to select.
            * calProb:                      A function used to calculate probability. (not normalized)
            * normProb:                     Normalized probability
        """
        def decide_num_of_selection(pairs):
            """We want to choose k non-duplicate item from pairs. But what if k < # of non-zero in pairs.
            This function is used for this purpose -> avoid error in the function of np.random.choice
            Args:
                k:     number of neighbor that we want to select.
                pairs: normalized probabilities pairs
            """
            nnz = np.count_nonzero(map(lambda info_pair: info_pair[1], pairs))
            return self.k if nnz >= self.k else nnz

        def cal_prob(pairs, RS_list):
            """A function used to calculate probability (not normalized)
            """
            return [(pair[0][0], 1.0 * exp(self.epsilon * pair[0][1][0] / (2 * self.k * pair[1][1]))) for pair in zip(pairs, RS_list)]

        def norm_prob(prob_pairs):
            """A function used to normalize probability.
            """
            sum_prob = sum([pair[1] for pair in prob_pairs])
            normalized_prob_pairs = [(pair[0], pair[1] / sum_prob) for pair in prob_pairs]
            return normalized_prob_pairs

        def get_max_RS(RS_list):
            """Get max RS (local). Needed by mathematics method
            """
            return sorted(RS_list, key=lambda x: - abs(x[1]))[0][1]

        def weighted_pick(weights,n_picks):
            """weighted random selection
                returns: n_picks random indexes.
                the chance to pick the index i
                is give by the weight weights[i].
            """
            if type(weights) is not list:
                return 0
            t = np.cumsum(weights)
            s = np.sum(weights)
            st = list(set(np.searchsorted(t, rand(n_picks) * s)))
            left = n_picks - len(st)
            new = []
            if left != 0:
                new += weighted_pick(weights, left)
            return st + new
        #
        pairs = sorted(pairs, key=lambda x: - abs(x[1][0]))
        RS_list = [(pair[0], pair[1][1]) for pair in pairs]
        splites, modif_pairs = self.preparePS(pairs, getMaxRS(RS_list))
        prob_pairs = calProb(modif_pairs, RS_list)
        normalized_prob_pairs = normProb(prob_pairs)
        index_choice = weighted_pick(
            map(lambda (iid, prob): prob, normalized_prob_pairs),
            decide_num_of_selection(normalized_prob_pairs))
        return map(lambda ind: pairs[ind], index_choice)

    def private_neighbor_selection(self, rdd):
        """a function to private select neighbor.
        Args:
            rdd:                   (((id1, id2), (sim, RS)), ...)
        """
        return self.findNeighbor(rdd).map(lambda (my_id, info_pair): (my_id, self.get_private_neighbor(info_pair)))

    def get_nonprivate_neighbor(self, pairs):
        """Use this function to get current items' priavte neighbor user.
        """
        pairs = sorted(pairs, key=lambda x: - abs(x[1][0]))
        return pairs[: self.k]

    def noPrivateNeighborSelection(self, rdd):
        """a function to no-private select neighbor.
        """
        return self.findNeighbor(
            rdd).map(
            lambda (my_id, info_pair): (
                my_id, self.get_nonprivate_neighbor(info_pair)))

    def noise_perturbation(self, rdd):
        """In the perturbation step, we add independent Laplace noise to the neighbor selected in previous step.
        Args:
            rdd:      (id, [(id1, [sim1, RS1]), (id2, [sim2, RS2]), ...])
        """
        def cal_laplace(info):
            """Use this function to calculate independent Laplace noise.
            Args:
                info:    (sim, RS)
            """
            return np.random.laplace(0, abs(info[1]) / self.epsilon)
        return rdd.map(lambda (my_id, pairs): (my_id, map(lambda (neigh_id, info): (neigh_id, info[0] + cal_laplace(info)), pairs)))

    def nonnoise_perturbation(self, rdd):
        """It is no-noise perturbation step, we simply remove RS.
        Args:
            rdd:      (id, [(id1, [sim1, RS1]), (id2, [sim2, RS2]), ...])
        """
        return rdd.map(lambda (my_id, pairs): (my_id, map(lambda (neigh_id, info): (neigh_id, info[0]), pairs)))
