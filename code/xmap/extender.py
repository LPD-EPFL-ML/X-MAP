# -*- coding: utf-8 -*-
"""A function used to extend the baseline similarity."""


class ExtendSim:
    def __init__(self, top_k):
        """
        Args:
            top_k: define the size of most similar neighbor.
        """
        self.top_k = top_k

    def convert_to_list(self, items, chunk):
        return zip(*[iter(items)]*chunk)

    def find_items(self, rdd, BB_items_bd):
        """return valid item information.
        arg:
            rdd: (iid, [(iid, sim, mutu, frac_mutu)*])*
        return:
            iid, (BB_BB, BB_NB), (NB_BB, NB_NN)
        """
        def get_knn(iter_items):
            for iid, info in iter_items:
                info.sort(key=lambda pair: abs(pair[1]), reverse=True)
                domain_label = iid[: 2]
                if iid in BB_items_bd.value:
                    BB_BB = [pair for pair in info
                             if domain_label not in pair[0]][: self.top_k]
                    BB_NB = [pair for pair in info
                             if domain_label in pair[0]][: self.top_k]
                    yield iid, (BB_BB, BB_NB), None
                else:
                    NB = [pair for pair in info
                          if pair[0] in BB_items_bd.value]
                    if len(NB) != 0:
                        NB_BB = NB[: self.top_k]
                        NB_NN = [pair for pair in info
                                 if pair[0] not in NB][: self.top_k]
                        yield iid, None, (NB_BB, NB_NN)
        return rdd.mapPartitions(get_knn)

    def sim_extend(self, BB, NB, knn_BB_bd, knn_NB_bd):
        """Extend similarity
        Args:
        Returns:
        """
        def combineBBWithOtherInSingleDomain(iter_items):
            for iid, (NB_BB, NB_NN) in iter_items:
                for info1 in NB_BB:
                    yield info1[0], [(iid, [line[0] for line in NB_NN])]
        #
        def extendBB_S(rdd):
            def helper(iter_items):
                for iid, line in iter_items:
                    targetBB = [l for l in KNNBB_bd.value[iid].keys() if "T:" in l]
                    for v in targetBB:
                        yield (v, iid), line
            return rdd.mapPartitions(helper)
        #
        def extendBB_T(rdd):
            def helper(iter_items):
                for iid, line in iter_items:
                    sourceBB = [l for l in KNNBB_bd.value[iid].keys() if "S:" in l]
                    for v in sourceBB:
                        yield (iid, v), line
            return rdd.mapPartitions(helper)
        #
        def finalExtend(joined_BB):
            def getSim(paths):
                info            = {}
                result          = []
                KNN_BB_List     = KNNBB_bd.value.keys()
                KNN_NB_List     = KNNNB_bd.value.keys()
                for path in paths:
                    pairs       = zip(path[0: len(path) - 1], path[1: len(path)])
                    tmp_info    = []
                    for pair in pairs:
                        if pair not in info.keys():
                            if pair[0] in KNN_BB_List and pair[1] in KNNBB_bd.value[pair[0]]:
                                tmp = KNNBB_bd.value[pair[0]][pair[1]]
                            elif pair[1] in KNN_BB_List and pair[0] in KNNBB_bd.value[pair[1]]:
                                tmp = KNNBB_bd.value[pair[1]][pair[0]]
                            elif pair[0] in KNN_NB_List and pair[1] in KNNNB_bd.value[pair[0]]:
                                tmp = KNNNB_bd.value[pair[0]][pair[1]]
                            elif pair[1] in KNN_NB_List and pair[0] in KNNNB_bd.value[pair[1]]:
                                tmp = KNNNB_bd.value[pair[1]][pair[0]]
                            info.update({pair: tmp})
                        tmp_info += [info[pair]]
                    sim_info    = [l[0] for l in tmp_info]
                    mutu_info   = [l[1] for l in tmp_info]
                    frac_mutu   = [l[2] for l in tmp_info]
                    result      += [((path[0], path[-1]), (float(sum([a * b for a, b in zip(sim_info, mutu_info)])) / sum(mutu_info) if sum(mutu_info) else 0.0,
                                                                reduce(lambda a, b: a * b, frac_mutu)))]
                return result
            #
            def helper(iter_items):
                for (id_pair, (BB_T, BB_S)) in iter_items:
                    s_path = [id_pair]
                    for NB, NN_list in BB_S:
                        s_path += [id_pair + (NB,)]
                        for NN in NN_list:
                            s_path += [id_pair + (NB, NN)]
                    for NB, NN_list in BB_T:
                        t_path = []
                        final_path = []
                        for p in s_path:
                            t_path += [(NB, ) + p]
                        for p in t_path:
                            for NN in NN_list:
                                final_path += [(NN, ) + p]
                    my_path = s_path + t_path + final_path
                    yield getSim(my_path)
            return joined_BB.mapPartitions(helper)
        #
        BB_other_intra          = NB.mapPartitions(combineBBWithOtherInSingleDomain).reduceByKey(lambda a, b: a + b).cache()
        BB_other_intra_in_S     = BB_other_intra.filter(lambda (iid, info): "S:" in iid)
        BB_other_intra_in_T     = BB_other_intra.filter(lambda (iid, info): "T:" in iid)
        extendedBB_S            = extendBB_S(BB_other_intra_in_S)
        extendedBB_T            = extendBB_T(BB_other_intra_in_T)
        joined_BB               = extendedBB_T.join(extendedBB_S)
        return finalExtend(joined_BB)

    def getFinalExtension(self, crossExtended):
        """If item-item pair exists several paths, then use certainty and extension similarity to get the final similarity
            Combine the information of multiple paths of pair (item1, item2) if it exists multiple paths.
        Args:
            crossExtended:          The information in the form of (((iid1, iid2), (sim, certainty)),
                                                                    ((iid1, iid2), (sim, certainty)),
                                                                    ...)
        Returns:
            combinedCrossExtension: The informatino in the form of (((iid1, iid2), (sim, certainty)), ...)
                                    For the key, there is no duplicatation.
        """
        def getSim(pairs):
            similarity  = [pair[0] for pair in pairs]
            certainty   = [pair[1] for pair in pairs]
            return float(np.array(similarity).dot(np.array(certainty))) / sum(certainty)
        #
        def combination(iter_items):
            for iid, info in iter_items:
                tmp = {}
                result = []
                for pair in info:
                    if pair[0] not in tmp.keys():
                        tmp.update({pair[0]: [pair[1:]]})
                    else:
                        tmp[pair[0]] += [pair[1:]]
                for key in tmp.keys():
                    result += [(key, getSim(tmp[key]))]
                yield iid, result
        #
        return (crossExtended.flatMap(lambda line: line)
                            .map(lambda (iids, info): (iids[0], [(iids[1], ) + info]))
                            .reduceByKey(lambda a, b: a + b)
                            .mapPartitions(combination)
                )
