# -*- coding: utf-8 -*-
"""A function used to extend the baseline similarity."""
from functools import reduce

import numpy as np


class ExtendSim:
    def __init__(self, top_k):
        """
        Args:
            top_k: define the size of most similar neighbor.
        """
        self.top_k = top_k

    def find_knn_items(self, rdd, BB_items_bd):
        """return valid item information.
        arg:
            rdd: (iid, [(iid, sim, mutu, frac_mutu)*])*
        return:
            in the form of `iid, (BB_BB, BB_NB), (NB_BB, NB_NN).
            BB_*/NB_* in the form of `iid, sim, mutu, frac_mutu`.
            use BB_NB as an example where BB define the type of iid,
            while NB define the type that iid connect to.
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

    def sim_extend(self, BB_info, NB_info, knn_BB_bd, knn_NB_bd):
        """Extend similarity."""
        def combine_BB_withother_in_singledomain(iter_items):
            """combine BB item with other items for each domain.
            return:
                NB_NN iid, (BB iid, [NB_NN iid]*)
            """
            for iid, (NB_BB, NB_NN) in iter_items:
                """
                NB_BB: [(BB iid, sim, mutu, frac_mutu)*]
                NB_NN: [(NN iid, sim, mutu, frac_mutu)*]
                """
                for info in NB_BB:
                    yield info[0], [(iid, [line[0] for line in NB_NN])]

        def extend_BB_source(sourceRDD):
            """connect BB item in target domain with items in source domain.
            (BB_target, BB_source), connections
            """
            def helper(iter_items):
                for iid, line in iter_items:
                    for v in knn_BB_bd.value[iid].keys():
                        if "T:" in v:
                            yield (v, iid), line
            return sourceRDD.mapPartitions(helper)

        def extend_BB_target(rdd):
            """connect BB item in source domain with item in target domain.
            (BB_target, BB_source), connections
            """
            def helper(iter_items):
                for iid, line in iter_items:
                    for v in knn_BB_bd.value[iid].keys():
                        if "S:" in v:
                            yield (iid, v), line
            return rdd.mapPartitions(helper)

        def calculate_path_confidence(sim_info, mutu_info, frac_mutu):
            """calculate the confidence of the path."""
            denominator = sum([a * b for a, b in zip(sim_info, mutu_info)])
            numerator = sum(mutu_info)
            s_p = 1.0 * denominator / numerator if numerator else 0.0
            c_p = reduce(lambda a, b: a * b, frac_mutu)
            return s_p, c_p

        def get_final_sim(paths):
            final_score = []
            local_db = {}
            knn_BB_iids = knn_BB_bd.value.keys()
            knn_NB_iids = knn_NB_bd.value.keys()
            for path in paths:
                iid_pairs = zip(path[0: len(path) - 1], path[1: len(path)])
                tmp_info = []
                for iid1, iid2 in iid_pairs:
                    if (iid1, iid2) not in local_db.keys():
                        if iid1 in knn_BB_iids \
                                and iid2 in knn_BB_bd.value[iid1]:
                            tmp = knn_BB_bd.value[iid1][iid2]
                        elif iid2 in knn_BB_iids \
                                and iid1 in knn_BB_bd.value[iid2]:
                            tmp = knn_BB_bd.value[iid2][iid1]
                        elif iid1 in knn_NB_iids \
                                and iid2 in knn_NB_bd.value[iid1]:
                            tmp = knn_NB_bd.value[iid1][iid2]
                        elif iid2 in knn_NB_iids \
                                and iid1 in knn_NB_bd.value[iid2]:
                            tmp = knn_NB_bd.value[iid2][iid1]
                        local_db.update({(iid1, iid2): tmp})
                    tmp_info += [local_db[(iid1, iid2)]]
                sim_info = [l[0] for l in tmp_info]
                mutu_info = [l[1] for l in tmp_info]
                frac_mutu = [l[2] for l in tmp_info]
                final_score.append(
                    ((path[0], path[-1]),
                     calculate_path_confidence(sim_info, mutu_info, frac_mutu))
                    )
            return final_score

        def final_nonjoint_extend(nonjoint_BB):
            """extend path for items that only linked to BB_target.
            arg:
                nonjoint_BB: (target_iid, source_iid), source_info
            """
            def helper(iter_items):
                for iid_pair, source in iter_items:
                    """iid_pair in the form of (target_iid, source_iid).
                    source_path: from BB_target to item in source domain.
                    """
                    source_path = [iid_pair]
                    for NB_iid, NN_iids in source:
                        source_path += [iid_pair + (NB_iid,)]
                        for NN_iid in NN_iids:
                            source_path += [iid_pair + (NB_iid, NN_iid)]
                    yield get_final_sim(source_path)
            return nonjoint_BB.mapPartitions(helper)

        def final_joint_extend(joined_BB):
            """extend path for items that have additional items in each domain.
            arg:
                joined_BB: (target_iid, source_iid), (source_info, target_info)
            """
            def helper(iter_items):
                for iid_pair, (source, target) in iter_items:
                    """iid_pair in the form of (target_iid, source_iid).
                    source_path: from BB_target to item in source domain.
                    target_path: from NB_target to item in source domain.
                    longest_path: from NN_target to item in source domain.
                    """
                    source_path = [iid_pair]
                    for NB_iid, NN_iids in source:
                        source_path += [iid_pair + (NB_iid,)]
                        for NN_iid in NN_iids:
                            source_path += [iid_pair + (NB_iid, NN_iid)]

                    for NB_iid, NN_iids in target:
                        target_path = []
                        longest_path = []
                        for p in source_path:
                            target_path += [(NB_iid, ) + p]
                        for p in target_path:
                            for NN_iid in NN_iids:
                                longest_path += [(NN_iid, ) + p]
                        yield get_final_sim(target_path + longest_path)
            return joined_BB.mapPartitions(helper)

        BB_other_intra = NB_info.mapPartitions(
            combine_BB_withother_in_singledomain).reduceByKey(
            lambda a, b: a + b).cache()
        BB_other_intra_source = BB_other_intra.filter(lambda l: "S:" in l[0])
        BB_other_intra_target = BB_other_intra.filter(lambda l: "T:" in l[0])
        extended_BB_source = extend_BB_source(BB_other_intra_source)
        extended_BB_target = extend_BB_target(BB_other_intra_target)
        joined_extended_BB = extended_BB_source.join(extended_BB_target)
        final_joint_extended = final_joint_extend(joined_extended_BB)
        final_nonjoint_extended = final_nonjoint_extend(extended_BB_source)

        return final_joint_extended.union(final_nonjoint_extended)

    def get_final_extension(self, cross_extended):
        """Deal with the case of multiple path among an item-item pair.
            If item-item pair exists several paths,
            then use s_p and c_p to get the final similarity
        Args:
            cross_extended: in the form of [((iid1, iid2), (s_p, c_p))*]
        Returns:
            xsim: in the form of (iid1, [(iid2, sim)*])
        """
        def swap_info(line):
            """adjust the position of the information."""
            iids, info = line
            return iids[0], [(iids[1], ) + info]

        def get_sim(pairs):
            similarity = np.array([pair[0] for pair in pairs])
            certainty = np.array([pair[1] for pair in pairs])
            return 1.0 * similarity.dot(certainty) / np.sum(certainty)

        def merge(iter_items):
            for iid, info in iter_items:
                local_db = dict()
                final_sim = []
                for pair in info:
                    if pair[0] not in local_db.keys():
                        local_db.update({pair[0]: [pair[1:]]})
                    else:
                        local_db[pair[0]] += [pair[1:]]
                for key in local_db.keys():
                    final_sim.append((key, get_sim(local_db[key])))
                yield iid, final_sim

        return cross_extended.flatMap(lambda x: x).map(swap_info).reduceByKey(
            lambda a, b: a + b).mapPartitions(merge)
