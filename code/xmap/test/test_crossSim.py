# -*- coding: utf-8 -*-
"""test crossSim.py."""

from os.path import join

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

from xmap.utils.assist import crosssim_pipeline
from xmap.core.crossSim import CrossSim


if __name__ == '__main__':
    # define spark function.
    myconf = SparkConf().setAppName(
        "xmap recommendation: crossSim components")
    sc = SparkContext(conf=myconf)
    sqlContext = SQLContext(sc)

    # define parameters.
    path_root = "file:/home/tlin/notebooks/data"
    path_pickle_train = join(path_root, "cache/two_domain/split_data/train")
    path_pickle_test = join(path_root, "cache/two_domain/split_data/test")
    path_pickle_baseline_sim = join(
        path_root, "cache/two_domain/baseline_sim/basesim")
    path_pickle_extended_sim = join(
        path_root, "cache/two_domain/extend_sim/extendsim")
    path_pickle_private_mapped_sim = join(
        path_root, "cache/two_domain/private_mapping/privatemap")
    path_pickle_nonprivate_mapped_sim = join(
        path_root, "cache/two_domain/private_mapping/nonprivatemap")
    path_pickle_userbased_alterEgo = join(
        path_root, "cache/two_domain/cross_sim/user_based_alterEgo")
    path_pickle_itembased_alterEgo = join(
        path_root, "cache/two_domain/cross_sim/item_based_alterEgo")
    path_pickle_alterEgo_userbased_sim = join(
        path_root, "cache/two_domain/cross_sim/targetdomain_userbased_sim")
    path_pickle_alterEgo_itembased_sim = join(
        path_root, "cache/two_domain/cross_sim/targetdomain_itembased_sim")

    # load data.
    training_dataRDD = sc.pickleFile(path_pickle_train).cache()
    private_mapped_sim = sc.pickleFile(path_pickle_private_mapped_sim)

    # for user based.
    # init class
    cross_sim = CrossSim(method='cosine_user', num_atleast=50)

    # start
    user_based_alterEgo, item_based_alterEgo, \
        user_based_dict_bd, item_based_dict_bd, \
        user_info_bd, item_info_bd, \
        targetdomain_sim = crosssim_pipeline(
            sc, cross_sim, training_dataRDD, private_mapped_sim)

    user_based_alterEgo.saveAsPickleFile(path_pickle_userbased_alterEgo)
    item_based_alterEgo.saveAsPickleFile(path_pickle_itembased_alterEgo)
    targetdomain_sim.saveAsPickleFile(path_pickle_alterEgo_userbased_sim)

    # for item based.
    # init class
    cross_sim = CrossSim(method='cosine_item', num_atleast=50)

    # start
    targetdomain_sim = cross_sim.calculate_sim(
        item_based_alterEgo, user_based_alterEgo,
        item_info_bd, user_info_bd)
    targetdomain_sim.saveAsPickleFile(path_pickle_alterEgo_itembased_sim)
