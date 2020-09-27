# -*- coding: utf-8 -*-
"""test crossSim.py."""

from os.path import join

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

from xmap.utils.assist import recommender_calculate_sim_pipeline
from xmap.core.recommenderSim import RecommenderSim


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
    path_pickle_private_alterEgo = join(
        path_root, "cache/two_domain/generator/private_alterEgo")
    path_pickle_nonprivate_alterEgo = join(
        path_root, "cache/two_domain/generator/nonprivate_alterEgo")
    path_pickle_userbased_alterEgo = join(
        path_root, "cache/two_domain/cross_sim/user_based_alterEgo")
    path_pickle_itembased_alterEgo = join(
        path_root, "cache/two_domain/cross_sim/item_based_alterEgo")
    path_pickle_alterEgo_userbased_sim = join(
        path_root, "cache/two_domain/cross_sim/targetdomain_userbased_sim")
    path_pickle_alterEgo_itembased_sim = join(
        path_root, "cache/two_domain/cross_sim/targetdomain_itembased_sim")

    # load data.
    trainRDD = sc.pickleFile(path_pickle_train).cache()
    private_alterEgo = sc.pickleFile(path_pickle_private_alterEgo)
    nonprivate_alterEgo = sc.pickleFile(path_pickle_nonprivate_alterEgo)

    # for user based.
    # init class
    cross_sim = RecommenderSim(method='cosine_user', num_atleast=50)

    # start
    user_based_alterEgo, item_based_alterEgo, \
        user_based_dict_bd, item_based_dict_bd, \
        user_info_bd, item_info_bd, \
        alterEgo_sim = recommender_calculate_sim_pipeline(
            sc, cross_sim, private_alterEgo)

    user_based_alterEgo.saveAsPickleFile(path_pickle_userbased_alterEgo)
    item_based_alterEgo.saveAsPickleFile(path_pickle_itembased_alterEgo)
    alterEgo_sim.saveAsPickleFile(path_pickle_alterEgo_userbased_sim)

    # for item based.
    # init class
    cross_sim = RecommenderSim(method='cosine_item', num_atleast=50)

    # start
    alterEgo_sim = cross_sim.calculate_sim(
        item_based_alterEgo, user_based_alterEgo,
        item_info_bd, user_info_bd)
    alterEgo_sim.saveAsPickleFile(path_pickle_alterEgo_itembased_sim)
