# -*- coding: utf-8 -*-
"""test recommender.py."""


from os.path import join

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

from xmap.core.recommenderSim import RecommenderSim
from xmap.core.recommenderPrediction import RecommenderPrediction
from xmap.utils.assist import write_txt
from xmap.utils.assist import recommender_prediction_pipeline


if __name__ == '__main__':
    # define spark function.
    myconf = SparkConf().setAppName(
        "xmap recommendation: crossSim components")
    sc = SparkContext(conf=myconf)
    sqlContext = SQLContext(sc)

    # Refactor: use the actual paramaters file
    path_local = "/opt/spark_apps/code"
    path_para = join(path_local, "parameters.yaml")
    para = load_parameter(path_para)

    # define parameters.
    path_root = para['init']['path_hdfs']
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
    path_pickle_private_policy_userbased_sim = join(
        path_root, "cache/two_domain/policy/policy_userbased_sim")
    path_pickle_private_policy_itembased_sim = join(
        path_root, "cache/two_domain/policy/policy_itembased_sim")

    path_txt_userbased_mae = join(
        path_root, "cache/two_domain/recommendation/userbased_mae.txt")
    path_txt_itembased_mae = join(
        path_root, "cache/two_domain/recommendation/itembased_mae.txt")

    # load data.
    trainRDD = sc.pickleFile(path_pickle_train).cache()
    testRDD = sc.pickleFile(path_pickle_test).cache()

    alterEgo_userbased_sim = sc.pickleFile(
        path_pickle_alterEgo_userbased_sim).cache()
    alterEgo_itembased_sim = sc.pickleFile(
        path_pickle_alterEgo_itembased_sim).cache()

    userbased_sim_pair = sc.pickleFile(
        path_pickle_private_policy_userbased_sim).cache()
    itembased_sim_pair = sc.pickleFile(
        path_pickle_private_policy_itembased_sim).cache()

    item_based_alterEgo = sc.pickleFile(
        path_pickle_itembased_alterEgo).cache()
    user_based_alterEgo = sc.pickleFile(
        path_pickle_userbased_alterEgo).cache()

    # init class
    recommender_userbased = RecommenderPrediction(
        alpha=0.1, method='cosine_user')
    recommender_itembased = RecommenderPrediction(
        alpha=0.1, method='cosine_item')

    cross_sim = RecommenderSim(method='cosine_user', num_atleast=50)

    # build broadcast.
    user_info = cross_sim.get_info(user_based_alterEgo)
    item_info = cross_sim.get_info(item_based_alterEgo)

    user_info_bd = sc.broadcast(user_info.collectAsMap())
    item_info_bd = sc.broadcast(item_info.collectAsMap())

    user_based_dict_bd = sc.broadcast(user_based_alterEgo.collectAsMap())
    item_based_dict_bd = sc.broadcast(item_based_alterEgo.collectAsMap())

    userbased_simpair_dict_bd = sc.broadcast(
        userbased_sim_pair.collectAsMap())
    itembased_simpair_dict_bd = sc.broadcast(
        itembased_sim_pair.collectAsMap())

    # user based model.
    mae = recommender_prediction_pipeline(
            recommender_userbased, cross_sim, testRDD,
            userbased_simpair_dict_bd,
            user_based_dict_bd, item_based_dict_bd,
            user_info_bd, item_info_bd)
    write_txt(mae, path_txt_userbased_mae)

    # item based model.
    mae = recommender_prediction_pipeline(
            recommender_itembased, cross_sim, testRDD,
            itembased_simpair_dict_bd,
            user_based_dict_bd, item_based_dict_bd,
            user_info_bd, item_info_bd)
    write_txt(mae, path_txt_itembased_mae)
