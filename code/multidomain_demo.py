# -*- coding: utf-8 -*-
"""call different components and do multiple domain recommendation."""


from os.path import join
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

from xmap.core.baselinerClean import BaselinerClean
from xmap.core.baselinerSplit import BaselinerSplit
from xmap.core.baselinerSim import BaselinerSim
from xmap.core.extender import ExtendSim
from xmap.core.generator import Generator
from xmap.core.recommenderSim import RecommenderSim
from xmap.core.recommenderPrivacy import RecommenderPrivacy
from xmap.core.recommenderPrediction import RecommenderPrediction

from xmap.utils.assist import load_parameter
from xmap.utils.assist import baseliner_clean_data_pipeline
from xmap.utils.assist import baseliner_split_multidomain_data_pipeline
from xmap.utils.assist import baseliner_calculate_sim_pipeline
from xmap.utils.assist import extender_pipeline
from xmap.utils.assist import generator_pipeline
from xmap.utils.assist import recommender_calculate_sim_pipeline
from xmap.utils.assist import recommender_privacy_pipeline
from xmap.utils.assist import recommender_prediction_pipeline


if __name__ == '__main__':
    # define spark function.
    myconf = SparkConf().setAppName(
        "xmap recommendation: xmap (multi domain)").set(
        "spark.broadcast.factory",
        "org.apache.spark.broadcast.TorrentBroadcastFactory").set(
        "spark.io.compression.codec", "snappy").set(
        "spark.broadcast.compress", "true").set(
        "spark.akka.frameSize", "1024").set(
        "spark.driver.maxResultSize", "16g").set(
        "spark.python.worker.memory", "16g").set(
        "spark.shuffle.compress", "true").set(
        "spark.rdd.compress", "true")
    sc = SparkContext(conf=myconf)
    sqlContext = SQLContext(sc)

    # define parameters.
    path_local = "/home/tlin/notebooks"
    path_para = join(path_local, "parameters.yaml")
    para = load_parameter(path_para)

    path_raw_movie = join(
        para['init']['path_hdfs'], para['init']['path_movie'])
    path_raw_book = join(
        para['init']['path_hdfs'], para['init']['path_book'])
    path_raw_music = join(
        para['init']['path_hdfs'], para['init']['path_music'])

    # baseliner
    baseliner_cleansource_1_tool = BaselinerClean(
        para['baseliner']['num_atleast_rating'],
        para['baseliner']['size_subset'],
        para['baseliner']['date_from'],
        para['baseliner']['date_to'], domain_label="S:1:")
    baseliner_cleansource_2_tool = BaselinerClean(
        para['baseliner']['num_atleast_rating'],
        para['baseliner']['size_subset'],
        para['baseliner']['date_from'],
        para['baseliner']['date_to'], domain_label="S:2:")
    baseliner_cleantarget_tool = BaselinerClean(
        para['baseliner']['num_atleast_rating'],
        para['baseliner']['size_subset'],
        para['baseliner']['date_from'],
        para['baseliner']['date_to'], domain_label="T:")
    baseliner_splitdata_tool = BaselinerSplit(
        para['baseliner']['num_left'],
        para['baseliner']['ratio_split'],
        para['baseliner']['ratio_both'],
        para['init']['seed'])
    baseliner_calculate_sim_tool = BaselinerSim(
        para['baseliner']['calculate_baseline_sim_method'],
        para['baseliner']['calculate_baseline_weighting'])

    sourceRDD1 = baseliner_clean_data_pipeline(
        sc, baseliner_cleansource_1_tool,
        path_raw_movie,
        para['init']['is_debug'], para['init']['num_partition'])

    sourceRDD2 = baseliner_clean_data_pipeline(
        sc, baseliner_cleansource_2_tool,
        path_raw_movie,
        para['init']['is_debug'], para['init']['num_partition'])

    targetRDD = baseliner_clean_data_pipeline(
        sc, baseliner_cleantarget_tool,
        path_raw_book,
        para['init']['is_debug'], para['init']['num_partition'])

    trainRDD1, trainRDD2, testRDD \
        = baseliner_split_multidomain_data_pipeline(
            sc, baseliner_splitdata_tool, sourceRDD1, sourceRDD2, targetRDD)

    item2item_simRDD1 = baseliner_calculate_sim_pipeline(
        sc, baseliner_calculate_sim_tool, trainRDD1)
    item2item_simRDD2 = baseliner_calculate_sim_pipeline(
        sc, baseliner_calculate_sim_tool, trainRDD2)

    # extender
    extendsim_tool = ExtendSim(para['extender']['extend_among_topk'])
    extendedsimRDD1 = extender_pipeline(
        sc, sqlContext, baseliner_calculate_sim_tool,
        extendsim_tool, item2item_simRDD1)
    extendedsimRDD2 = extender_pipeline(
        sc, sqlContext, baseliner_calculate_sim_tool,
        extendsim_tool, item2item_simRDD2)

    # generator
    generator_tool = Generator(
        para['generator']['mapping_range'],
        para['generator']['private_epsilon'],
        para['baseliner']['calculate_baseline_sim_method'],
        para['generator']['private_rpo'])

    alterEgo_profile1 = generator_pipeline(
        generator_tool,
        trainRDD1, extendedsimRDD1, para['generator']['private_flag'])
    alterEgo_profile2 = generator_pipeline(
        generator_tool,
        trainRDD2, extendedsimRDD2, para['generator']['private_flag'])
    alterEgo_profile3 = alterEgo_profile1.union(alterEgo_profile2)

    # recommender
    recommender_sim_tool = RecommenderSim(
        para['recommender']['calculate_xmap_sim_method'],
        para['recommender']['calculate_xmap_weighting'])
    recommender_privacy_tool = RecommenderPrivacy(
        para['recommender']['mapping_range'],
        para['recommender']['private_epsilon'],
        para['recommender']['private_rpo'])
    recommender_prediction_tool = RecommenderPrediction(
        para['recommender']['decay_alpha'],
        para['recommender']['calculate_xmap_sim_method'])

    _, _, user_based_dict_bd1, item_based_dict_bd1, \
        user_info_bd1, item_info_bd1, \
        alterEgo_sim1 = recommender_calculate_sim_pipeline(
            sc, recommender_sim_tool, alterEgo_profile1)
    _, _, user_based_dict_bd2, item_based_dict_bd2, \
        user_info_bd2, item_info_bd2, \
        alterEgo_sim2 = recommender_calculate_sim_pipeline(
            sc, recommender_sim_tool, alterEgo_profile2)
    _, _, user_based_dict_bd3, item_based_dict_bd3, \
        user_info_bd3, item_info_bd3, \
        alterEgo_sim3 = recommender_calculate_sim_pipeline(
            sc, recommender_sim_tool, alterEgo_profile3)

    private_preserve_simpair1 = recommender_privacy_pipeline(
        recommender_privacy_tool, alterEgo_sim1,
        para['recommender']['private_flag'])
    private_preserve_simpair2 = recommender_privacy_pipeline(
        recommender_privacy_tool, alterEgo_sim2,
        para['recommender']['private_flag'])
    private_preserve_simpair3 = recommender_privacy_pipeline(
        recommender_privacy_tool, alterEgo_sim3,
        para['recommender']['private_flag'])

    simpair_dict_bd1 = sc.broadcast(private_preserve_simpair1.collectAsMap())
    simpair_dict_bd2 = sc.broadcast(private_preserve_simpair2.collectAsMap())
    simpair_dict_bd3 = sc.broadcast(private_preserve_simpair3.collectAsMap())

    mae1 = recommender_prediction_pipeline(
        recommender_prediction_tool, recommender_sim_tool,
        testRDD, simpair_dict_bd1,
        user_based_dict_bd1, item_based_dict_bd1,
        user_info_bd1, item_info_bd1)

    mae2 = recommender_prediction_pipeline(
        recommender_prediction_tool, recommender_sim_tool,
        testRDD, simpair_dict_bd1,
        user_based_dict_bd2, item_based_dict_bd2,
        user_info_bd2, item_info_bd2)

    mae3 = recommender_prediction_pipeline(
        recommender_prediction_tool, recommender_sim_tool,
        testRDD, simpair_dict_bd3,
        user_based_dict_bd3, item_based_dict_bd3,
        user_info_bd3, item_info_bd3)

    print(
        'rmse alterEgo from source domain 1: {}' +
        'rmse alterEgo from source domain 2: {}' +
        'rmse alterEgo from source domain 1 + 2: {}'.format(mae1, mae2, mae3))
    sc.stop()
