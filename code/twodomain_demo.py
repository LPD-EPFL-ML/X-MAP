# -*- coding: utf-8 -*-
"""call different components and do two domain recommendation."""

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
from xmap.utils.assist import baseliner_split_data_pipeline
from xmap.utils.assist import baseliner_calculate_sim_pipeline
from xmap.utils.assist import extender_pipeline
from xmap.utils.assist import generator_pipeline
from xmap.utils.assist import recommender_calculate_sim_pipeline
from xmap.utils.assist import recommender_privacy_pipeline
from xmap.utils.assist import recommender_prediction_pipeline


if __name__ == '__main__':
    # define spark function.
    myconf = SparkConf().setAppName(
        "xmap recommendation: xmap (two domain)").set(
        "spark.broadcast.factory",
        "org.apache.spark.broadcast.TorrentBroadcastFactory").set(
        "spark.io.compression.codec", "snappy").set(
        "spark.broadcast.compress", "true").set(
        "spark.akka.frameSize", "1024").set(
        "spark.driver.maxResultSize", "16g").set(
        "spark.python.worker.memory", "16g").set(
        "spark.shuffle.compress", "true").set(
        "spark.rdd.compress", "true").set(
        "spark.executor.cores", "8")
    sc = SparkContext(conf=myconf)
    sqlContext = SQLContext(sc)

    # # define parameters.
    path_local = "/home/tlin/notebooks"
    path_para = join(path_local, "parameters.yaml")
    para = load_parameter(path_para)

    path_raw_movie = join(
        para['init']['path_hdfs'], para['init']['path_movie'])
    path_raw_book = join(
        para['init']['path_hdfs'], para['init']['path_book'])

    # baseliner
    clean_source_tool = BaselinerClean(
        para['baseliner']['num_atleast_rating'],
        para['baseliner']['size_subset'],
        para['baseliner']['date_from'],
        para['baseliner']['date_to'], domain_label="S:")
    clean_target_tool = BaselinerClean(
        para['baseliner']['num_atleast_rating'],
        para['baseliner']['size_subset'],
        para['baseliner']['date_from'],
        para['baseliner']['date_to'], domain_label="T:")
    split_data_tool = BaselinerSplit(
        para['baseliner']['num_left'],
        para['baseliner']['ratio_split'],
        para['baseliner']['ratio_both'],
        para['init']['seed'])
    itemsim_tool = BaselinerSim(
        para['baseliner']['calculate_baseline_sim_method'],
        para['baseliner']['calculate_baseline_weighting'])

    sourceRDD = baseliner_clean_data_pipeline(
        sc, clean_source_tool, path_raw_movie, para['init']['is_debug'])
    targetRDD = baseliner_clean_data_pipeline(
        sc, clean_target_tool, path_raw_book, para['init']['is_debug'])

    trainRDD, testRDD = baseliner_split_data_pipeline(
        sc, split_data_tool, sourceRDD, targetRDD)

    item2item_simRDD = baseliner_calculate_sim_pipeline(
        sc, itemsim_tool, trainRDD)

    # extender
    extendsim_tool = ExtendSim(para['extender']['extend_among_topk'])
    extendedsimRDD = extender_pipeline(
        sc, sqlContext, itemsim_tool, extendsim_tool, item2item_simRDD)

    # generator
    generator_tool = Generator(
        para['generator']['mapping_range'],
        para['generator']['private_epsilon'],
        para['baseliner']['calculate_baseline_sim_method'],
        para['generator']['private_rpo'])

    alterEgo_profile = generator_pipeline(
        generator_tool, extendedsimRDD,
        para['generator']['private_flag'])

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

    user_based_alterEgo, item_based_alterEgo, \
        user_based_dict_bd, item_based_dict_bd, \
        user_info_bd, item_info_bd, \
        alterEgo_sim = recommender_calculate_sim_pipeline(
            sc, recommender_sim_tool, alterEgo_profile)

    private_preserve_simpair = recommender_privacy_pipeline(
        recommender_privacy_tool, recommender_sim_tool, alterEgo_sim,
        para['recommender']['private_flag'])

    simpair_dict_bd = sc.broadcast(private_preserve_simpair.collectAsMap())

    mae = recommender_prediction_pipeline(
            recommender_prediction_tool, recommender_sim_tool,
            testRDD, simpair_dict_bd,
            user_based_dict_bd, item_based_dict_bd,
            user_info_bd, item_info_bd)