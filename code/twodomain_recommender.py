# -*- coding: utf-8 -*-
"""
call different components and do two domain recommendation.

This is the full twodomain demo, for experimentations see twodomain_prep and twodomain_reccomender
"""

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

from xmap.utils.assist import write_to_disk
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
        "spark.rdd.compress", "true")
    sc = SparkContext(conf=myconf)
    sqlContext = SQLContext(sc)

    # define parameters.
    path_local = "/opt/spark_apps/code"
    path_para = join(path_local, "parameters.yaml")
    para = load_parameter(path_para)
    
    # Locations of cached RDDs
    path_pickle_train = join(para['init']['path_hdfs'], "cache/traindata")
    path_pickle_test  = join(para['init']['path_hdfs'], "cache/testdata")
    path_pickle_alterego  = join(para['init']['path_hdfs'], "cache/alterEgo")
    
    path_raw_movie = join(
        para['init']['path_hdfs'], para['init']['path_movie'])
    path_raw_book = join(
        para['init']['path_hdfs'], para['init']['path_book'])


    ### Assume the twodomain_prep has been ran ###

    # load data
    testRDD = sc.pickleFile(path_pickle_test).cache()
    alterEgo_profile = sc.pickleFile(path_pickle_alterego).cache()

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

    _, _, user_based_dict_bd, item_based_dict_bd, \
        user_info_bd, item_info_bd, \
        alterEgo_sim = recommender_calculate_sim_pipeline(
            sc, recommender_sim_tool, alterEgo_profile)

    private_preserve_simpair = recommender_privacy_pipeline(
        recommender_privacy_tool, alterEgo_sim,
        para['recommender']['private_flag'])

    simpair_dict_bd = sc.broadcast(private_preserve_simpair.collectAsMap())

    mae = recommender_prediction_pipeline(
        recommender_prediction_tool, recommender_sim_tool,
        testRDD, simpair_dict_bd,
        user_based_dict_bd, item_based_dict_bd,
        user_info_bd, item_info_bd)

    results = {
        "mae": mae
    }

    write_to_disk(results, para, join(path_local, "data", "output"))
    sc.stop()
