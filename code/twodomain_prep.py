# -*- coding: utf-8 -*-
"""call different components and do two domain recommendation.
twodomain_prep runs all of the components except for the reccomender, 
such that all of the RDDs involved can be cached. This makes testing much 
easier
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

    path_raw_movie = join(
        para['init']['path_hdfs'], para['init']['path_movie'])
    path_raw_book = join(
        para['init']['path_hdfs'], para['init']['path_book'])

    # baseliner
    baseliner_cleansource_tool = BaselinerClean(
        para['baseliner']['num_atleast_rating'],
        para['baseliner']['size_subset'],
        para['baseliner']['date_from'],
        para['baseliner']['date_to'], domain_label="S:")
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

    sourceRDD = baseliner_clean_data_pipeline(
        sc, baseliner_cleansource_tool,
        path_raw_book,
        para['init']['is_debug'], para['init']['num_partition'])
    targetRDD = baseliner_clean_data_pipeline(
        sc, baseliner_cleantarget_tool,
        path_raw_movie,
        para['init']['is_debug'], para['init']['num_partition'])

    trainRDD, testRDD = baseliner_split_data_pipeline(
        sc, baseliner_splitdata_tool, sourceRDD, targetRDD)

    item2item_simRDD = baseliner_calculate_sim_pipeline(
        sc, baseliner_calculate_sim_tool, trainRDD)

    # extender
    extendsim_tool = ExtendSim(para['extender']['extend_among_topk'])
    extendedsimRDD = extender_pipeline(
        sc, sqlContext, baseliner_calculate_sim_tool,
        extendsim_tool, item2item_simRDD)

    # generator
    generator_ tool = Generator(
        para['generator']['mapping_range'],
        para['generator']['private_epsilon'],
        para['baseliner']['calculate_baseline_sim_method'],
        para['generator']['private_rpo'])

    alterEgo_profile = generator_pipeline(
        generator_tool,
        trainRDD, extendedsimRDD, para['generator']['private_flag'])

    ### At this point the prep is done ###
    # Time to serialize all the things
    path_pickle_train = join(para['init']['path_hdfs'], "cache/traindata")
    path_pickle_test  = join(para['init']['path_hdfs'], "cache/testdata")
    path_pickle_alterego  = join(para['init']['path_hdfs'], "cache/alterEgo")

    testRDD.saveAsPickleFile(path_pickle_test)
    alterEgo_profile.saveAsPickleFile(path_pickle_alterego)