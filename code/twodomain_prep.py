
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
    path_local = "/home/tlin/notebooks"
    path_para = join(path_local, "parameters.yaml")
    para = load_parameter(path_para)
    path_pickle_movie = join(path_local, "cache/two_domain/clean_data/movie")
    path_pickle_book = join(path_local, "cache/two_domain/clean_data/book")

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

    # Pickle these
    sourceRDD.saveAsPickleFile(path_pickle_book)
    targetRDD.saveAsPickleFile(path_pickle_movie)