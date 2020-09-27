# -*- coding: utf-8 -*-
"""test the cleanData.py."""

from os.path import join
from pyspark import SparkContext, SparkConf

from xmap.core.baselinerClean import BaselinerClean
from xmap.utils.assist import baseliner_clean_data_pipeline


if __name__ == '__main__':
    # define spark function.
    myconf = SparkConf().setAppName(
        "xmap recommendation: clean data components")
    sc = SparkContext(conf=myconf)

    # Refactor: use the actual paramaters file
    path_local = "/opt/spark_apps/code"
    path_para = join(path_local, "parameters.yaml")
    para = load_parameter(path_para)

    path_raw_movie = join(para['init']['path_hdfs'], para['init']['path_movie'])
    path_raw_book = join(para['init']['path_hdfs'], para['init']['path_book'])

    path_pickle_movie = join(para['init']['path_hdfs'], "cache/two_domain/clean_data/movie")
    path_pickle_book = join(para['init']['path_hdfs'], "cache/two_domain/clean_data/book")

    num_atleast_rating = 5
    num_observation = 66666
    date_from = 2012
    date_to = 2013

    # A demo for the class.
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

    cleaned_movieRDD = baseliner_clean_data_pipeline(
        sc, clean_source_tool, path_raw_movie)
    cleaned_bookRDD = baseliner_clean_data_pipeline(
        sc, clean_target_tool, path_raw_book)

    cleaned_movieRDD.saveAsPickleFile(path_pickle_movie)
    cleaned_bookRDD.saveAsPickleFile(path_pickle_book)
