# -*- coding: utf-8 -*-
"""test the splitData.py."""

from os.path import join

from pyspark import SparkContext, SparkConf

from xmap.core.baselinerSplit import BaselinerSplit
from xmap.utils.assist import baseliner_split_data_pipeline


if __name__ == '__main__':
    # define spark function.
    myconf = SparkConf().setAppName(
        "xmap recommendation: split data components")
    sc = SparkContext(conf=myconf)

    # Refactor: use the actual paramaters file
    path_local = "/opt/spark_apps/code"
    path_para = join(path_local, "parameters.yaml")
    para = load_parameter(path_para)

    # define parameters.
    path_pickle_movie = join(para['init']['path_hdfs'], "cache/two_domain/clean_data/movie")
    path_pickle_book = join(para['init']['path_hdfs'], "cache/two_domain/clean_data/book")
    path_pickle_train = join(para['init']['path_hdfs'], "cache/two_domain/split_data/train")
    path_pickle_test = join(para['init']['path_hdfs'], "cache/two_domain/split_data/test")

    num_left = para['baseliner']['num_left']
    ratio_split = para['baseliner']['ratio_split']
    ratio_both = para['baseliner']['ratio_both']
    seed = para['init']['seed']

    # A demo for the class.
    movieRDD = sc.pickleFile(path_pickle_movie)
    bookRDD = sc.pickleFile(path_pickle_book)
    sourceRDD, targetRDD = movieRDD, bookRDD

    split_data = BaselinerSplit(num_left, ratio_split, ratio_both, seed)

    trainRDD, testRDD = baseliner_split_data_pipeline(
        sc, split_data, sourceRDD, targetRDD)

    trainRDD.saveAsPickleFile(path_pickle_train)
    testRDD.saveAsPickleFile(path_pickle_test)
