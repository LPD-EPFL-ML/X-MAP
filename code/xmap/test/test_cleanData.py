# -*- coding: utf-8 -*-
"""test the cleanData.py."""

from os.path import join
from pyspark import SparkContext, SparkConf

from xmap.core.cleanData import CleanData
from xmap.utils.assist import clean_data_pipeline


if __name__ == '__main__':
    # define spark function.
    myconf = SparkConf().setAppName(
        "xmap recommendation: clean data components")
    sc = SparkContext(conf=myconf)

    # define parameters.
    path_root = "file:/home/tlin/notebooks/data"
    path_raw_movie = join(path_root, "raw/movie.txt")
    path_raw_book = join(path_root, "raw/movie.txt")
    path_pickle_movie = join(path_root, "cache/two_domain/clean_data/movie")
    path_pickle_book = join(path_root, "cache/two_domain/clean_data/book")

    num_atleast_rating = 5
    num_observation = 66666
    date_from = 2012
    date_to = 2013

    # A demo for the class.
    clean_source_tool = CleanData(
        num_atleast_rating, num_observation,
        date_from, date_to, domain_label="S:")
    clean_target_tool = CleanData(
        num_atleast_rating, num_observation,
        date_from, date_to, domain_label="T:")

    cleaned_movieRDD = clean_data_pipeline(
        sc, clean_source_tool, path_raw_movie)
    cleaned_bookRDD = clean_data_pipeline(
        sc, clean_target_tool, path_raw_book)

    cleaned_movieRDD.saveAsPickleFile(path_pickle_movie)
    cleaned_bookRDD.saveAsPickleFile(path_pickle_book)
