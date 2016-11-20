# -*- coding: utf-8 -*-
"""test the baselineSim.py."""

from os.path import join

from pyspark import SparkContext, SparkConf

from xmap.core.baselineSim import BaselineSim
from xmap.utils.assist import itembasedsim_pipeline


if __name__ == '__main__':
    # define spark function.
    myconf = SparkConf().setAppName(
        "xmap recommendation: baseline sim components").set(
        "spark.driver.maxResultSize", "1g").set(
        "spark.python.worker.memory", "2g")

    sc = SparkContext(conf=myconf)

    # define parameters.
    path_root = "file:/home/tlin/notebooks/data"
    path_pickle_train = join(path_root, "cache/two_domain/split_data/train")
    path_pickle_test = join(path_root, "cache/two_domain/split_data/test")
    path_pickle_baseline_sim = join(
        path_root, "cache/two_domain/baseline_sim/basesim")

    # A demo for the class.
    trainRDD = sc.pickleFile(path_pickle_train)
    testRDD = sc.pickleFile(path_pickle_test)

    itemsim = BaselineSim(method='ad_cos', num_atleast=50)

    item2item_simRDD = itembasedsim_pipeline(sc, itemsim, trainRDD)

    item2item_simRDD.saveAsPickleFile(path_pickle_baseline_sim)
