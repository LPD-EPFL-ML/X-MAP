# -*- coding: utf-8 -*-
"""test the extender.py."""
from os.path import join

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

from xmap.core.baselinerSim import BaselinerSim
from xmap.core.extender import ExtendSim
from xmap.utils.assist import extender_pipeline


if __name__ == '__main__':
    # define spark function.
    myconf = SparkConf().setAppName(
        "xmap recommendation: extender sim components")
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

    # A demo for the class.
    itemsim = BaselinerSim(
        para['baseliner']['calculate_baseline_sim_method'],
        para['baseliner']['calculate_baseline_weighting'])
    extendsim = ExtendSim(top_k=10)

    testRDD = sc.pickleFile(path_pickle_test)
    item2item_simRDD = sc.pickleFile(path_pickle_baseline_sim)

    final_extended_sim = extender_pipeline(
            sc, sqlContext, itemsim, extendsim, item2item_simRDD)

    final_extended_sim.saveAsPickleFile(path_pickle_extended_sim)
