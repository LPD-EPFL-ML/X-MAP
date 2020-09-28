# -*- coding: utf-8 -*-
"""test the privateMapping.py."""

from os.path import join

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

from xmap.core.generator import Generator
from xmap.utils.assist import generator_pipeline


if __name__ == '__main__':
    # define spark function.
    myconf = SparkConf().setAppName(
        "xmap recommendation: generator components")
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
    path_pickle_private_alterEgo = join(
        path_root, "cache/two_domain/generator/private_alterEgo")
    path_pickle_nonprivate_alterEgo = join(
        path_root, "cache/two_domain/generator/nonprivate_alterEgo")

    # A demo for the class.
    generator_tool = Generator(
        mapping_range=1, privacy_epsilon=0.6, sim_method='ad_cos', rpo=0.1)

    trainRDD = sc.pickleFile(path_pickle_train).cache()
    extended_simRDD = sc.pickleFile(path_pickle_extended_sim)

    private_alterEgo_profile = generator_pipeline(
        generator_tool, trainRDD, extended_simRDD, private=True)

    nonprivate_alterEgo_profile = generator_pipeline(
        generator_tool, extended_simRDD, private=False)

    private_alterEgo_profile.saveAsPickleFile(
        path_pickle_private_alterEgo)
    nonprivate_alterEgo_profile.saveAsPickleFile(
        path_pickle_nonprivate_alterEgo)
