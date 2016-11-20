# -*- coding: utf-8 -*-
"""test the privateMapping.py."""

from os.path import join

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

from xmap.core.privateMapping import PrivateMapping
from xmap.utils.assist import private_mapping_pipeline


if __name__ == '__main__':
    # define spark function.
    myconf = SparkConf().setAppName(
        "xmap recommendation: private mapping components")
    sc = SparkContext(conf=myconf)
    sqlContext = SQLContext(sc)

    # define parameters.
    path_root = "file:/home/tlin/notebooks/data"
    path_pickle_train = join(path_root, "cache/two_domain/split_data/train")
    path_pickle_test = join(path_root, "cache/two_domain/split_data/test")
    path_pickle_baseline_sim = join(
        path_root, "cache/two_domain/baseline_sim/basesim")
    path_pickle_extended_sim = join(
        path_root, "cache/two_domain/extend_sim/extendsim")
    path_pickle_private_mapped_sim = join(
        path_root, "cache/two_domain/private_mapping/privatemap")
    path_pickle_nonprivate_mapped_sim = join(
        path_root, "cache/two_domain/private_mapping/nonprivatemap")

    # A demo for the class.
    private_map = PrivateMapping(
        mapping_range=1, privacy_epsilon=0.6, sim_method='ad_cos', rpo=0.1)

    extended_simRDD = sc.pickleFile(path_pickle_extended_sim)

    private_mappedRDD, _ = private_mapping_pipeline(
        private_map, extended_simRDD, private=True)

    nonprivate_mappedRDD, _ = private_mapping_pipeline(
        private_map, extended_simRDD, private=False)

    private_mappedRDD.saveAsPickleFile(path_pickle_private_mapped_sim)
    nonprivate_mappedRDD.saveAsPickleFile(path_pickle_nonprivate_mapped_sim)
