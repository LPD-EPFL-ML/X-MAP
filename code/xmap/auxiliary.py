# -*- coding: utf-8 -*-
"""An auxilary file for xmap."""


def clean_data_pipeline(sc, clean_tool, path_rawdata):
    """a pipeline to clean the data."""
    dataRDD = sc.textFile(path_rawdata, 30)
    parsedRDD = clean_tool.parse_data(dataRDD)
    filteredRDD = clean_tool.filter_data(parsedRDD)
    cleanedRDD = clean_tool.clean_data(filteredRDD).cache()
    partial_data = clean_tool.take_partial_data(cleanedRDD)
    partialRDD = sc.parallelize(partial_data, 30).cache()
    return partialRDD
