# -*- coding: utf-8 -*-
"""An auxilary file for xmap."""


def clean_data_pipeline(clean_tool, dataRDD):
    """a pipeline to clean the data."""
    parsedRDD = clean_tool.parse_data(dataRDD)
    filteredRDD = clean_tool.filter_data(parsedRDD)
    cleanedRDD = clean_tool.clean_data(filteredRDD).cache()
    partialRDD = clean_tool.take_partial_data(cleanedRDD)
    return partialRDD
