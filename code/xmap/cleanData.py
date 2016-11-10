# -*- coding: utf-8 -*-
"""A class to clean the dataset."""
import re
from datetime import datetime

from pyspark import SparkContext, SparkConf


class CleanData:
    def __init__(
            self, num_atleast_rating, num_observation,
            date_from, date_to, domain_label):
        """initialize  parameter.
        Args:
            num_atleast_rating:
                For each user, it rated at least 'num_atleast_rating' items.
            num_observation:
                The number of observation we want to choose.
            date_from:
                The starting date of ratings.
            date_to:
                The end date of ratings.
            domain_label:
                Add a domain label to the original dataset.
        """
        self.num_atleast_rating = num_atleast_rating
        self.num_observation = num_observation
        self.period = range(date_from, date_to + 1)
        self.label = domain_label

    def parse_time(self, s):
        """convert unix timestamp to readable date.
        Args:
            s: unix timestamp (string)
        Returns:
            datetime
        """
        return datetime.fromtimestamp(float(s))

    def parse_line(self, iterators):
        """parse a line.
        Args:
            iterators: data in lines with format: 'uid, iid, rating, datetime'.
        Returns:
            (uid, [domain_label + iid, rating, datetime])
        """
        for line in iterators:
            splitted_line = re.split('\\s+', line)
            parsed_time = self.parse_time(splitted_line[3])
            if parsed_time.year in self.period:
                yield (
                    splitted_line[0], (
                        self.label + splitted_line[1],
                        float(splitted_line[2]), parsed_time))

    def parse_data(self, originalRDD):
        """parse the dataset."""
        return originalRDD.mapPartitions(self.parse_line)

    def take_partial_data(self, dataRDD):
        """take partial data to do the computation."""
        return dataRDD.take(self.num_observation)

    def remove_invalid(self, iterators):
        """remove invalid rating, e.g., old or duplicate.
        iterators is a set of line in the form of [uid, (iid, rating, time)*].
        """
        def check_invalid(line, tmpdict):
            """check data and update invalid data.
            e.g., we only use latest data.
            """
            iid, rating, time = line
            if time > tmpdict[iid][2]:
                tmpdict[iid] = line
            return tmpdict

        for uid, ratings in iterators:
            tmpdict = {}
            for token in ratings:
                # token: in the form of (iid, rating, time)
                if token[0] in tmpdict.keys():
                    tmpdict = check_invalid(token, tmpdict)
                else:
                    tmpdict[token[0]] = token
            yield uid, list(tmpdict.values())

    def filter_data(self, dataRDD):
        """filter the invalid data.
        e.g., 1. duplicate data; 2. same uid, iid, but with different rating.
        """
        seq_op = (lambda a, b: a + [b])
        comb_op = (lambda a, b: a + b)
        groupedRDD = dataRDD.aggregateByKey([], seq_op, comb_op)
        return groupedRDD.mapPartitions(self.remove_invalid)

    def clean_data(self, filteredRDD):
        """remove users who have low rating frequency."""
        return filteredRDD.filter(
            lambda line: len(line[1]) >= self.num_atleast_rating)


if __name__ == '__main__':
    # define spark function.
    myconf = SparkConf().setAppName(
        "xmap recommendation: clean data components")
    sc = SparkContext(conf=myconf)

    # define parameters.
    path_raw_movie_data = "../data/raw/movie.txt"
    path_raw_book_data = "../data/raw/movie.txt"

    num_atleast_rating = 5
    num_observation = 10000
    date_from = 2012
    date_to = 2013

    # A demo for the class.
    movieRDD = sc.textFile(path_raw_movie_data, 30)
    bookRDD = sc.textFile(path_raw_book_data, 30)

    clean_source_tool = CleanData(
        num_atleast_rating, num_observation,
        date_from, date_to, domain_label="S")
    clean_target_tool = CleanData(
        num_atleast_rating, num_observation,
        date_from, date_to, domain_label="T")
