# -*- coding: utf-8 -*-
"""A class to split the data."""
import random
from os.path import join

from pyspark import SparkContext, SparkConf

from auxiliary import split_data_pipeline


class SplitData:
    def __init__(self, num_left, ratio_split, ratio_both, seed):
        """ Initialize parameter.
        Args:
            num_left:
                For each overlap user, if we choose hide current user's rating
                in the target domain, then this value will be used to determine
                whether we should hide all information (cold-start)
                or hide partial information (sparsity).
            ratio_split:
                It will split the original overlap dataset to two parts
                (ratio_split, 1 - ratio_split).
                To be noticed that:
                  1. non-overlap data: do nothing.
                  2. overlap data:
                    (1 Training: hide target domain's rating based on num_left.
                    (2 Test: use this part for evaluation.
            ratio_both:
                its value \in [0.2, 1 - ratio_split].
                overlap users will keep their ratings in each domain.
        """
        self.num_left = num_left
        self.ratio_split = ratio_split
        self.ratio_both = ratio_both
        self.seed = seed
        random.seed(seed)

    def find_overlap_user(self, sourceRDD, targetRDD):
        """Find overlap users.
        return a RDD of uid.
        """
        return sourceRDD.keys().intersection(targetRDD.keys())

    def find_overlap_user_multidomain(self, sourceRDD1, sourceRDD2, targetRDD):
        """Find overlap users.
        return a RDD of uid.
        """
        return sourceRDD1.keys().intersection(
            sourceRDD2.keys()).intersection(targetRDD.keys())

    def distinguish_data(self, overlap_userRDD_bd, sourceRDD, targetRDD):
        """Distinguish data from overlap data to non-overlap data."""
        def in_split(iterators):
            for line in iterators:
                if line[0] in overlap_userRDD_bd.value:
                    yield line

        def out_split(iterators):
            for line in iterators:
                if line[0] not in overlap_userRDD_bd.value:
                    yield line

        overlap_sourceRDD = sourceRDD.mapPartitions(in_split)
        non_overlap_sourceRDD = sourceRDD.mapPartitions(out_split)
        overlap_targetRDD = targetRDD.mapPartitions(in_split)
        non_overlap_targetRDD = targetRDD.mapPartitions(out_split)

        return non_overlap_sourceRDD, overlap_sourceRDD, \
            non_overlap_targetRDD, overlap_targetRDD

    def determine_remaining(self, iterators):
        """split test dataset into (source, remain, remove).
        Args:
            iterators: (uid, lines), where line=[(iid, rating, time)*]
        Returns:
            uid, source data, remaining data(target), removed data(target)
            The remaining data can be empty.
        """
        def decide_remaining(mydict):
            remaining_dict = random.sample(mydict, self.num_left)
            test_dict = [
                item for item in mydict if item not in remaining_dict]
            return [remaining_dict, test_dict]

        for uid, lines in iterators:
            source = [line for line in lines if "S:" in line[0]]
            target = filter(lambda line: line not in source, lines)
            remain, remove = decide_remaining(target)
            yield uid, source, remain, remove

    def split_data(
            self, non_overlap_sourceRDD, overlap_sourceRDD,
            non_overlap_targetRDD, overlap_targetRDD):
        """split data to several parts.
        Returns:
            trainingDataRDD: (uid, (dict(source), dict(target)))
            testDataRDD: Same as trainingDataRDD
        """
        unionRDD = overlap_sourceRDD.union(
            overlap_targetRDD).reduceByKey(lambda a, b: a + b)
        splits = unionRDD.randomSplit(
            [self.ratio_split, self.ratio_both,
             1 - self.ratio_split - self.ratio_both],
            seed=self.seed)

        splits[0].cache()
        test_part_dataRDD = splits[0].mapPartitions(
            self.determine_remaining).cache()
        test_dataRDD = test_part_dataRDD.map(lambda line: (line[0], line[3]))
        training_dataRDD = non_overlap_sourceRDD.union(
            non_overlap_targetRDD).union(
            splits[1]).union(splits[2]).union(
            test_part_dataRDD.map(lambda line: (line[0], line[1] + line[2]))
            )
        return training_dataRDD, test_dataRDD

    def split_data_multipledomain(
            self, nonoverlap_source1RDD, overlap_source1RDD,
            nonoverlap_source2RDD, overlap_source2RDD,
            non_overlap_targetRDD, overlap_targetRDD):
        """split data based on the data from multiple domain."""
        unionRDD = overlap_source1RDD.union(
            overlap_source2RDD).union(
            overlap_targetRDD).reduceByKey(lambda a, b: a + b)
        splits = unionRDD.randomSplit(
            [self.ratio_split, self.ratio_both,
             1 - self.ratio_split - self.ratio_both],
            seed=self.seed)

        splits[0].cache()
        test_part_dataRDD = splits[0].mapPartitions(
            self.determine_remaining).cache()
        test_dataRDD = test_part_dataRDD.map(lambda line: (line[0], line[3]))
        training_overlapRDD = test_part_dataRDD.map(
                lambda line: (line[0], line[1] + line[2])).union(
            splits[1]).union(splits[2])

        training_overlap_source1RDD = training_overlapRDD.map(
            lambda uid, lines: (
                uid, [line for line in lines
                      if ("S:1:" in line[0] or "T:" in line[0])]))
        training_overlap_source2RDD = training_overlapRDD.map(
            lambda uid, lines: (
                uid, [line for line in lines
                      if ("S:2:" in line[0] or "T:" in line[0])]))

        training_source1RDD = nonoverlap_source1RDD.union(
            training_overlap_source1RDD).union(
            non_overlap_targetRDD)

        training_source2RDD = nonoverlap_source2RDD.union(
            training_overlap_source2RDD).union(
            non_overlap_targetRDD)
        return training_source1RDD, training_source2RDD, test_dataRDD

if __name__ == '__main__':
    # define spark function.
    myconf = SparkConf().setAppName(
        "xmap recommendation: split data components")
    sc = SparkContext(conf=myconf)

    # define parameters.
    path_root = "file:/home/tlin/notebooks/data"
    path_pickle_movie = join(path_root, "cache/two_domain/clean_data/movie")
    path_pickle_book = join(path_root, "cache/two_domain/clean_data/book")
    path_pickle_train = join(path_root, "cache/two_domain/split_data/train")
    path_pickle_test = join(path_root, "cache/two_domain/split_data/test")

    num_left = 0
    ratio_split = 0.2
    ratio_both = 0.8
    seed = 666

    # A demo for the class.
    movieRDD = sc.pickleFile(path_pickle_movie)
    bookRDD = sc.pickleFile(path_pickle_book)
    sourceRDD, targetRDD = movieRDD, bookRDD

    split_data = SplitData(num_left, ratio_split, ratio_both, seed)

    training_dataRDD, test_dataRDD = split_data_pipeline(
        sc, split_data, sourceRDD, targetRDD)

    training_dataRDD.saveAsPickleFile(path_pickle_train)
    test_dataRDD.saveAsPickleFile(path_pickle_test)
