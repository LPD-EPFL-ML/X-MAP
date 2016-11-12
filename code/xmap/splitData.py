# -*- coding: utf-8 -*-
"""A class to split the data."""
import random
from pyspark import SparkContext, SparkConf


class SplitData:
    def __init__(self, num_left, ratio_split, ratio_both, seed):
        """ Initialize parameter.
        Args:
            num_left:
                For the overlap user, if current user choose to hide his rating
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

    def findOverLapUser(self, sourceRDD, targetRDD):
        """Find overlap users
        Args:
            sourceRDD: RDD of data from source domain
            targetRDD: RDD of data from target domain
        Returns:
            overlapUserRDD: RDD of overlap user in these two domains
        """
        return sourceRDD.keys().intersection(targetRDD.keys())

    def recognizeData(self, overlapUserRDD_bd, sourceRDD, targetRDD):
        """Distinguish data from overlap data to non-overlap data.
        Args:
            overlapUserRDD: RDD of overlap user
            sourceRDD: RDD of data from source domain
            targetRDD: RDD of data from target domain
        Returns:
            nonOverlapSourceRDD: RDD of non-overlap user in source domain
            overlapSourceRDD: RDD of overlap user in source domain
            nonOverlapTargetRDD: RDD of non-overlap user in target domain
            overlapTargetRDD: RDD of overlap user in target domain
        """
        def inSplit(iterators):
            for line in iterators:
                if line[0] in overlapUserRDD_bd.value:
                    yield line
        def outSplit(iterators):
            for line in iterators:
                if line[0] not in overlapUserRDD_bd.value:
                    yield line
        overlapSourceRDD    = sourceRDD.mapPartitions(inSplit)
        nonOverlapSourceRDD = sourceRDD.mapPartitions(outSplit)
        overlapTargetRDD    = targetRDD.mapPartitions(inSplit)
        nonOverlapTargetRDD = targetRDD.mapPartitions(outSplit)
        return (nonOverlapSourceRDD, overlapSourceRDD, nonOverlapTargetRDD, overlapTargetRDD)

    def splitData(self, nonOverlapSourceRDD, overlapSourceRDD, nonOverlapTargetRDD, overlapTargetRDD):
        """In this function, we will split data to several part
        Args:
            nonOverlapSourceRDD: RDD of non-overlap user in source domain
            overlapSourceRDD: RDD of overlap user in source domain
            nonOverlapTargetRDD: RDD of non-overlap user in target domain
            overlapTargetRDD: RDD of overlap user in target domain
        Returns:
            trainingDataRDD: Its structure is "(key, list(dict(source), dict(target)))"
            testDataRDD: Same as trainingDataRDD
        """
        def determineRemain(iters):
            """Use this function, we can have split test dataset into following parts.
            Args:
                uid, lines in iterators
            Returns:
                uid, source data, remained data (target), removed data (target)
                More precisely, remained data can be empty.
            """
            def decideRemain(mydict):
                remainDict      = random.sample(mydict, self.sizeOfRemaining)
                testDict        = [item for item in mydict if item not in remainDict]
                return [remainDict, testDict]
            for uid, lines in iters:
                source          = [line for line in lines if "S:" in line[0]]
                target          = filter(lambda line: line not in source, lines)
                remain, remove  = decideRemain(target)
                yield uid, source, remain, remove
        #
        splits = (overlapSourceRDD.union(overlapTargetRDD)
                                  .reduceByKey(lambda a, b: a + b)
                                  .randomSplit([self.ratioOfSplit, self.ratioOfBoth, 1 - self.ratioOfSplit - self.ratioOfBoth], seed = self.splitSeed)
                                )
        splits[0].cache()

        #
        testPartsDataRDD        = splits[0].mapPartitions(determineRemain).cache()
        testDataRDD             = testPartsDataRDD.map(lambda line: (line[0], line[3]))
        trainingDataRDD         = (nonOverlapSourceRDD.union(nonOverlapTargetRDD)
                                                    .union(splits[1])
                                                    .union(splits[2])
                                                    .union(testPartsDataRDD.map(lambda line: (line[0], line[1] + line[2])))
                                    )
        return (trainingDataRDD, testDataRDD)
