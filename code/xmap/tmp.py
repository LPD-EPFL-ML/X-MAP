myconf = SparkConf().setAppName(
    "xmap recommendation: clean data components").set(
    "spark.broadcast.factory",
    "org.apache.spark.broadcast.TorrentBroadcastFactory").set(
    "spark.io.compression.codec", "snappy").set(
    "spark.broadcast.compress", "true").set(
    "spark.akka.frameSize", "1024").set(
    "spark.driver.maxResultSize", "16g").set(
    "spark.python.worker.memory", "16g").set(
    "spark.shuffle.compress", "true").set(
    "spark.rdd.compress", "true").set(
    "spark.executor.cores", "8")
