# X-MAP: Heterogeneous Recommendation at Large-Scale

X-MAP is a large-scale heterogeneous recommender which is built on top of Apache Spark and implemented in Python.

## Features
- Provides heterogeneous recommendation based on artificial **AlterEgos** of users across multiple application domains.
- Any recommendation algorithm can be run on using these AlterEgos.
- Provides formal privacy guarantees.


## Getting Started

### Prerequisites
X-MAP requires `Python 3`, `Numpy 1.10.4`, `Apache Spark 1.6.1` pre-installed on your machine.

#### Installation
Please refer to
[Anaconda](https://www.continuum.io/downloads), [Apache Spark](http://spark.apache.org/) for installation instructions.

#### Docker installation
We also provide a docker image for your convenience.

If you do not have `docker` and `docker-compose` installed on your machine, please check the [official configuration guidance](https://www.docker.com/products/overview) for installation guidelines.
Otherwise, open your console, go to the `platform` folder, and execute the below-mentioned command to setup the corresponding docker image.

```
docker-compose build
```

## Build
Once you modify the scripts in `X-MAP` folder, you should rebuild the package using the following command:

```
python setup.py install
```

You will find an egg file, located in `dist/xmap-0.1.0-py3.5.egg`, that you will use for your application.

## Running the tests
### Prerequisites
X-MAP is tested on real-traces of [Amazon Dataset](https://snap.stanford.edu/data/web-Amazon.html). For current implementation, the input data follows the following format:
`<userid>\t<itemid>\t<rating>\t<timestamp>`.

Note that the timestamp is required if you want to incorporate temporal behaviour which is also provided by the AlterEgos.

### Run X-MAP
We provide here two demonstrations: `twodomain_demo.py` and `multidomain_demo.py`. You can also tune the parameters in the file `parameters.yaml`.

Note that the scipt should run successfully using the docker image that we provided. Please check your local system settings (e.g., directory path) while working with the application.

#### Run X-MAP locally on a machine with 4 cores using docker
A simple example of how to run X-MAP on a local machine.

```
spark-submit --master local[4] \
    --py-files dist/xmap-0.1.0-py3.5.egg twodomain_demo.py
```

#### Run X-MAP on a distributed setup
A simple example of how to run X-MAP on a Cluster.

```
spark-submit 	--py-files xmap-0.1.0-py3.5.egg \
				--num-executors 30 --executor-cores 3 --executor-memory 12g \
				--driver-memory 12g --driver-cores 4 twodomain_demo.py
```

## Advanced Usage
### Use publicly available library with AlterEgos
X-MAP can be easily used with any publicly available library. We provide an example below for using Spark's built-in MLlib library with X-MAP.

```
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from xmap.core import *

# use component in xmap to build alterEgo profile.
sourceRDD = baseliner_clean_data_pipeline(...)
targetRDD = baseliner_clean_data_pipeline(...)
trainRDD, testRDD = baseliner_split_data_pipeline(...)
item2item_simRDD = baseliner_calculate_sim_pipeline(...)
extendedsimRDD = extender_pipeline(...)
alterEgo_profileRDD = generator_pipeline(...)

# build MLlib to do matrix factorization

als = ALS(...)
model = als.fit(alterEgo_profileRDD)
predictions = model.transform(testRDD)
evaluator = RegressionEvaluator(...)
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))
```
## Support
Please raise potential bugs on [github] (https://github.com/LPD-EPFL-ML/X-MAP/issues). If you have an open-ended or a research related question, you can post it on: [X-MAP group] (https://groups.google.com/forum/#!forum/x-map).
