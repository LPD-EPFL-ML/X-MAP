# X-<sub>MAP</sub>

Xmap is the implementation of the paper `What You Might Like To Read After Watching Interstellar`. The system is built upon Apache Spark, and written by Python.


## Getting Started

### Prerequisites

You should have `Python 3`, `Numpy 1.10.4`, `Apache Spark 1.6.1` installed on your computer.

#### Installing
Please refer to
[Anaconda](https://www.continuum.io/downloads), [Apache Spark](http://spark.apache.org/) for installization.

#### Installing (Docker)
We also provide a docker image for your convenience.

If you do not have `docker` and `docker-compose` installed on your computer, please check the [official configuration guidance](https://www.docker.com/products/overview) for help.
Otherwise, open your console and go to the top-level folder `platform`, and type the command below to setup the corresponding docker image.

```
docker-compose build
```

## Build
Once you modify the scripts in `xmap` folder, you should rebuild the package through the following command:

```
python setup.py install
```

You will find an egg file that located in `dist/xmap-0.1.0-py3.5.egg`, and you will use this file for your application.

## Running the tests
### Prerequisites
Xmap use [Amazon Dataset](https://snap.stanford.edu/data/web-Amazon.html) for its experiment. For current implementation, the input data follows the following format:
`userid itemid rating time`, seperated by `\t`.

### Run Xmap
Here we provide two examples `twodomain_demo.py` and `multidomain_demo.py`, and you can set your parameters in the file `parameters.yaml`.

Note that the scipt should run successfully under provided docker image. Please check your local paramesters (e.g., directory path) when working on your own case.

#### Run Xmap locally on 4 cores with docker
A simple example of how to run script locally.

```
spark-submit --master local[4] \
    --py-files dist/xmap-0.1.0-py3.5.egg twodomain_demo.py
```

#### Distributed case on a Cluster
A simple example of how to run script on a Cluster.

```
spark-submit 	--py-files xmap-0.1.0-py3.5.egg \
				--num-executors 30 --executor-cores 3 --executor-memory 12g \
				--driver-memory 12g --driver-cores 4 twodomain_demo.py
```

## Advance Usage
### Introduce other methodology
Xmap can be easily extend to matrix factorization verision by using built-in MLlib in Spark. Please check a simplified example below:

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
