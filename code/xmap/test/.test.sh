CODE=test_crossSim.py

spark-submit --master local[4] \
    --py-files ../../dist/xmap-0.1.0-py3.5.egg ${CODE}
