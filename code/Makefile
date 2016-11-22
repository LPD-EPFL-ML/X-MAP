build_package:
	python setup.py install

run_demo:
	spark-submit --master local[4] \
	    --py-files dist/xmap-0.1.0-py3.5.egg twodomain_demo.py
