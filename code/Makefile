build_package:
	python setup.py install

run_demo:
	spark-submit --master local[16] \
	    --py-files dist/xmap-0.1.0-py3.5.egg twodomain_demo.py

upload_server:
	rsync -av -e ssh dist/xmap-0.1.0-py3.5.egg tlin@access.grid5000.fr:rennes/code

build_computation:
	sh .build_computation.sh

build_container:
	docker build --tag xmap:1.5 platform/

run_container:
	docker run -v .:/home/tlin/notebooks:z docker.io/itamtao/spark
