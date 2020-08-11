build_package:
	python setup.py install

run_demo:
	spark-submit --master local[16] \
	    --py-files dist/xmap-0.1.0-py3.8.egg twodomain_demo.py

upload_server:
	rsync -av -e ssh dist/xmap-0.1.0-py3.5.egg tlin@access.grid5000.fr:rennes/code

build_computation:
	sh .build_computation.sh

build_container:
	docker build --tag xmap:2.0 platform/

run_container:
	docker run --ulimit nofile=1048576:1048576 -v .:/home/tlin/notebooks xmap:2.0