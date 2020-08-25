FROM jupyter/pyspark-notebook:749017d878d6

MAINTAINER Tao Lin <itamtao@gmail.com>

USER root

# create and switch user
RUN useradd --create-home --shell /bin/zsh tlin
# RUN yum -y install git numpy scipy makefile

ENV PATH="/usr/local/spark-1.6.1-bin-hadoop2.6/bin/:${PATH}"

WORKDIR /home/tlin/notebooks
