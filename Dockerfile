FROM nvidia/cuda:12.1.0-base-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get upgrade -y

# * Without installing `python3-pip` there will be no pip 
#   and `pip3`, `python3.8 -m pip`, `python3 -m pip` won't work.
# * Git is needed to install ptls lib via pipenv (see Pipfile)
# * Wget is needed to install cuda
RUN apt-get install -y \
    python3.8 \
    python3-venv \
    python3-pip \
    git \
    wget 

# Curl is needed for cownloading original data (bin/get-data.sh)
RUN apt-get install -y curl


# `pip3 install pipenv` leads to pipenv that uses python 3.6 and 
# fails to run `pipenv sync --dev`
RUN python3.8 -m pip install pipenv

WORKDIR /app
# ADD https://github.com/dllllb/ptls-experiments .
COPY . .





# Install packages exactly as specified in Pipfile.lock
RUN pipenv sync --dev

# Expose ports
# 6006 - Tensorflow
# 8082 - Luigi
# 4041 - Spark UI
EXPOSE 6006 8082 4041





# Spark installation is not needed. It's done during pipenv sync above. However
# java needs to be installed
RUN apt-get install -y openjdk-8-jdk

#########################################################################################
################################## Spark Installation ###################################
#########################################################################################

# # Spark dependencies
# # RUN apt-get install -y openjdk-8-jre
# RUN apt install -y openjdk-11-jdk

# # Environment variables (adjust Spark version if needed)
# # ENV SPARK_VERSION=3.4.2
# ENV SPARK_VERSION=3.3.0
# ENV SPARK_PACKAGE=spark-${SPARK_VERSION}-bin-hadoop3

# # Spark installation directory
# ENV SPARK_HOME=/usr/spark-${SPARK_VERSION}

# # Set PATH for Spark binaries
# ENV PATH $PATH:${SPARK_HOME}/bin

# # Download and install Spark (with Hadoop 3)
# RUN curl -sL --retry 3 \
# #   "https://dlcdn.apache.org/spark/spark-${SPARK_VERSION}/${SPARK_PACKAGE}.tgz" \
#   "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/${SPARK_PACKAGE}.tgz" \
#   | gunzip \
#   | tar x -C /usr/ \
#   && mv /usr/$SPARK_PACKAGE $SPARK_HOME \
#   && chown -R root:root $SPARK_HOME

# Run command
CMD pipenv shell "luigid --background; tensorboard --logdir lightning_logs/ --bind_all --port 6006 &"