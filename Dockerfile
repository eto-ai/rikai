FROM apache/spark-py:v3.1.3 AS builder

USER root
ARG RIKAI_VERSION
ENV SPARK_VERSION="3.2.1"
RUN apt -y -qq update && apt install -y -qq sudo curl aria2 && yes | pip install rikai[jupyter,pytorch]==${RIKAI_VERSION}

RUN mkdir -p /opt/rikai/notebooks
COPY ./notebooks/Coco.ipynb /opt/rikai/notebooks/Coco.ipynb
COPY ./notebooks/Mojito.ipynb /opt/rikai/notebooks/Mojito.ipynb
EXPOSE 8888
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--notebook-dir=/opt/rikai/notebooks"]