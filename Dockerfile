ARG SPARK_VERSION="3.2.1"

FROM hseeberger/scala-sbt:11.0.14.1_1.6.2_2.12.15 AS jar_builder

COPY ./src /opt/rikai/src
COPY ./project /opt/rikai/project
COPY ./build.sbt /opt/rikai/build.sbt

WORKDIR /opt/rikai
RUN sbt clean compile package && cp $(ls ./target/scala-2.12/rikai_2.12-*.jar | sort | tail -n 1) /opt/rikai/


FROM apache/spark-py:v${SPARK_VERSION} AS jupyter

USER root

RUN apt -y -qq update && apt install -y -qq sudo curl aria2

#RUN mkdir -p /opt/rikai/wheels
#COPY --from=whl_builder /opt/rikai/dist/rikai-*.whl /opt/rikai/wheels/
#RUN pip3 install /opt/rikai/wheels/rikai-*.whl
COPY ./python /opt/rikai/python
COPY ./README.md /opt/rikai/README.md
WORKDIR /opt/rikai/python
RUN pip3 install -e ".[jupyter,pytorch]"

COPY --from=jar_builder /opt/rikai/rikai_2.12-*.jar /opt/spark/jars/

RUN mkdir -p /opt/rikai/notebooks
COPY ./notebooks/Coco.ipynb /opt/rikai/notebooks/Coco.ipynb
COPY ./notebooks/Mojito.ipynb /opt/rikai/notebooks/Mojito.ipynb
EXPOSE 8888
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--notebook-dir=/opt/rikai/notebooks"]