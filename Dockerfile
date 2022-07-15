ARG SPARK_VERSION="3.2.1"
FROM hseeberger/scala-sbt:11.0.14.1_1.6.2_2.12.15 AS jar_builder
# This builder just produces the jar that we'll copy into the final image later

COPY ./src /opt/rikai/src
COPY ./project /opt/rikai/project
COPY ./build.sbt /opt/rikai/build.sbt

WORKDIR /opt/rikai
RUN sbt clean publishLocal

FROM apache/spark-py:v${SPARK_VERSION} AS whl_builder
# BUild wheels for rikai and dependencies

USER root

COPY ./python /opt/rikai/python
COPY ./README.md /opt/rikai/README.md
WORKDIR /opt/rikai/python
RUN python3 setup.py bdist_wheel
RUN pip3 wheel -r /opt/rikai/python/docker-requirements.txt

FROM apache/spark-py:v${SPARK_VERSION} AS jupyter

USER root

RUN apt -y -qq update && \
    apt install -y -qq sudo curl aria2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# copy the wheels over and install all of them
RUN mkdir -p /opt/rikai/wheels
COPY --from=whl_builder /opt/rikai/python/dist/rikai-*.whl /opt/rikai/wheels/
COPY --from=whl_builder /opt/rikai/python/*.whl /opt/rikai/wheels/
RUN pip3 install --no-cache /opt/rikai/wheels/*.whl && \
    rm -rf /tmp/* /var/tmp/* /opt/rikai/wheels

# Copy the jar to the class path
COPY --from=jar_builder /root/.ivy2/local/ai.eto/rikai_2.12/*/jars/rikai_2.12.jar /opt/spark/jars/

RUN mkdir -p /opt/rikai/notebooks
COPY ./notebooks/Coco.ipynb /opt/rikai/notebooks/Coco.ipynb
COPY ./notebooks/Mojito.ipynb /opt/rikai/notebooks/Mojito.ipynb
EXPOSE 8888
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--notebook-dir=/opt/rikai/notebooks"]