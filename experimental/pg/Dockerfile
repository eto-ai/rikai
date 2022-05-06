FROM postgres:14

RUN apt-get update -y \
	&& apt-get install -y -qq postgresql-plpython3-14 make python3-pip \
	&& apt-get clean \
	&& pip install rikai[pytorch]>0.1 \
	&& rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN mkdir -p /build
COPY sql /build/sql
COPY Makefile rikai.control dist/*.whl /build
RUN cd /build && make install && pip install /build/*.whl