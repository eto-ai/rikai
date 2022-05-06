EXTENSION = rikai
DATA = sql/rikai--0.1.sql

PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)


docker:
	docker build -t rikai-pg .
.PHONY: docker

run: docker
	docker run -v ${PWD}/tools/init.sql:/docker-entrypoint-initdb.d/0.sql \
		-e POSTGRES_HOST_AUTH_METHOD=trust \
		-p 5432:5432 rikai-pg
.PHONY: run

