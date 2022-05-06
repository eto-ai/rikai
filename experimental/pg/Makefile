EXTENSION = rikai
DATA = sql/rikai--0.1.sql

PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)


build:
	docker build -t rikai-pg .
PHONY: build
