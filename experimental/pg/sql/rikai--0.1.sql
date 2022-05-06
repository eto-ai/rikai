-- Create Schema to contain all Rikai functionality
CREATE SCHEMA IF NOT EXISTS ml;

-- Semantic Types
CREATE TYPE image AS (uri TEXT);

CREATE TYPE detection AS (label TEXT, box box, score real);


-- Tables for ML metadata
CREATE TABLE ml.models (
	name VARCHAR(128) PRIMARY KEY,
	flavor VARCHAR(128),
	model_type VARCHAR(128),
	options JSONB
);
CREATE INDEX IF NOT EXISTS model_flavor_idx
ON ml.models (flavor, model_type);

-- Functions
CREATE FUNCTION ml.version()
RETURNS TEXT
AS $$
	import rikai
	return rikai.__version__.version
$$ LANGUAGE plpython3u;

CREATE FUNCTION ml.create_model_trigger()
RETURNS TRIGGER
AS $$
	plpy.info("Creating model: ", TD)
	model_name = TD["new"]["name"]
	flavor = TD["new"]["flavor"]
	stmt = (
		"CREATE FUNCTION ml.{}(img image) ".format(model_name) +
		"RETURNS detection[] " +
		"AS $BODY$ " +
		"	return None " +
		"$BODY$ LANGUAGE plpython3u;"
	)
	plpy.execute(stmt)
	return None
$$ LANGUAGE plpython3u;

CREATE TRIGGER create_model
AFTER INSERT ON ml.models
FOR EACH ROW
EXECUTE FUNCTION ml.create_model_trigger();
