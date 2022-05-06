-- Create Schema to contain all Rikai functionality
CREATE SCHEMA IF NOT EXISTS ml;

-- Semantic Types
CREATE TYPE image AS (uri TEXT);

CREATE TYPE detection AS (label TEXT, label_id int, box box, score real);


-- Tables for ML metadata
CREATE TABLE ml.models (
	name VARCHAR(128) NOT NULL PRIMARY KEY,
	flavor VARCHAR(128) NOT NULL,
	model_type VARCHAR(128) NOT NULL,
	uri VARCHAR(1024),
	options JSONB DEFAULT '{}'::json
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

-- Trigger to create a model inference function after
-- creating a model entry.
CREATE FUNCTION ml.create_model_trigger()
RETURNS TRIGGER
AS $$
    model_name = TD["new"]["name"]
    plpy.info("Creating model: ", model_name)
    flavor = TD["new"]["flavor"]
    model_type = TD["new"]["model_type"]
    uri = TD["new"].get("uri")
    if uri is not None:
        # Quoted URI
        uri = "'{}'".format(uri)
    stmt = (
		"CREATE FUNCTION ml.{}(img image) ".format(model_name) +
		"RETURNS detection[] " +
		"AS $BODY$\n" +
        "    from rikai.experimental.pg.model import load_model\n" +
        "    if 'model' not in SD:\n" +
        "        plpy.info('Loading model: flavor={} type={})')\n".format(flavor, model_type) +
        "        SD['model'] = load_model('{}', '{}', {})\n".format(flavor, model_type, uri) +
        "    preds = SD['model'].predict(img)\n" +
		"    return preds\n" +
		"$BODY$ LANGUAGE plpython3u;"
	)
    plpy.execute(stmt)
    return None
$$ LANGUAGE plpython3u;

CREATE TRIGGER create_model
AFTER INSERT ON ml.models
FOR EACH ROW
EXECUTE FUNCTION ml.create_model_trigger();
