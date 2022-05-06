CREATE TYPE image AS (uri TEXT);


CREATE FUNCTION ml_version()
RETURNS TEXT
AS $$
	import rikai
	return rikai.__version__.version
$$ LANGUAGE plpython3u;


CREATE TYPE detection AS (label TEXT, box box, score real);

CREATE FUNCTION ssd(img image)
RETURNS SETOF detection
AS $$
	plpy.info("Image is:", img)
	return [{"label": "ssd", "box": ((1, 2), (3, 4)), "score": 0.85}]
$$ LANGUAGE plpython3u;

CREATE TABLE models (
	name VARCHAR(128) PRIMARY KEY,
	flavor VARCHAR(128),
	model_type VARCHAR(128),
	options JSONB
);
CREATE INDEX IF NOT EXISTS model_flavor_idx
ON models (flavor, model_type);

CREATE FUNCTION ml_detection(model TEXT, img image)
RETURNS SETOF detection
AS $$
	return []
$$ LANGUAGE plpython3u;