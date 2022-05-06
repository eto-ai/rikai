CREATE TYPE image AS (uri TEXT, data bytea);


CREATE FUNCTION ml_version()
RETURNS TEXT
AS $$
	import rikai
	return rikai.__version__.version
$$ LANGUAGE plpython3u;


CREATE TYPE detection AS (label TEXT, box box, score real);

CREATE FUNCTION ssd(img image)
RETURNS detection[]
AS $$
	return [{"label": "ssd", "box": ((1, 2), (3, 4)), "score": 0.85}]
$$ LANGUAGE plpython3u;

