CREATE EXTENSION plpython3u;
CREATE EXTENSION rikai;

INSERT INTO ml.models (name, flavor, model_type)
VALUES ('ssd', 'pytorch', 'ssd')
