# Experimental Postgres Rikai Extension

PostgreSQL Rikai Extension

```sh
pip install rikai
make install
```

To load `rikai` PostgreSQL extension

```sql
CREATE EXTENSION plpython3u;
CREATE EXTENSION rikai;
```

Create a model via `INSERT INTO`

```sql
INSERT INTO ml.models
(name, flavor, model_type, uri)
VALUES
('cat_detector', 'pytorch', 'ssd', 's3://bucket/to/cat.pth')
```

A function `ml.<model_name>` will be created.

To use the registered model for inference:

```sql
SELECT ml.cat_detector(image) FROM cat_dataset
```

Show all registered models

```sql
SELECT * FROM ml.models;
```

## Limitations

- `rikai` needs to be installed with the system python.
- Batch inference.
- Not ready for production yet.
