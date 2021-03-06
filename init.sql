GRANT ALL PRIVILEGES ON DATABASE "image_db" to fabio_test;

CREATE TABLE IF NOT EXISTS image_classes (
   image_id SERIAL PRIMARY KEY,
   pixel_data BYTEA NOT NULL,
   class INT NOT NULL
);