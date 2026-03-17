-- PostgreSQL

CREATE TABLE IF NOT EXISTS persons (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,
    deleted_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reports (
    id SERIAL PRIMARY KEY,
    person_id INTEGER NOT NULL,
    observations TEXT NOT NULL,
    evidence TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP,
    FOREIGN KEY (person_id) REFERENCES persons(id)
);

-- Índices para optimizar consultas comunes
CREATE INDEX IF NOT EXISTS idx_persons_email ON persons(email);
CREATE INDEX IF NOT EXISTS idx_persons_deleted_at ON persons(deleted_at);
CREATE INDEX IF NOT EXISTS idx_reports_person_id ON reports(person_id);
CREATE INDEX IF NOT EXISTS idx_reports_deleted_at ON reports(deleted_at);
CREATE INDEX IF NOT EXISTS idx_reports_created_at ON reports(created_at);
