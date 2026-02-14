#!/bin/bash

if ! command -v python3.11 &> /dev/null
then
    echo "python3.11 could not be found, please install it first."
    exit 1
fi

if ! command -v sqlite3 &> /dev/null
then
    echo "sqlite3 could not be found, please install it first."
    exit 1
fi

if [ ! -f database/db.sql ]; then
    echo "database/db.sql not found, please ensure the file exists."
    exit 1
fi

rm -rf ./venv

python3.11 -m venv venv

lspci | grep -i nvidia

nvidia-smi

nvcc --version

./venv/bin/python --version

./venv/bin/python -m pip install --upgrade pip

./venv/bin/pip install --no-cache-dir -r requirements.txt

rm -f database/database.sqlite3

sqlite3 database/database.sqlite3 < database/db.sql

mkdir -p mediafiles

mkdir -p staticfiles

mkdir -p prompts

mkdir -p vision

mkdir -p .ultralytics

mkdir -p logs

echo -e "\033[0;32mSetup complete.\033[0m"
