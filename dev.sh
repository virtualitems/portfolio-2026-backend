#!/bin/bash

sudo mkdir -p /var/run/demo.alejandrocr.co && sudo chown ec2-user:ec2-user /var/run/demo.alejandrocr.co

sudo -u ec2-user $(pwd)/venv/bin/uvicorn server.main:app --uds /var/run/demo.alejandrocr.co/unix.sock --reload --proxy-headers --app-dir $(pwd) --lifespan off --log-level debug

# sudo docker run redis:alpine --name redis -p 6379:6379