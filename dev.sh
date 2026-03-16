#!/bin/bash

sudo mkdir -p /var/run/demo.alejandrocr.co && sudo chown ec2-user:ec2-user /var/run/demo.alejandrocr.co

cd $(pwd) && sudo -u ec2-user $(pwd)/venv/bin/hypercorn server.main:app --bind unix:/var/run/demo.alejandrocr.co/unix.sock --reload --debug

# sudo docker run redis:alpine --name redis -p 6379:6379