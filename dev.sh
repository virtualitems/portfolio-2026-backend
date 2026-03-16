#!/bin/bash

# Eliminar cualquier socket existente
sudo rm -f /var/run/demo.alejandrocr.co/unix.sock && \
# Crear directorio con permisos adecuados
sudo mkdir -p /var/run/demo.alejandrocr.co && \
sudo chown ec2-user:nginx /var/run/demo.alejandrocr.co && \
sudo chmod 775 /var/run/demo.alejandrocr.co && \
# Lanzar hypercorn (que creará el socket automáticamente)
sudo -u ec2-user $(pwd)/venv/bin/python -m hypercorn server.main:app --bind unix:/var/run/demo.alejandrocr.co/unix.sock --umask 113 --reload --debug

namei -nom /var/run/demo.alejandrocr.co/unix.sock

# sudo docker run redis:alpine --name redis -p 6379:6379