FROM nginx:latest

RUN rm /etc/nginx/conf.d/default.conf

COPY default.conf /etc/nginx/conf.d/default.conf

RUN mkdir -p /var/log/nginx/tekiai

EXPOSE 80

# Iniciar nginx
CMD ["nginx", "-g", "daemon off;"]
