#!/bin/bash

#przejdż do katalogu gdzie jest skrypt
directory=$(dirname $0)
cd $directory


#zmienne
image_name=rta_1
docker_name=rta_1


#zbuduj obraz
echo "buduje obraz dockera o ${image_name}"
docker build -t $image_name:latest .

#usuń kontener jeżeli jakiś teraz istnieje
echo "usuwam kontener jeżeli istnieje"
docker rm -f $docker_name || true


#uruchom obraz
echo "uruchamiam nowy kontener o nazwie: ${docker_name} z obrazu ${image_name}"
docker run --name $docker_name $image_name