#!/bin/bash

#przejdż do katalogu gdzie jest skrypt
directory=$(dirname $0)
cd $directory

echo "podaj nazwę commita"
read commit_name



git add .
git commit -m "${commit_name}"
git push origin main