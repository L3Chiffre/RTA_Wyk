FROM python:3
WORKDIR /usr/src
RUN pip install numpy
RUN pip install pandas
RUN pip install scikit-learn
COPY ./src .
cmd ["python","./Class_perceptron.py"]
