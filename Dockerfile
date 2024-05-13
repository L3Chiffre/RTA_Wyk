FROM python:3
RUN pip install numpy
RUN pip intall pandas
RUN pip install scikit-learn
COPY Class_perceptron.py .
cmd ["python","./Class_perceptron.py"]
