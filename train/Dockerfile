FROM tiangolo/python-machine-learning:python3.7

RUN conda install -y flask numpy pandas scikit-learn xlrd
RUN conda install -y joblib

COPY ./src/train_model.py /app/train_model.py
COPY ./src/util.py /app/util.py
COPY ./resources/ENB2012_data.xlsx /app/ENB2012_data.xlsx
COPY ./bin/start.sh /start.sh

RUN chmod +x /start.sh