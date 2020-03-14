FROM tiangolo/python-machine-learning:python3.7

RUN conda install -y flask numpy pandas scikit-learn
COPY ./src/predict_model.py /app/predict_model.py
COPY ./src/util.py /app/util.py
COPY ./model/HL_model.pickle /app/HL_model.pickle
COPY ./model/CL_model.pickle /app/CL_model.pickle
COPY ./model/model.joblib /app/model.joblib
COPY ./bin/start.sh /start.sh
EXPOSE 8080

RUN chmod +x /start.sh