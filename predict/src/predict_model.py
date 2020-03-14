from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import pandas as pd
import json
import joblib

app = Flask(__name__)

HL_model = p.load(open('/app/HL_model.pickle', 'rb'))
CL_model = p.load(open('/app/CL_model.pickle', 'rb'))
final_pipeline = joblib.load('/app/model.joblib')

@app.route('/')
@app.route('/index')
def index():
    return "Welcome - this is the Heating-Cooling Load Model!"

@app.route('/load/', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data, index=[0])
    test_processed = final_pipeline.transform(df)
    final_predictions = {}
    final_predictions['HL'] = HL_model.predict(test_processed)[0]
    final_predictions['CL'] = CL_model.predict(test_processed)[0]
    return json.dumps(final_predictions)

if __name__ == '__main__':
    app.run(port=8080, debug=True, host='0.0.0.0')