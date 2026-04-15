import pickle
from pathlib import Path

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import sklearn.metrics._scorer as scorer


if not hasattr(scorer, '_passthrough_scorer') and hasattr(scorer, '_PassthroughScorer'):
    scorer._passthrough_scorer = scorer._PassthroughScorer


BASE_DIR = Path(__file__).resolve().parent


def load_pickle(file_name):
    with (BASE_DIR / file_name).open('rb') as file_obj:
        loaded_object = pickle.load(file_obj)

    if hasattr(loaded_object, 'best_estimator_'):
        return loaded_object.best_estimator_

    return loaded_object

app = Flask(__name__)
model = load_pickle('housepred.pkl')
scaler = load_pickle('scaler.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    a_data = (np.array(list(data.values())).reshape(1,-1))
    new_data = scaler.transform(a_data)
    output = model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=model.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))


if __name__ =="__main__":
    app.run(debug=True)