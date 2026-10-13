from flask import Flask, request, render_template, jsonify
from flask import Response
import os
from flask_cors import CORS, cross_origin
from prediction_Validation_Insertion import pred_validation
from trainingModel import trainModel
from training_Validation_Insertion import train_validation
#import flask_monitoringdashboard as dashboard
from predictFromModel import prediction

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
#dashboard.bind(app)
CORS(app)


def get_filepath():
    """JSON ya form-data, dono se 'filepath' nikalta hai (415 error se bachne ke liye)."""
    data = request.get_json(silent=True)
    if data and data.get('filepath'):
        return data['filepath']
    if request.form.get('filepath'):
        return request.form['filepath']
    return None


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRouteClient():
    try:
        path = get_filepath()
        if path is None:
            return Response("Error Occurred! filepath not provided")

        pred_val = pred_validation(path)  # object initialization

        pred_val.prediction_validation()  # calling the prediction_validation function

        pred = prediction(path)  # object initialization

        # predicting for dataset present in database
        path = pred.predictionFromModel()
        return Response("Prediction File created at %s!!!" % path)

    except Exception as e:
        return Response("Error Occurred! %s" % e)


@app.route("/train", methods=['POST'])
@cross_origin()
def trainRouteClient():
    try:
        path = get_filepath()
        if path is None:
            return Response("Error Occurred! filepath not provided")

        train_valObj = train_validation(path)  # object initialization

        train_valObj.train_validation()  # calling the training_validation function

        trainModelObj = trainModel()  # object initialization
        trainModelObj.trainingModel()  # training the model for the files in the table

    except Exception as e:
        return Response("Error Occurred! %s" % e)
    return Response("Training successfull!!")


@app.route("/results", methods=['GET'])
@cross_origin()
def resultsRoute():
    """Predictions.csv ka summary JSON mein deta hai (dashboard ke liye)."""
    try:
        import math
        import pandas as pd
        df = pd.read_csv("Prediction_Output_File/Predictions.csv")
        preds = pd.to_numeric(df['Predictions'], errors='coerce').dropna().astype(int).tolist()
        total = len(preds)
        if total == 0:
            return jsonify(error="Predictions.csv khali hai"), 404
        defaults = sum(1 for p in preds if p == 1)
        size = max(1, math.ceil(total / 10))
        buckets = []
        for i in range(0, total, size):
            chunk = preds[i:i + size]
            d = sum(1 for p in chunk if p == 1)
            buckets.append({"label": "%d-%d" % (i + 1, i + len(chunk)), "defaults": d, "safe": len(chunk) - d})
        sample = [{"row": i + 1, "pred": preds[i]} for i in range(min(12, total))]
        return jsonify(total=total, defaults=defaults, safe=total - defaults,
                       rate=round(100.0 * defaults / total, 1), buckets=buckets, sample=sample)
    except FileNotFoundError:
        return jsonify(error="Abhi tak koi prediction nahi bani"), 404
    except Exception as e:
        return jsonify(error=str(e)), 500


port = int(os.getenv("PORT", 5001))
if __name__ == "__main__":
    app.run(port=port, debug=True)