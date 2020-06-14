from flask import Flask, request, render_template
from flask import Response
from flask_cors import CORS, cross_origin
import flask_monitoringdashboard as dashboard
import os
from prediction_Validation_Insertion import pred_validation
from predictFromModel import prediction
from trainingModel import trainModel
from training_Validation_Insertion import train_validation

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
dashboard.bind(app)
CORS(app)

@app.route("/", methods=['GET']) # Main Page
@cross_origin() # To handle various route
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST']) # Prediction Route
@cross_origin()
def predictRouteClient():
    try:
        if request.json is not None: # prection file access from Postman server

            path = request.json['filepath'] # Obtained file path

            pred_val = pred_validation(path) # object initialization

            pred_val.prediction_validation() # calling the prediction_validation function

            pred = prediction(path) # object initialization

            path = pred.predictionFromModel() # predicting for dataset using saved model

            return Response("Prediction File created at %s!!!" % path) # Predicted file path

        elif request.form is not None: # prection file access from Web API

            path = request.form['filepath'] # Obtained file path

            pred_val = pred_validation(path) # object initialization

            pred_val.prediction_validation() # calling the prediction_validation function

            pred = prediction(path) # object initialization

            path = pred.predictionFromModel() # predicting for dataset using saved model

            return Response("Prediction File created at %s!!!" % path) # Predicted file path

    except ValueError:
        return Response("Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("Error Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" %e)

@app.route("/train", methods=['POST']) # Training Route
@cross_origin()
def trainRouteClient():
    try:
        if request.json['filepath'] is not None: # prection file access from Postman server (Main folder file:- RawData)

            path = request.json['filepath'] # Obtained file path

            train_valObj = train_validation(path) # object initialization

            train_valObj.train_validation() # calling the training_validation function

            trainModelObj = trainModel() # object initialization

            trainModelObj.trainingModel() # training the model for the files in the table

    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)
    return Response("Training successfull!!")

port = int(os.getenv("PORT",5001)) # port obtained for local machine

if __name__ == "__main__":
    # app.run(debug=True) # Enable when upload to production
    app.run(port=port,debug=True) # Disable when upload to production
