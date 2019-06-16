# grab-demandforecast
Submission for grab-demandforecast Grab Challenge on Traffic Demand Forecast

grab.py - contains all the wrapper functions which would train and ingest the data for evaluation (holdout data)
GrabChallenge.ipynb - Pipeline walkthrough of thoughts process and wrapper functions also on modelling choice.
models - models folder which contains all the weights for each geolocation model
trafficDemandData/mockTest.csv - mockTest.csv is the mockdata on how the evaluation function will have to be used, mainly for validation of function

Training data set was ommitted due to size and some models are still being trained and will not be able to meet submission and will have to trained by evaluator

Sidenote: To parallelize training of model and to add retrain logic when evaluation is weak on validation data set, no retrain logic due to scale of intention
