def prepData(data,geolocation=None,lags=4):
    ##Code was written to use lag 4 due to vector of 5 map to vector of 5 (based of the problem)
    import random
    import numpy as np
    geoChoice = random.choice(data.geohash6)
    if geolocation != None:
        geoChoice = geolocation
    print("GeoHash: {}".format(geoChoice))
    oneRegionData = data[data.geohash6 == geoChoice]
    oneRegionData.sort_values(by=['day','timestampInt'],inplace=True,ascending=False)
    for l in range(1,lags+1):
        oneRegionData['demandLagTermT_'+str(l)] = oneRegionData['demand'].shift(-l)
        oneRegionData['timestampIntLagTermT_'+str(l)] = oneRegionData['timestampInt'].shift(-l)
    #range forward is fixed as we are predicting forward    
    for l in range(1,6):
        oneRegionData['demandForwardTermT_'+str(l)] = oneRegionData['demand'].shift(l)
    oneRegionData=oneRegionData.dropna()
    oneRegionData=oneRegionData.drop(['day','geohash6','timestamp'], axis=1)
    
    forwardData = [f for f in oneRegionData.axes[1] if "ForwardTerm" in f]
    demandLagData = [l for l in oneRegionData.axes[1] if "demandLagTerm" in l]
    timestampIntLagData = [l for l in oneRegionData.axes[1] if "timestampIntLagTerm" in l]
    
    forwardData = oneRegionData[forwardData]
    demandLagData = oneRegionData[['demand']+demandLagData]
    timestampIntLagData = oneRegionData[['timestampInt']+timestampIntLagData]
    
    X=np.hstack((demandLagData[::-1].values,timestampIntLagData[::-1].values))
    X=np.reshape(X,(X.shape[0],int(X.shape[1]/2),2))
    
    y=forwardData[::-1].values
    y=y.reshape((y.shape[0],y.shape[1],1))
    
    return y,X
    
def splitData(X,y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,shuffle=False)
    return X_train, X_test, y_train, y_test

def evaluateTrainingModel(model,X_test,y_test):
    from sklearn.metrics import mean_squared_error
    y_hat=model.predict(X_test)
    y_test=y_hat.reshape(y_test.shape[0],y_test.shape[1])
    y_hat=y_hat.reshape(y_hat.shape[0],y_hat.shape[1])
    return mean_squared_error(y_test, y_hat)

def modelData(X,y,geolocation,verbose=0,epochs=200,batch_size=32):
    from keras.models import Sequential
    from keras.layers import Dense,LSTM,Flatten
    from keras.layers import RepeatVector,Dropout,TimeDistributed
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    X_train, X_test, y_train, y_test = splitData(X,y)
    modelPath = "models/"+geolocation+".hdf5"
    checkpointer = ModelCheckpoint(modelPath, verbose=0, save_weights_only=True)
    earlyStop= EarlyStopping(monitor='loss', mode='min')
    model = Sequential()
    model.add(LSTM(200, activation='relu',input_shape=X.shape[-2:],return_sequences=True))
    model.add(LSTM(200, activation='relu',return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')
    print(model.summary())
    model.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,verbose=verbose,callbacks=[checkpointer,earlyStop])
    res=evaluateTrainingModel(model,X_test,y_test)
    return res

def trainAllModel(csvFile="trafficDemandData/training.csv"):
    import pandas as pd 
    data = pd.read_csv(csvFile)
    data['timestampInt']=data['timestamp'].map(lambda x:int((x.split(':'))[0])*60+int((x.split(':'))[1]))
    data['timestampInt']=data['timestampInt'].map(lambda x:x/15)
    import os 
    for geolocation in list(set(data['geohash6'].values)):
        print("INFO: Training Model for Geohash {}".format(geolocation))
        modelPath = "models/"+geolocation+".hdf5"
        if os.path.isfile(modelPath):
            print ("INFO: Existing Model Present, Skipping {}".format(geolocation))
        else:
            y,X=prepData(data,geolocation=geolocation,lags=4)
            try:
                print("INFO: MSE Result: {}".format(modelData(X,y,geolocation)))
            except Exception as e:
                print("ERROR: {}".format(e))
                print("WARNING: Skipping {}".format(geolocation))

                
def ingestHoldOutData(csvFile):
    import pandas as pd 
    import numpy as np
    data = pd.read_csv(csvFile) 
    data['timestampInt']=data['timestamp'].map(lambda x:int((x.split(':'))[0])*60+int((x.split(':'))[1]))
    data['timestampInt']=data['timestampInt'].map(lambda x:x/15)
    data.sort_values(by=['geohash6','day','timestampInt'],inplace=True,ascending=False)
    return data

def createTestFeature(data, lags=4):
    import numpy as np
    for l in range(1,lags+1):
        data['demandLagTermT_'+str(l)] = data['demand'].shift(-l)
        data['timestampIntLagTermT_'+str(l)] = data['timestampInt'].shift(-l)
    data=data.dropna()
    data=data.drop(['day','geohash6','timestamp'],axis=1)
    demandLagData = [l for l in data.axes[1] if "demandLagTerm" in l]
    timestampIntLagData = [l for l in data.axes[1] if "timestampIntLagTerm" in l]
    demandLagData = data[['demand']+demandLagData]
    timestampIntLagData = data[['timestampInt']+timestampIntLagData]
        
    X=np.hstack((demandLagData[::-1].values,timestampIntLagData[::-1].values))
    X=np.reshape(X,(X.shape[0],int(X.shape[1]/2),2))
    return X[-1]

def createModel(X):
    from keras.models import Sequential
    from keras.layers import Dense,LSTM,Flatten
    from keras.layers import Dropout,TimeDistributed
    model = Sequential()
    model.add(LSTM(200, activation='relu',input_shape=X.shape[-2:],return_sequences=True))
    model.add(LSTM(200, activation='relu',return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(1)))
    return model

def evaluateModel(model,X_test,y_test):
    from sklearn.metrics import mean_squared_error
    y_hat=model.predict(X_test)
    y_hat=y_hat.reshape(y_hat.shape[1])
    return mean_squared_error(y_test, y_hat)

def testModel(csvFile,y):
    '''given input data, loop through each distinct region
    Requirement: y must be available as an object with key-value pair of {region:t+1,t+2,t+3,t+4,T+5}, similar to X
    Loop through keys to read models and evaluate.
    save mse to global
    '''
    import os
    assert (type(y)==dict), "y is required to a be a key-value pair with {region:[t+1,t+2,t+3,t+4,t+5]}"
    res=[]
    data=ingestHoldOutData(csvFile)
    for geolocation in list(set(data['geohash6'].values)):
        X=createTestFeature(data[data.geohash6==geolocation])
        modelPath = "models/"+geolocation+".hdf5"
        if os.path.isfile(modelPath):
            print ("INFO: Importing {} Model for prediction".format(geolocation))
        else:
            raise Exception("{} model wight not found, please re-run training of weights".format(geolocation))
        X=X.reshape(1,X.shape[0],X.shape[1])
        model = createModel(X)
        model.load_weights(modelPath)
        res.append(evaluateModel(model,X,y[geolocation]))
    return res
        