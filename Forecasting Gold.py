import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from sklearn.metrics import mean_squared_error,r2_score
plt.style.use('seaborn-whitegrid')
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('logammulia.csv',sep=';',parse_dates = ['tanggal'])
model = load_model('model.goldprice')

def split_train(dataframe, col):
    return dataframe.loc[:1561,col]
df_new = {}
df_new["Train"]= split_train(df, "harga")

def split_test(dataframe, col):
    return dataframe.loc[1562-60:,col]
df_new["Test"]= split_test(df, "harga")
transform_train = {}
transform_test = {}
scaler = {}

sc = MinMaxScaler(feature_range=(0,1))
a0 = np.array(df_new["Train"])
a1 = np.array(df_new["Test"])
a0 = a0.reshape(a0.shape[0],1)
a1 = a1.reshape(a1.shape[0],1)
transform_train = sc.fit_transform(a0)
transform_test = sc.fit_transform(a1)
scaler = sc
del a0
del a1

trainset = {}
testset = {}
X_train = []
y_train = []
for i in range(60,1562):
    X_train.append(transform_train[i-60:i,0])
    y_train.append(transform_train[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)
trainset["X"] = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
trainset["y"] = y_train
    
   
X_test = []
y_test = []    
for i in range(60, 605):
    X_test.append(transform_test[i-60:i,0])
    y_test.append(transform_test[i,0])
X_test, y_test = np.array(X_test), np.array(y_test)
testset["X"] = np.reshape(X_test, (X_test.shape[0], X_train.shape[1], 1))
testset["y"] = y_test


y_true = scaler.inverse_transform(testset["y"].reshape(-1,1))
y_pred = scaler.inverse_transform(model.predict(testset["X"]))
accuracy_result = {}
accuracy_result["True"] = y_true
accuracy_result["Pred"] = y_pred
accuracy_result = r2_score(y_true,y_pred)
valid = df.loc[1562:]
valid = valid.set_index('tanggal')
valid['Predictions'] = y_pred
valid['True'] = y_true
    
plt.figure(figsize=(14,6))
plt.title('Gold Prices Prediction with LSTM {}'.format(accuracy_result))
plt.plot(valid['True'])
plt.plot(valid['Predictions'])
plt.ticklabel_format(style='plain', axis = 'y')
plt.show()
