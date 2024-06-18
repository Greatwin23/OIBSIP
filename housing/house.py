import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

#importing data set
data = pd.read_csv(r'C:\Users\great\housing\Housing.csv')
print(data.head())
print(data.tail())

#understanding our data
print("no of rows and columns in dataset",data.shape)
print(data.info())

#checking our data is there any missing values
print(data.isnull().sum())
categorical_col =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']#colums have yes or no values
print(data[categorical_col])

#Binary encoding process converting categorical value(y/n) into numerical value(1/0)
def binary_map(x):
    """
    Function to map 'yes' and 'no' values to 1 and 0, respectively.
    
    Parameters:
    x (pandas Series): Input Series containing 'yes' and 'no' values.
    
    Returns:
    pandas Series: Mapped Series with 'yes' mapped to 1 and 'no' mapped to 0.
    """
    return x.map({'yes':1,'no':0})
data[categorical_col]=data[categorical_col].apply(binary_map)
print(data[categorical_col])

#Handle categorical varibale with dummy variable
dum_col=pd.get_dummies(data['furnishingstatus'])
print(dum_col)
dum_col=pd.get_dummies(data['furnishingstatus'],drop_first=True)
print(dum_col)
print("converting categorical values(T,F) into 0 and 1")

def dm(x):
    return x.map({True:1,False:0})
dum=dum_col.apply(dm)#converting process
print(dum)
print("Add the semifur,unfur columns to dataset")
data = pd.concat([data, dum], axis=1)
print(data.head())
print("dropped furnishingstatus column")
data.drop(['furnishingstatus'], axis=1, inplace=True)
print(data)


#Splitting data into training and testing phases
np.random.seed(0)
df_train, df_test = train_test_split(data, train_size=0.7, test_size=0.3, random_state=100)
print("no of records in train set",df_train.shape)
print("no of recors in test set",df_test.shape)


#Scaling data for equal contribution of features
scalar=MinMaxScaler()
col_to_scale = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price']
df_train[col_to_scale]=scalar.fit_transform(df_train[col_to_scale])
x_train=df_train
y_train=df_train.pop('price')
lr=LinearRegression()
lr.fit(x_train,y_train)
col_to_scale = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']
df_test[col_to_scale] = scalar.fit_transform(df_test[col_to_scale])#scale test data


#testing the model
y_test = df_test.pop('price')
x_test = df_test
predict=lr.predict(x_test)

#r2 defines goodness of fit 
r2 = r2_score(y_test, predict)
print("r2 score",r2)
mse = mean_squared_error(y_test, predict)
print("Mean Squared Error:", mse)
#comparing actual and predicted values
y_test.shape
#reshape array into 2d for ml algo reuirements
y_test_matrix = y_test.values.reshape(-1,1)

#flatten which convert multi darray into 1d arr
data_frame = pd.DataFrame({'actual': y_test_matrix.flatten(), 'predicted': predict.flatten()})
print(data_frame)
fig = plt.figure()
plt.scatter(y_test, predict)
plt.title('Actual vs Prediction')
plt.xlabel('Actual', fontsize=15)
plt.ylabel('Predicted', fontsize=15)
plt.show()
