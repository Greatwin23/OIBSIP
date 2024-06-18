import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

#importing data set
data = pd.read_csv(r'C:\Users\great\housing\Housing.csv')
print(data.head())#which prints 1st 5 records in the data set
print(data.tail())#last 5 records

#understanding our data
print("no of rows and columns in dataset",data.shape)
print(data.info())

#checking our data is there any missing values
print(data.isnull().sum())#if null value is present,sum it and display
categorical_col =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']#colums have yes or no values
print(data[categorical_col])

#binary encoding process converting categorical value(y/n) values into numerical value(1/0)
#we do this process because ml algo requires numerical input

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

#handle categorical varibale with dummy variable
dum_col=pd.get_dummies(data['furnishingstatus'])
print(dum_col)
dum_col=pd.get_dummies(data['furnishingstatus'],drop_first=True)#drop the 1st column
print(dum_col)
print("converting categorical values(T,F) into 0 and 1")
def dm(x):
    return x.map({True:1,False:0})
dum=dum_col.apply(dm)#converting process
print(dum)
print("Add the semifur,unfur columns to dataset")
data = pd.concat([data, dum], axis=1)#add the new columns (semifurnished,unfurnished) to dataset#axis=1 concate along with column
print(data.head())
print("dropped furnishingstatus column")
data.drop(['furnishingstatus'], axis=1, inplace=True)#axiz1 drop tha column,inplace=tru directly apply change(drop) to df without return new one
print(data)

#splitting data into training and testing phases
np.random.seed(0)#random genrating numbers starts from 0
df_train, df_test = train_test_split(data, train_size=0.7, test_size=0.3, random_state=100)
print("no of records in train set",df_train.shape)#display how many records are used for training
print("no of recors in test set",df_test.shape)

#scaling data for equal contribution of features
scalar=MinMaxScaler()
col_to_scale = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price']#cols to scale
df_train[col_to_scale]=scalar.fit_transform(df_train[col_to_scale])#scale train data
x_train=df_train
y_train=df_train.pop('price')#pop which seperate the mention column and kept it
lr=LinearRegression()
lr.fit(x_train,y_train)

#coff=lr.coef_
#print("cofficients",coff)#use to understand the relation b/w two features and target var
col_to_scale = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']
df_test[col_to_scale] = scalar.fit_transform(df_test[col_to_scale])#scale test data
#testing the model
y_test = df_test.pop('price')
x_test = df_test
predict=lr.predict(x_test)
#r2 defines goodness of fit 
r2 = r2_score(y_test, predict)
print("r2 score",r2)#higher r2 score means it good model
mse = mean_squared_error(y_test, predict)#mse = diff b/w acutal and predict values
print("Mean Squared Error:", mse)#lower mse means it good model
#comparing actual and predicted values
y_test.shape
#reshape array into 2d for ml algo reuirements
y_test_matrix = y_test.values.reshape(-1,1)#-1 it figure out size of array and ensure that fit in reshape array?,1=no of cols
# in df each column must be 1d array
#flatten which cnvrt multi darray into 1d arr
data_frame = pd.DataFrame({'actual': y_test_matrix.flatten(), 'predicted': predict.flatten()})
print(data_frame)
fig = plt.figure()
plt.scatter(y_test, predict)
plt.title('Actual vs Prediction')
plt.xlabel('Actual', fontsize=15)
plt.ylabel('Predicted', fontsize=15)
plt.show()
