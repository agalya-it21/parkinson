import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

#Loading the data from csv file to a pandas Dataframe
parkinsons_data = pd.read_csv('/content/parkinsons.csv')


#Printing first 5 Rows of the DataFrame
parkinsons_data.head()


#No. of Rows and Columns in the DataFrame 
parkinsons_data.shape

# More Information about the Datset
parkinsons_data.info()

# Checking for missing value in each column
parkinsons_data.isnull().sum()


#Getting some StatisticalMeasures about the data
parkinsons_data.describe()


# Distribution of Target Variable
parkinsons_data['status'].value_counts()


#Grouping the Data based on the target variable
parkinsons_data.groupby('status').mean()


X = parkinsons_data.drop(columns=['name','status'], axis = 1)
Y = parkinsons_data['status']

print(X)

print(Y)


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2 , random_state=2)

print(X.shape, X_train.shape, X_test.shape)

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(X_train)


model = svm.SVC(kernel='linear')

# Training the SVM model with Training data
model.fit(X_train, Y_train)


# Computing the accuracy score for training and testing data
import matplotlib.pyplot as plt
train_scores = []
test_scores = []
c_values = np.arange(0.1, 1.1, 0.1)
for c in c_values:
    model = svm.SVC(kernel='linear', C=c)
    model.fit(X_train, Y_train)
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(Y_train, train_pred)
    train_scores.append(train_acc)
    
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(Y_test, test_pred)
    test_scores.append(test_acc)

# Plotting the accuracy scores as a graph
plt.plot(c_values, train_scores, label='Training accuracy')
plt.plot(c_values, test_scores, label='Testing accuracy')
plt.legend()
plt.title('Accuracy scores')
plt.xlabel('C values')
plt.ylabel('Accuracy score')
plt.show()


# accuracy score on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)



print('Accuracy score of training data : ', training_data_accuracy)


input_data = (297.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)

# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the data
std_data = scaler.transform(input_data_reshaped)

prediction = model.predict(std_data)
print(prediction)


if (prediction[0] == 0):
  print("The Person does not have Parkinsons Disease")

else:
  print("The Person has Parkinsons")