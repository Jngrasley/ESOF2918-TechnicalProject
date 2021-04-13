#ESOF 2918 TECH PROJECT
#Members:
    #Jarrod
    #Jack
    #Mark

#beginning imports
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import classification_report

#define functions used later in the code, these functions convert string data to corresponding
def admission_to_numeric():
    covid_data.loc[covid_data['Admission'] == 'Urgent', 'Admission'] = 0.0
    covid_data.loc[covid_data['Admission'] == 'Trauma', 'Admission'] = 1.0
    covid_data.loc[covid_data['Admission'] == 'Emergency', 'Admission'] = 2.0
def severity_to_numeric():
    covid_data.loc[covid_data['Severity'] == 'Minor', 'Severity'] = 0.0
    covid_data.loc[covid_data['Severity'] == 'Moderate', 'Severity'] = 1.0
    covid_data.loc[covid_data['Severity'] == 'Extreme', 'Severity'] = 2.0
def age_to_numeric():
    covid_data.loc[covid_data['Age'] == '0-10', 'Age'] = 0.0
    covid_data.loc[covid_data['Age'] == '11-20', 'Age'] = 1.0
    covid_data.loc[covid_data['Age'] == '21-30', 'Age'] = 2.0
    covid_data.loc[covid_data['Age'] == '31-40', 'Age'] = 3.0
    covid_data.loc[covid_data['Age'] == '41-50', 'Age'] = 4.0
    covid_data.loc[covid_data['Age'] == '51-60', 'Age'] = 5.0
    covid_data.loc[covid_data['Age'] == '61-70', 'Age'] = 6.0
    covid_data.loc[covid_data['Age'] == '71-80', 'Age'] = 7.0
    covid_data.loc[covid_data['Age'] == '81-90', 'Age'] = 8.0
    covid_data.loc[covid_data['Age'] == '91-100', 'Age'] = 9.0
def range_to_numeric():
    covid_data.loc[covid_data['Range'] == '0-10', 'Range'] = 0.0
    covid_data.loc[covid_data['Range'] == '11-20', 'Range'] = 1.0
    covid_data.loc[covid_data['Range'] == '21-30', 'Range'] = 2.0
    covid_data.loc[covid_data['Range'] == '31-40', 'Range'] = 3.0
    covid_data.loc[covid_data['Range'] == '41-50', 'Range'] = 4.0
    covid_data.loc[covid_data['Range'] == '51-60', 'Range'] = 5.0
    covid_data.loc[covid_data['Range'] == '61-70', 'Range'] = 6.0
    covid_data.loc[covid_data['Range'] == '71-80', 'Range'] = 7.0
    covid_data.loc[covid_data['Range'] == '81-90', 'Range'] = 8.0
    covid_data.loc[covid_data['Range'] == '91-100', 'Range'] = 9.0
    covid_data.loc[covid_data['Range'] == '100+', 'Range'] = 10.0

#introduce the covid datasets
covid_data = pd.read_csv('coviddataset.csv') #no delimiters required

#declare splitting constants
    #this is how many lines will be used to train the models
dataset_row_count = len(covid_data.index)
dataset_split = (int(dataset_row_count*0.05)) #5% of the dataset will be used for training the models 

#pre-processing and imputation
    #stage includes reducing csv to only the prediction columns
    #begin preprocessing with dropping irrelavent tables
    #important columns include the city code of patient, severity of illness, age, and LOS
colList = list(covid_data.columns)
covid_data = covid_data[[colList[11]] + [colList[12]] + [colList[13]] + [colList[15]] + [colList[17]]]

#alter column names for proper identification in results
covid_data = covid_data.rename(columns={"Stay": "Range", "Severity of Illness": "Severity", "City_Code_Patient": "City", "Type of Admission": "Admission"})

#imputation correction based on the higher avg value
covid_data['City'].fillna(value=8, inplace=True)
covid_data.loc[covid_data['Range'] == 'More than 100 Days', 'Range'] = '100+'

#convert string values to numerical values, see notepad file for referenced values!
admission_to_numeric()
severity_to_numeric()
age_to_numeric()
range_to_numeric()
#convert numerical objects to float64 for model
covid_data['Admission'] = covid_data.Admission.astype(float)
covid_data['Severity'] = covid_data.Severity.astype(float)
covid_data['Age'] = covid_data.Age.astype(float)
covid_data['Range'] = covid_data.Range.astype(float)

print(covid_data.info())

#begin coding for the logistic regression algorithm

#split the dataset into 2 parts, one part for training it and one part for testing it
LR_training_data = covid_data.iloc[:dataset_split, :]
LR_testing_data = covid_data.iloc[dataset_split+1:, :]

#define inputs and outputs for the model
LR_training_input = scale(LR_training_data)
LR_training_output = LR_training_data['Range'].values

#begin defining the model and fit to the dataset
LRModel = LogisticRegression(multi_class='multinomial', solver='sag')
start = time.time()
LRModel.fit(LR_training_input, LR_training_output)
stop = time.time() 
print(LRModel.score(LR_training_input, LR_training_output))

#now that the model has been trained, input the testing values and compare results
LR_testing_input = scale(LR_testing_data)

LR_testing_output = LRModel.predict(LR_testing_input)
print(classification_report(LR_testing_output, LR_testing_data['Range'].values))
LR_training_time = stop - start
print(f"Training time: {LR_training_time}s")

#begin coding for the neural networks algorithm

#split the nn modeldataset into 2 parts, one part for training it and one part for testing it
NN_training_data = covid_data.iloc[:dataset_split, :]
NN_testing_data = covid_data.iloc[dataset_split+1:, :]
      
#define inputs and outputs for the model
NN_training_input = scale(NN_training_data)
NN_training_output = NN_training_data['Range'].values

#begin defining the model and fit to the dataset
NNModel = MLPClassifier()
start = time.time()
NNModel.fit(NN_training_input, NN_training_output)
stop = time.time()
print(NNModel.score(NN_training_input, NN_training_output))

NN_testing_input = scale(NN_testing_data)
NN_testing_output = NNModel.predict(NN_testing_input)

print(classification_report(NN_testing_output, NN_testing_data['Range'].values))
NN_training_time = stop - start
print(f"Training time: {NN_training_time}s")

#begin coding for the decision trees algorithm

#split the dt dataset into 2 parts, one part for training it and one part for testing it
DT_training_data = covid_data.iloc[:dataset_split, :]
DT_testing_data = covid_data.iloc[dataset_split+1:, :]

#define inputs and outputs for the model
DT_training_input = scale(DT_training_data)
DT_training_output = DT_training_data['Range'].values

#begin defining the model and fit to the dataset
DTModel = DecisionTreeRegressor()
start = time.time()
DTModel.fit(DT_training_input, DT_training_output)
stop = time.time()
print(DTModel.score(DT_training_input, DT_training_output))

DT_testing_input = scale(DT_testing_data)
DT_testing_output = DTModel.predict(DT_testing_input)

print(classification_report(DT_testing_output, DT_testing_data['Range'].values))
DT_training_time = stop - start
print(f"Training time: {DT_training_time}s")


#the following is for displaying the results in a presentatble format
   
#bar graph
#first one shows all the ranges
ranges = ['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100','100+']
ranges_labels = np.arange(len(ranges))
bar_width = 0.40

#create new arrays that are filled with the algorithm results
range_result_array = []
LR_range_result_array = []
NN_range_result_array = []
DT_range_result_array = []

updated_covid_data = covid_data.iloc[dataset_split+1:, :]

counting = np.array(updated_covid_data['Range'])
for x in range(11):
    range_result_array.append(np.count_nonzero(counting == (x)))

counting = np.array(LR_testing_output).reshape(-1,1)
for x in range(11):
    LR_range_result_array.append(np.count_nonzero(counting == (x)))

counting = np.array(NN_testing_output).reshape(-1,1)
for x in range(11):
    NN_range_result_array.append(np.count_nonzero(counting == (x)))

counting = np.array(DT_testing_output).reshape(-1,1)
for x in range(11):
    DT_range_result_array.append(np.count_nonzero(counting == (x)))

figure, range_ax = plt.subplots(figsize=(12,6))
range_bar1 = range_ax.bar(ranges_labels - bar_width*(3/4), range_result_array, bar_width/2, label='Dataset')
range_bar2 = range_ax.bar(ranges_labels - bar_width/4, LR_range_result_array, bar_width/2, label='Logistic Regression')
range_bar3 = range_ax.bar(ranges_labels + bar_width/4, NN_range_result_array, bar_width/2, label='Neural Network')
range_bar4 = range_ax.bar(ranges_labels + bar_width*(3/4), DT_range_result_array, bar_width/2, label='Decision Tree')

range_ax.set_title('Range Comparisons')
range_ax.set_xlabel('Stay Ranges')
range_ax.set_xticks(ranges_labels)
range_ax.set_xticklabels(ranges)
range_ax.set_ylabel('Number of Patients')
range_ax.set_yticks([0,20000,40000,60000,80000,100000])

range_ax.legend()

def bar_labels(bars):
    for bar in bars:
        h = bar.get_height()
        range_ax.annotate('{}'.format(h), xy=(bar.get_x() + bar.get_width() / 2, h),xytext=(0, 3),textcoords="offset points",ha='center', va='bottom',rotation=90)

bar_labels(range_bar1)
bar_labels(range_bar2)
bar_labels(range_bar3)
bar_labels(range_bar4)

plt.show()

#pie graphs
#shows percentage of short/normal/long for all algorithm predictions

pie_array = [0,0,0]
counting = np.array(updated_covid_data['Range'])
pie_array[0] += (np.count_nonzero(counting == (0)))
pie_array[1] += (np.count_nonzero(counting == (1)))
for x in range(9):
    pie_array[2] += (np.count_nonzero(counting == (x+2)))

LR_pie_array = [0,0,0]
counting = np.array(LR_testing_output)
LR_pie_array[0] = (np.count_nonzero(counting == (0)))
LR_pie_array[1] = (np.count_nonzero(counting == (1)))
for x in range(9):
    LR_pie_array[2] += (np.count_nonzero(counting == (x+2)))

NN_pie_array = [0,0,0]
counting = np.array(NN_testing_output)
NN_pie_array[0] = (np.count_nonzero(counting == (0)))
NN_pie_array[1] = (np.count_nonzero(counting == (1)))
for x in range(9):
    NN_pie_array[2] += (np.count_nonzero(counting == (x+2)))
    
DT_pie_array = [0,0,0]
counting = np.array(DT_testing_output)
DT_pie_array[0] = (np.count_nonzero(counting == (0)))
DT_pie_array[1] = (np.count_nonzero(counting == (1)))
for x in range(9):
    DT_pie_array[2] += (np.count_nonzero(counting == (x+2)))

pie_labels = ['Short', 'Normal', 'Long']

plt.pie(pie_array, labels=pie_labels, autopct='%1.1f%%')
plt.title("Dataset LOS %")
plt.axis('equal')
plt.show()
plt.pie(LR_pie_array, labels=pie_labels, autopct='%1.1f%%')
plt.title("Logistic Regression LOS %")
plt.axis('equal')
plt.show()
plt.pie(NN_pie_array, labels=pie_labels, autopct='%1.1f%%')
plt.title("Neural Network LOS %")
plt.axis('equal')
plt.show()
plt.pie(DT_pie_array, labels=pie_labels, autopct='%1.1f%%')
plt.title("Decision Tree LOS %")
plt.axis('equal')
plt.show()

#end code