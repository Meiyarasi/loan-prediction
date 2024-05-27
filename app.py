
from flask import Flask,render_template,request,redirect, redirect, url_for, session
from flask_cors import CORS,cross_origin
import pandas as pd
import numpy as np


app=Flask(__name__)
cors=CORS(app)
# model=pickle.load(open('Decision_Tree_Model.pkl','rb'))
 

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
#the model used to fit&predict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#pipeline with its' preprocessor's transformers
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

#used for estimating model accuracy and getting reports
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
#Model 1 Personal Loan Prediction

df = pd.read_csv(r"C:\Users\Pavithran\Desktop\Loan_Approval_Prediction\LoanPrediction.csv")
personal_loan = df
personal_loan.drop('Type_of_employement',axis=1, inplace=True)
personal_loan.drop('Gram',axis=1, inplace=True)
personal_loan.drop('Gold_loan_amount',axis=1, inplace=True)
personal_loan.drop('Score',axis=1, inplace=True)
personal_loan.drop('Fee_structure',axis=1, inplace=True)
personal_loan.drop('Edu_loan_amount',axis=1, inplace=True)
personal_loan.drop('Edu_coapp_income',axis=1, inplace=True)
personal_loan.drop('Edu_credit_history',axis=1, inplace=True)
personal_loan.drop('Gold_loan_status',axis=1, inplace=True)
personal_loan.drop('Edu_loan_status',axis=1, inplace=True)
personal_loan.drop('Gold_loan_term',axis=1, inplace=True)
print("Model1")

X=personal_loan.drop('Per_loan_status',axis=1)
y=personal_loan['Per_loan_status']
le = LabelEncoder()
y = LabelEncoder().fit_transform(personal_loan['Per_loan_status'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 12)



xgb = XGBClassifier(learning_rate=0.4,max_depth=7)
xgb.fit(np.asmatrix(X_train), y_train)
print("Personal Loan Model Trained")
score = xgb.score(np.asmatrix(X_test), y_test)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
y_pred = xgb.predict(X_test)
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
  for j in range(conf_matrix.shape[1]):
    ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
print(plt.show())

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
data = precision_recall_fscore_support(y_test, y_pred, average='macro')

print('Precision: ', data[0]*100)
print('F1 Score: ',data[2]*100)
print('Recall: ', data[1]*100)
# sample = np.array([27,3, 20000, 100000, 12, 1]).reshape(1,-1)
# print(sample)
# print(xgb.predict(sample))


#Gold loan
gold_loan = pd.read_csv(r"C:\Users\Pavithran\Desktop\Loan_Approval_Prediction\LoanPrediction.csv")
gold_loan.head()

gold_loan.drop('Age',axis=1, inplace=True)
gold_loan.drop('Type_of_employement',axis=1, inplace=True)
gold_loan.drop('Work_experience',axis=1, inplace=True)
gold_loan.drop('Income',axis=1, inplace=True)
gold_loan.drop('Per_Loan_amount',axis=1, inplace=True)
gold_loan.drop('Per_loan_amount_term',axis=1, inplace=True)
gold_loan.drop('Per_credit_history',axis=1, inplace=True)
gold_loan.drop('Score',axis=1, inplace=True)
gold_loan.drop('Edu_loan_amount',axis=1, inplace=True)
gold_loan.drop('Edu_coapp_income',axis=1, inplace=True)
gold_loan.drop('Edu_credit_history',axis=1, inplace=True)
gold_loan.drop('Per_loan_status',axis=1, inplace=True)
gold_loan.drop('Edu_loan_status',axis=1, inplace=True)
gold_loan.drop('Fee_structure',axis=1, inplace=True)
print("Model2")

X=gold_loan.drop('Gold_loan_status',axis=1)
y=gold_loan['Gold_loan_status']
le = LabelEncoder()
y = LabelEncoder().fit_transform(gold_loan['Gold_loan_status'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 12)

goldmodel = XGBClassifier(learning_rate=0.4,max_depth=7)
goldmodel.fit(np.asmatrix(X_train), y_train)
print("Gold Loan Model Trained")
score = goldmodel.score(np.asmatrix(X_test), y_test)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
y_pred = goldmodel.predict(X_test)
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
  for j in range(conf_matrix.shape[1]):
    ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
print(plt.show())

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
# print('Precision: ', precision_score(y_test, y_pred)*100)
data = precision_recall_fscore_support(y_test, y_pred, average='macro')

print('Precision: ', data[0]*100)
print('F1 Score: ',data[2]*100)
print('Recall: ', data[1]*100)

#education loan
education_loan = pd.read_csv(r"C:\Users\Pavithran\Desktop\Loan_Approval_Prediction\LoanPrediction.csv")

education_loan.drop('Type_of_employement',axis=1, inplace=True)
education_loan.drop('Gram',axis=1, inplace=True)
education_loan.drop('Gold_loan_amount',axis=1, inplace=True)
education_loan.drop('Age',axis=1, inplace=True)
education_loan.drop('Work_experience',axis=1, inplace=True)
education_loan.drop('Income',axis=1, inplace=True)
education_loan.drop('Per_Loan_amount',axis=1, inplace=True)
education_loan.drop('Per_credit_history',axis=1, inplace=True)
education_loan.drop('Gold_loan_status',axis=1, inplace=True)
education_loan.drop('Per_loan_status',axis=1, inplace=True)
education_loan.drop('Per_loan_amount_term',axis=1, inplace=True)
education_loan.drop('Gold_loan_term',axis=1, inplace=True)
print("model3")

X=education_loan.drop('Edu_loan_status',axis=1)
y=education_loan['Edu_loan_status']

le = LabelEncoder()
y = LabelEncoder().fit_transform(education_loan['Edu_loan_status'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 12)

edumodel = XGBClassifier(learning_rate=0.4,max_depth=7)
edumodel.fit(np.asmatrix(X_train), y_train)
print("Educational Loan Model Trained")
score = edumodel.score(np.asmatrix(X_test), y_test)


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
y_pred = edumodel.predict(X_test)
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
  for j in range(conf_matrix.shape[1]):
    ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
print(plt.show())

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
# print('Precision: ', precision_score(y_test, y_pred)*100)
data = precision_recall_fscore_support(y_test, y_pred, average='macro')

print('Precision: ', data[0]*100)
print('F1 Score: ',data[2]*100)
print('Recall: ', data[1]*100)


@app.route('/',methods=['GET'])
def home():
    return render_template('home.html')





@app.route('/loan',methods=['GET','POST'])
def loan():
    return render_template('typesOfLoanPage.html')




@app.route('/personal')
def personal():
    return render_template('Personal.html')


@app.route('/predict',methods=['GET' , 'POST'])
def predict():
    if request.method == "POST":
        name = request.form.get('name')
        age = int(request.form.get('age'))
        temployment = request.form.get('temployment')
        experience = int(request.form.get('experience'))
        Income = int(request.form.get('Income'))
        Loan_amount = int(request.form.get('Loan_amount'))
        Loan_amount_term = int(request.form.get('Loan_amount_term'))
        credit_history = int(request.form.get('Credit_history'))
        data = np.array([age,experience, Income,Loan_amount, Loan_amount_term, credit_history]).reshape(1,-1)
        print(data)
        try:
            prediction = xgb.predict(data)
            print(prediction)
            if prediction == 1:
                pred = "Loan_Approved"
                return render_template('temp.html')
            else:
                pred = "Not_Approved"
                return render_template('notapproved.html')
        # prediction=model.predict(pd.DataFrame(columns=['Age', 'Work_experience', 'Income', 'Per_Loan_amount', 'Per_loan_amount_term', 'Per_credit_history'],
        #                         data=np.array([age,experience, Income,Loan_amount, Loan_amount_term, credit_history]).reshape(1, 11)))
        except:

            print(Exception)


@app.route('/gold')
def gold():
    return render_template('Gold.html')


@app.route('/predict1',methods=['GET' , 'POST'])
def predict1():
    if request.method == "POST":
        name = request.form.get('name')
        gram = int(request.form.get('gram'))
        Loan_amount = int(request.form.get('Loan_amount'))
        Loan_amount_term = int(request.form.get('Loan_amount_term'))
        
        data = np.array([gram,Loan_amount, Loan_amount_term]).reshape(1,-1)
        print(data)
        try:
            prediction = goldmodel.predict(data)
            print(prediction)
            if prediction == 1:
                pred = "Loan_Approved"
                return render_template('temp.html')
            else:
                pred = "Not_Approved"
                return render_template('notapproved.html')
        # prediction=model.predict(pd.DataFrame(columns=['Age', 'Work_experience', 'Income', 'Per_Loan_amount', 'Per_loan_amount_term', 'Per_credit_history'],
        #                         data=np.array([age,experience, Income,Loan_amount, Loan_amount_term, credit_history]).reshape(1, 11)))
        except:

            print(Exception)


@app.route('/education')
def education():
    return render_template('Educational.html')


@app.route('/predict2',methods=['GET' , 'POST'])
def predict2():
    if request.method == "POST":
        name = request.form.get('name')
        score = int(request.form.get('score'))
        Fee_structure = int(request.form.get('Fee_structure'))
        Cincome = int(request.form.get('Cincome'))
        Loan_amount = int(request.form.get('Loan_amount'))
        Credit_history = int(request.form.get('Credit_history'))
        
        data = np.array([score,Fee_structure,Loan_amount,Cincome,Credit_history]).reshape(1,-1)
        print(data)
        try:
            prediction = edumodel.predict(data)
            print(prediction)
            if prediction == 1:
                pred = "Loan_Approved"
                return render_template('temp.html')
            else:
                pred = "Not_Approved"
                return render_template('notapproved.html')
        # prediction=model.predict(pd.DataFrame(columns=['Age', 'Work_experience', 'Income', 'Per_Loan_amount', 'Per_loan_amount_term', 'Per_credit_history'],
        #                         data=np.array([age,experience, Income,Loan_amount, Loan_amount_term, credit_history]).reshape(1, 11)))
        except:

            print(Exception)

# @app.route('/predict',methods=['GET' , 'POST'])
# def predict():
#     if request.method == "POST":
#         name = request.form.get('name')

#         gender = request.form.get('gender')
#         marital_status = request.form.get('marital_status')
#         dependents = request.form.get('dependents')
#         education = request.form.get('education')
#         Self_Employed = request.form.get('Self_Employed')
#         ApplicantIncome = request.form.get('ApplicantIncome')
#         coapp_income = request.form.get('coapp_income')
#         loan_amount = request.form.get('loan_amount')
#         term = request.form.get('term')
#         credit_history = request.form.get('credit_history')
#         property_area = request.form.get('property_area')



#         prediction=model.predict(pd.DataFrame(columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'],
#                                 data=np.array([gender, marital_status, dependents, education, Self_Employed, ApplicantIncome, coapp_income, loan_amount, term, credit_history, property_area]).reshape(1, 11)))
        



if __name__=='__main__':
    app.run()







