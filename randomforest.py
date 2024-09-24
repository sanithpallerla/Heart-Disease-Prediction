'''Importing required packages'''
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import pandas
import pickle
import matplotlib.pyplot as plt

'''Importing dataset using pandas'''
df = pandas.read_csv('heart.csv')
y = df['HeartDisease']
x = df.drop(columns= ['HeartDisease'])

'''Splitting data into train and test subsets'''
x_train, x_test, y_train, y_test = train_test_split(x,y,shuffle=True,test_size=0.25,random_state=42)

'''Using Standard scaler to standardize input features'''
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)

model = RandomForestClassifier(random_state = 42)

model_parameters = {'n_estimators' : list(range(10,300,10)),
                    'criterion' : ['gini', 'entropy'],
                    'min_samples_split' : list(range(2,20)),
                    'min_samples_leaf' : list(range(1,20)),
                    'max_features' :['auto','log2',None],
                    'max_leaf_nodes' :list(range(1,20)),
                    'bootstrap' :[True,False],
                    'oob_score' : [True,False]
                    }

'''Using Cross validation to find best combination of parameters'''
model = RandomizedSearchCV(model,
                       param_distributions= model_parameters,
                       cv = StratifiedKFold(n_splits = 10),
                       scoring = "accuracy",
                       n_jobs = -1, verbose = 2)
'''Training model'''
model.fit(x_train,y_train)
'''Performing prediction for data'''
y_pred = model.predict(x_test)
y_train_pred = model.predict(x_train)

'''Generating Reports'''
print()
print("Decision Tree Model Accuracy",accuracy_score(y_test,y_pred))
print()
print('train report : \n',classification_report(y_train, y_train_pred))
print('test_report : \n',classification_report(y_test,y_pred))
print("---------------------------------------------------------------------------")

'''Exporting model'''
filename = 'Random-Forest-model.pickle'
if accuracy_score(y_test,y_pred) > 90:
    pickle.dump(model, open(filename, 'wb'))

'''Generating confusion matrix and writing file on disk'''
cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test,y_pred), display_labels=[False, True])
cm_display.plot()
plt.title = "Random Forest"
plt.savefig('Random Forest.png')
plt.show()