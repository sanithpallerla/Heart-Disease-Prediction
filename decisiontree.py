from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
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

model = DecisionTreeClassifier(random_state = 42)

model_parameters = {'criterion': ['gini', 'entropy'],
                    'splitter' :['best','random'],
                    'max_depth' : range(1,20),
                    'min_samples_split' : range(2,50),
                    'min_samples_leaf' : range(1,6),
                    'max_features' : ['auto', 'sqrt', 'log2', None]
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
filename = 'Decision-Tree-model.pickle'
if accuracy_score(y_test,y_pred) > 90:
    pickle.dump(model, open(filename, 'wb'))

'''Generating confusion matrix and writing file on disk'''
cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test,y_pred), display_labels=[False, True])
cm_display.plot()
plt.title = "Decision Tree"
plt.savefig('DecisionTree.png')
plt.show()