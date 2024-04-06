import pandas as pd
import seaborn as sns
import numpy as np
import pickle
from sklearn import linear_model
from sklearn.model_selection import train_test_split

df = pd.read_csv('INSERT CSV HERE')
print(df.head())

df = df.drop("Unnamed: 0", axis=1)

sns.lmplot(x='BLAH', y='BLAH', data=df)
sns.lmplot(x='BLAH', y='BLAH', data=df)

x_df = df.drop('COVID', axis=1)
y_df = df['COVID']

X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3)

model = linear_model.LinearRegression()

model.fit(X_train, y_train)
print(model.score(X_train, y_train))

prediction_test = model.predict(X_test)
print(y_test, prediction_test)
print("Mean sq. error btwn y_test and predicted =", np.mean(prediction_test-y_test)**2)

pickle.dump(model, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))