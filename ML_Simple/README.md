## Machine Learning Predictions using Python & Scikit-Learn

## Linear Regression Example
basic code for ML predictions, or see [Jupyter Notebook](/ML_Simple/simple_ML_LinearRegression.ipynb) example.
```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data = pd.read_csv('AmesHousing.txt', sep = '\t')
X = data[['Gr Liv Area']]
y = data['SalePrice']
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)
lr = LinearRegression()
#train the model.
lr.fit(X_train, Y_train)
predictionsY = lr.predict(X_test)
```

## Logistic Regression Example
basic code for ML predictions, or see [Jupyter Notebook](/ML_Simple/simple_ML_LogisticRegression.ipynb) example.
```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
data = pd.read_csv('AmesHousing.txt', sep = '\t')
X = data[['Gr Liv Area']]
y = data['SalePrice']
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)
lr = LogisticRegression()
#train the model.
lr.fit(X_train, Y_train)
predictionsY = lr.predict(X_test)
```
