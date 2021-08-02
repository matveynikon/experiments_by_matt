from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import pandas as pd
iris = datasets.load_iris()
features=pd.DataFrame(iris['data'])
target=iris['target']
model=LogisticRegression(max_iter=1000)
model.fit(features,target)

import pickle
pickle.dump(model, open('model_iris', 'wb'))
