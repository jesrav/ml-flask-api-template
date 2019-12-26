import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Append project root folder to path
basedir = Path(__file__).resolve().parents[1]
sys.path.append(str(basedir))

# Import model class
from app.ml_models.model_classes.iris_classifier import RandomForestClassifierModel

# Paths
data_path = Path('train_iris_classifier')
serialized_models_path = basedir / Path('app/ml_models/serialized_models/')

# Get iris data
iris = pd.read_csv(data_path / 'iris.csv')

# Set the target and design matrix
target = 'species'
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']  
y = iris[target]
X = iris[features] 

# Simple test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

# Train and evaluate a classifier
model = RandomForestClassifierModel(features=features)

model.fit(X_train,y_train)
preds = model.predict(X_test)

print(f'Model accuracy on test set {sum(preds == y_test)/len(y_test)}')

# Train model on all data
model.fit(X, y)

# Serialize model
model.save(fname=serialized_models_path / 'model.pickle')

with open('openapi_specification.yaml', 'w') as f:
    f.write(model.get_api_spec().to_yaml())
