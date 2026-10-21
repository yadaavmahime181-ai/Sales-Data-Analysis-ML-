import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# load dataset
cus = pd.read_csv("dataset/customer.csv")
order = pd.read_csv("dataset/orders.csv")
df = pd.merge(cus, order, on='CustomerID'  , how='inner')

# Feature and Target
X = df[['Product Category','Product Sub-Category','Amount']]
y = df['Profit']

# Categorical and Numerical Columns
cat_cols = ['Product Category','Product Sub-Category']
num_cols = ['Amount']

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat',OneHotEncoder(handle_unknown='ignore'),cat_cols),
        ('num','passthrough',num_cols)
    ]
)

# Model Pipeline
model = Pipeline([
    ('preprocessor',preprocessor),
    ('regressor',RandomForestRegressor(random_state=42))
])

# Train-Test Split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=11)

# Train
model.fit(X_train,y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("R^2 Score :",r2_score(y_test,y_pred))

# Model Save
import joblib
joblib.dump(model,"model/model.pkl")
print("Model Saved Successfully!")