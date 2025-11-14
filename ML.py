# Test , Train , Split 

import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 

data = { 

    'study_hours': [2, 3, 4, 5, 6, 2, 8, 9, 10, 7], 

    'attendance': [60, 65, 70, 75, 80, 55, 85, 90, 95, 88], 

    'assignments_done': [3, 4, 5, 5, 6, 2, 7, 8, 9, 6], 

    'marks': [50, 55, 60, 65, 70, 45, 80, 85, 90, 75]   # dependent variable 

} 

df = pd.DataFrame(data)
print(df.head()) 

# Independent & dependent variables 
X = df[['study_hours' , 'attendance' , 'assignments_done']]
y = df['marks']

# Split into train-test sets 
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state=42)

# create model  
model = LinearRegression()
# Train model 
model.fit(X_train , y_train)
# predict 
y_predict = model.predict(X_test)
# find Error [method MSE]
MSE = mean_squared_error(y_test , y_predict) 
# check Accuracy
R2 = r2_score(y_test , y_predict)

print(f"MSE: {MSE:.2f}") 
print(f"R2 Score: {R2:.2f}") 
print(f"Model Score : {model.score(X_test, y_test):.2f}")

# Compare predictions 
print("\n Actual vs Predicted Marks:") 
Result = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict}) 
print(Result)



































  









