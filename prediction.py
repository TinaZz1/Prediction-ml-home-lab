# library import

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt # data visualization 
import seaborn as sns
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()

df = pd.DataFrame(housing.data,  columns = housing.feature_names)
df['MedHouseValue'] = housing.target

housig_copy = df.copy() # dataset copy

print(df.info())

print(df.describe())
print(df.head())


# data visualization

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

sns.histplot(df['MedHouseValue'], bins=50, kde=True)
plt.title("Median Home Value Distribution")
plt.show()

# splitting data into training and test sets
from sklearn.model_selection import train_test_split

X = df.drop('MedHouseValue', axis=1)
y = df['MedHouseValue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# training
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# model evaluation
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
coefficients = coefficients.sort_values(by='Coefficient', ascending=False)
print(coefficients)

# Chart
coefficients.plot(kind='bar', figsize=(10, 6), title='The impact of variables on home value')
plt.ylabel('Regression coefficients')
plt.show()


#Which variables have the greatest impact on the price of a home and why?

"""The biggest influence on the price of a house is the average income of residents (MedInc) - the higher the income in the area, the higher the prices of the property,
which is quite intuitive. The features of the house itself also matter, e.g. the number of rooms (AveRooms) - larger and more spacious houses usually cost more. The age of the building (HouseAge) also affects the value,
but its significance may vary depending on the location.

Generally speaking, the most important factors are related to the standard of living and the standard of the property."""





