from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pdd


data_h = pdd.read_csv('kc_house_data.csv')

Features1 = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']
target = 'price'
X1 = data_h[features1]
y1 = data_h[target]

X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

score = model.score(X_test, y_test)
print("Model R^2 Score:", score)
new_house = pdd.DataFrame({'bedrooms': [2], 'bathrooms': [2.5], 'sqft_living': [600], 'sqft_lot': [600], 'floors': [2], 'zipcode': [98008]})
predicted_price = model.predict(new_house)
print("Predicted Price:", predicted_price[0])
