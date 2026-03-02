# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Step 2: Load Dataset
data = pd.read_csv(r"C:\Users\jatin\OneDrive\Documents\GitHub\SCT_ML_01\house_price_prediction\data.csv")

# Step 3: Define Features (X) and Target (y)
X = data[['sqft', 'bedrooms', 'bathrooms']]
y = data['price']

# Step 4: Split Dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Create Linear Regression Model
model = LinearRegression()

# Step 6: Train the Model
model.fit(X_train, y_train)

# Step 7: Make Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate Model
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Step 9: Predict New House Price
new_house = np.array([[2000, 3, 2]])
predicted_price = model.predict(new_house)

print("Predicted Price for new house:", predicted_price[0])
plt.scatter(data['sqft'], data['price'])
plt.xlabel("Square Footage")
plt.ylabel("Price")
plt.title("Sqft vs Price")
plt.show()