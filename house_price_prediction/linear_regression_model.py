# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Step 2: Load Dataset
data = pd.read_csv(r"C:\Users\jatin\OneDrive\Documents\GitHub\SCT_ML_01\house_price_prediction\data.csv")

# Step 3: Define Features (X) and Target (y)
X = data[['sqft', 'bedrooms', 'bathrooms']]
y = data['price']

# Step 4: Create and Train Model
model = LinearRegression()
model.fit(X, y)

# Step 5: Take User Input
sqft = float(input("Enter Square Feet: "))
bedrooms = int(input("Enter Number of Bedrooms: "))
bathrooms = int(input("Enter Number of Bathrooms: "))

# Step 6: Convert Input into 2D Array
new_house = np.array([[sqft, bedrooms, bathrooms]])

# Step 7: Predict Price
predicted_price = model.predict(new_house)

# Step 8: Show Output
print("\nPredicted House Price:", round(predicted_price[0], 2))