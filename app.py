import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.title("Student Marks Prediction")

df = pd.read_csv("Student_Marks.csv")

X = df[['time_study']]
y = df['Marks']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.subheader("Model Performance")
st.write("MAE:", mean_absolute_error(y_test, y_pred))
st.write("R² Score:", r2_score(y_test, y_pred))

st.subheader("Predict Marks")
hours = st.slider("Study Hours", 0.0, 12.0, 5.0, 0.25)

if st.button("Predict"):
    pred = model.predict([[hours]])
    st.success(f"Predicted Marks: {pred[0]:.2f}")

fig, ax = plt.subplots()
ax.scatter(X, y)
ax.plot(X, model.predict(X))
ax.set_xlabel("Study Hours")
ax.set_ylabel("Marks")
st.pyplot(fig)
