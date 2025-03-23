import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pickle  # Add pickle import

# Example dataset: each query paired with a manually assigned complexity score
data = {
    "query": [
        "Analyze sales trends by product category",
        "Quick summary of revenue",
        "Explain in detail the correlation between marketing spend and sales performance",
        "Compare profit margins across regions",
        "Forecast next quarter's earnings",
        "List all products sold",
        "Calculate total revenue",
        "Analyze customer demographics for the past year",
        "Predict inventory needs for the holiday season",
        "Detail breakdown of monthly expenses",
        "Show top 10 best selling items",
        "Describe the relationship between advertising and sales",
        "Summarize yearly performance metrics",
        "Evaluate the impact of pricing changes on demand",
        "Compare monthly revenue against expenses",
        "Examine the seasonal variations in sales data",
        "Investigate the influence of economic factors on sales",
        "Determine customer churn rate for the last quarter",
        "Assess the effectiveness of recent marketing campaigns",
        "Report on the regional distribution of sales",
        "Identify emerging market trends",
        "Break down the revenue by product line",
        "Calculate the average transaction value",
        "List the top performing sales regions",
        "Visualize sales data with time series charts",
        "Determine the correlation between customer reviews and sales",
        "Perform a detailed competitor analysis",
        "Analyze website traffic and conversion rates",
        "Report on customer acquisition costs",
        "Compare year-over-year growth rates",
        "Identify key drivers of customer satisfaction",
        "Examine the impact of promotions on sales",
        "Evaluate product return rates",
        "Determine average order size",
        "Report on the performance of new product launches",
        "Summarize social media engagement metrics",
        "Analyze the efficiency of the supply chain",
        "Predict future sales based on historical data",
        "Assess risk factors affecting revenue",
        "Detail the cost breakdown for production",
        "Visualize customer segmentation data",
        "Investigate seasonal promotions and their effectiveness",
        "Outline the process for inventory management",
        "Determine best performing marketing channels",
        "Analyze discount strategies and their impact on sales",
        "Review annual profit and loss statements",
        "Summarize key financial ratios",
        "Report on employee performance metrics",
        "Identify potential market disruptions",
        "Analyze multi-channel sales performance"
    ],
    "complexity": [
        8, 3, 10, 7, 9,
        2, 3, 8, 9, 6,
        4, 7, 5, 9, 7,
        8, 10, 7, 8, 6,
        8, 6, 3, 5, 7,
        9, 10, 8, 7, 8,
        9, 7, 6, 4, 8,
        5, 9, 9, 8, 7,
        8, 8, 6, 7, 9,
        7, 5, 6, 10, 8
    ]
}

df = pd.DataFrame(data)

# Split the data into training and test sets for evaluation
X_train, X_test, y_train, y_test = train_test_split(df['query'], df['complexity'], test_size=0.2, random_state=42)

# Create a regression pipeline: vectorizer extracts textual features, then Linear Regression learns the relationship
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('regressor', LinearRegression())
])

# Train the model
pipeline.fit(X_train, y_train)

# Save the model to a file using pickle
with open('query_complexity_model.pkl', 'wb') as model_file:
    pickle.dump(pipeline, model_file)
print("Model saved to 'query_complexity_model.pkl'")

# Example: How to load the model later
# with open('query_complexity_model.pkl', 'rb') as model_file:
#     loaded_model = pickle.load(model_file)
# Then use loaded_model.predict() as needed

# Predict on the test set and evaluate
y_pred = pipeline.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

# Calculate accuracy (within 1 point tolerance)
tolerance = 1.0
accuracy = np.mean(np.abs(y_test - y_pred) <= tolerance) * 100

print("Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Accuracy (within {tolerance} point): {accuracy:.2f}%")

# Example: Predict the complexity of new queries
test_queries = [
    "analyze sales trends by region",
    "list top customers",
    "explain the detailed impact of economic factors on quarterly sales performance"
]

print("\nPredictions for sample queries:")
for query in test_queries:
    predicted_complexity = pipeline.predict([query])[0]
    complexity_level = "Low" if predicted_complexity < 5 else "Medium" if predicted_complexity < 8 else "High"
    print(f"Query: '{query}'")
    print(f"  → Predicted complexity: {predicted_complexity:.2f} ({complexity_level})")
