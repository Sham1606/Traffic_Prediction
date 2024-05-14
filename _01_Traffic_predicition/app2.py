from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for Matplotlib

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set your secret key for flash messages

# Load the data set from an Excel file (adjust the file path as needed)
df = pd.read_excel("C:/Users/Madhan/Downloads/_01_Traffic_predicition/gang_data_traffic.xlsx")

# Rename columns
df = df.rename(columns={'Unnamed: 0': 'index', 'DateTime': 'datetime', 'Summary': 'weather', 'Vehicles': 'vehicles', 'Funx ': 'funx'})

# Convert 'datetime' column to datetime format
df['datetime'] = pd.to_datetime(df['datetime'])

# Extract the day of the week and hour from 'datetime' column
df['day_of_week'] = df['datetime'].dt.day_name()
df['hour'] = df['datetime'].dt.hour

# Calculate quantile thresholds for 'vehicles' column
low_threshold = df['vehicles'].quantile(0.33)
high_threshold = df['vehicles'].quantile(0.66)

# Define conditions and labels for the 'status' column
conditions = [
    df['vehicles'] <= low_threshold,  # Low status
    (df['vehicles'] > low_threshold) & (df['vehicles'] <= high_threshold),  # Medium status
    df['vehicles'] > high_threshold  # High status
]
labels = ['low', 'medium', 'high']

# Create 'status' column using conditions and labels
df['status'] = np.select(conditions, labels)

# Encode the 'day_of_week' column
le_day_of_week = LabelEncoder()
df['day_of_week'] = le_day_of_week.fit_transform(df['day_of_week'])

# Define features and target
features = ['day_of_week', 'hour', 'weather', 'funx']
target = 'status'

# Split data into training and testing sets
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Function to predict status for a given date
def predict_for_date(model, date, weather, funx):
    predictions = []
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    day_of_week_encoded = le_day_of_week.transform([date_obj.strftime('%A')])[0]
    
    for hour in range(24):
        # Create a row for prediction
        row = pd.DataFrame([[day_of_week_encoded, hour, weather, funx]], columns=['day_of_week', 'hour', 'weather', 'funx'])
        # Predict the status
        pred = model.predict(row)
        # Append the prediction to the list
        predictions.append((hour, pred[0]))
    
    return predictions

# Route to render the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the date, weather, and funx values from the form
        date = request.form['date']
        weather = float(request.form['weather'])
        funx = float(request.form['funx'])
        
        # Predict status for the given date
        predictions = predict_for_date(rf_classifier, date, weather, funx)
        
        # Create a plot for predictions
        hours, statuses = zip(*predictions)
        plt.figure(figsize=(12, 6))
        
        # Map status labels to numerical values for plotting
        status_mapping = {'low': 0, 'medium': 1, 'high': 2}
        numerical_statuses = [status_mapping[status] for status in statuses]

        plt.style.use('dark_background')
        plt.plot(hours, numerical_statuses, color='cyan')
        
        plt.xlabel('Hour of Day')
        plt.ylabel('Predicted Status')
        
        # Set custom y-axis tick labels
        status_labels = ['low', 'medium', 'high']
        plt.xticks(fontsize=12,color='white') 
        plt.yticks([0, 1, 2], status_labels)
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
    
        plt.title(f'Predicted Hourly Status for Date: {date}')

        # Save the plot to a buffer
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        # Encode the image in base64 for display in HTML
        plot_data = base64.b64encode(img.getvalue()).decode()

        # Calculate and pass output values for 'low', 'medium', and 'high' ranges
        low_value = statuses.count('low')
        medium_value = statuses.count('medium')
        high_value = statuses.count('high')

        return render_template('results.html', plot_data=plot_data, low_value=low_value, medium_value=medium_value, high_value=high_value)
    
    return render_template('index1.html')

# Route to render the results page (optional)
@app.route('/results', methods=['GET', 'POST'])
def results():
    # Get data for results page if needed
    # For simplicity, let's just render the same results.html template as in the home route
    return render_template('results.html', plot_data='', low_value=0, medium_value=0, high_value=0)

if __name__ == '__main__':
    app.run(debug=True)
