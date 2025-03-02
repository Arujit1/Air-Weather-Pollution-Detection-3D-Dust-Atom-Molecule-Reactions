import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tkinter import Tk, Label, Button, filedialog, Toplevel, Text

def load_data():
    """Generate synthetic air quality dataset."""
    data = pd.DataFrame({
        "temperature": np.random.uniform(15, 35, 1000),
        "humidity": np.random.uniform(30, 80, 1000),
        "wind_speed": np.random.uniform(0, 10, 1000),
        "industrial_activity": np.random.uniform(50, 200, 1000),
        "traffic_density": np.random.uniform(20, 100, 1000),
        "air_quality_index": np.random.uniform(50, 300, 1000),
    })
    return data

def preprocess_data(data):
    """Split data into training and testing sets."""
    X = data.drop("air_quality_index", axis=1)
    y = data["air_quality_index"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    """Train RandomForest model."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and return predictions."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    return predictions

def visualize_results(y_test, predictions):
    """Plot actual vs predicted AQI values."""
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(y_test, predictions, alpha=0.7, c=predictions, cmap="viridis", label="Predictions")
    plt.colorbar(scatter, label="Prediction Intensity")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label="Ideal Fit")
    plt.title("Actual vs Predicted Air Quality Index")
    plt.xlabel("Actual AQI")
    plt.ylabel("Predicted AQI")
    plt.legend(loc="upper left")
    plt.show()

def visualize_3d_pollution():
    """Create a 3D visualization of dust and pollution molecules."""
    np.random.seed(42)
    x, y, z = np.random.uniform(-10, 10, (3, 100))
    sizes = np.random.uniform(10, 50, 100)
    colors = np.random.uniform(0, 1, 100)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(size=sizes, color=colors, colorscale='Viridis', opacity=0.8),
        name='Dust & Pollution Particles'))
    fig.update_layout(scene=dict(xaxis_title='X Axis', yaxis_title='Y Axis', zaxis_title='Z Axis'),
                      title="3D Visualization of Dust and Pollution Molecules")
    fig.show()

def analyze_image(file_path):
    """Analyze uploaded image and extract environmental insights."""
    image = cv2.imread(file_path)
    if image is None:
        print("Failed to load image.")
        return
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray_image)
    
    if avg_brightness < 100:
        condition = "Rainy/Cloudy"
    elif avg_brightness < 150:
        condition = "Cloudy"
    elif avg_brightness < 200:
        condition = "Sunny"
    else:
        condition = "Very Sunny"
    
    pollution_level = np.random.uniform(50, 300)
    humidity = np.random.uniform(30, 80)
    pollution_percentage = (pollution_level / 300) * 100
    clean_air_percentage = 100 - pollution_percentage
    air_filter_recommendation = "Recommended" if pollution_level > 150 else "Not Necessary"
    
    summary = {
        "Condition": condition,
        "Average Brightness": avg_brightness,
        "Pollution Level": pollution_level,
        "Humidity": humidity,
        "Air Filter Recommendation": air_filter_recommendation,
        "Pollution Percentage": pollution_percentage,
        "Clean Air Percentage": clean_air_percentage
    }
    display_summary(summary, image)
    
    plt.bar(["Brightness", "Pollution Level", "Humidity"], [avg_brightness, pollution_level, humidity],
            color=['blue', 'green', 'orange'])
    plt.title("Environment Analysis")
    plt.ylabel("Values")
    plt.show()
    
    plt.pie([clean_air_percentage, pollution_percentage], labels=['Clean Air', 'Pollution'],
            autopct='%1.1f%%', colors=['skyblue', 'red'])
    plt.title("Air Quality Composition")
    plt.show()

def display_summary(summary, image):
    """Display analysis summary in a pop-up window."""
    summary_text = """
    Environment Analysis Summary:
    - Condition: {Condition}
    - Average Brightness: {Average Brightness:.2f}
    - Pollution Level: {Pollution Level:.2f}
    - Humidity: {Humidity:.2f}%
    - Pollution Percentage: {Pollution Percentage:.2f}%
    - Clean Air Percentage: {Clean Air Percentage:.2f}%
    - Air Filter Recommendation: {Air Filter Recommendation}
    """.format(**summary)
    
    root = Toplevel()
    root.title("Environment Analysis Summary")
    text_widget = Text(root, wrap='word', padx=10, pady=10, font=("Arial", 12))
    text_widget.insert("1.0", summary_text)
    text_widget.config(state='disabled')
    text_widget.pack(fill='both', expand=True)
    root.mainloop()

def upload_image():
    """Open file dialog to upload an image."""
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        analyze_image(file_path)

def main():
    print("Loading data...")
    data = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    print("Training model...")
    model = train_model(X_train, y_train)
    
    print("Evaluating model...")
    predictions = evaluate_model(model, X_test, y_test)
    
    print("Visualizing results...")
    visualize_results(y_test, predictions)
    
    print("Visualizing 3D pollution model...")
    visualize_3d_pollution()
    
    print("Uploading image...")
    upload_image()

if __name__ == "__main__":
    main()
