import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the data
file_path = 'DEPREFINAL.csv'
df = pd.read_csv(file_path)

# Inspect the columns
print(df.columns)

# Select relevant columns for modeling based on the actual column names
features = ['Fasting Plasma Glucose', 'BMI', 'SystolicBP', 'DiastolicBP', 'A1C (percentage)']
target = 'DiabetesType'  # Assuming 'class' is the target column for diabetes type

# Clean the data by dropping rows with missing values in the relevant columns
df_clean = df[features + [target]].dropna()

# Label encode the target variable
label_encoder = LabelEncoder()
df_clean[target] = label_encoder.fit_transform(df_clean[target])

# Split the data into training and testing sets
X = df_clean[features]
y = df_clean[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train an SVM model
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred)
print(svm_accuracy)

# Class labels for the model (Type 1, Type 2, etc.)
class_labels = label_encoder.classes_

# Function to predict diabetes type and display results using the trained SVM model
def predict_diabetes():
    try:
        # Retrieve user input values
        fasting_glucose = float(entry_fasting_glucose.get())
        bmi = float(entry_bmi.get())
        systolic_bp = float(entry_systolic_bp.get())
        diastolic_bp = float(entry_diastolic_bp.get())
        a1c = float(entry_a1c.get())
        
        # Create the input array for prediction
        input_data = [[fasting_glucose, bmi, systolic_bp, diastolic_bp, a1c]]
        
        # Check for prediabetes case based on HbA1c levels
        if 5.7 <= a1c < 6.5:
            diabetes_type = "Prediabetes"
            risks = "Increased risk of developing Type 2 diabetes."
            procedures = "Lifestyle changes (diet, exercise), Regular monitoring, Weight management."
        else:
            # Make the prediction using the SVM model
            prediction = svm_model.predict(input_data)[0]
            diabetes_type = class_labels[prediction]
            
            # Define risks and procedures based on diabetes type
            if diabetes_type == "Type 1":
                risks = "Diabetic ketoacidosis, Hypoglycemia, Eye damage, Kidney damage."
                procedures = "Insulin therapy, Regular blood sugar monitoring, Healthy diet and exercise."
            elif diabetes_type == "Type 2":
                risks = "Cardiovascular disease, Neuropathy, Nephropathy, Retinopathy."
                procedures = "Lifestyle changes (diet, exercise), Oral medications, Insulin therapy (if needed)."
            else:
                diabetes_type = "Non-diabetic"
                risks = "Low risk of diabetes."
                procedures = "Maintain a healthy lifestyle."

        # Display the results in a message box
        messagebox.showinfo("Prediction Results", 
                            f"Predicted Diabetes Type: {diabetes_type}\n"
                            f"Potential Risks: {risks}\n"
                            f"Recommended Procedures: {procedures}\n"
                            f"Model Accuracy: {svm_accuracy * 100:.2f}%")
        
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values.")

# Function to visualize the comparison of diabetes types with respect to other features
def visualize_data():
    # Count the occurrence of each diabetes type
    diabetes_type_counts = df_clean[target].value_counts()
    diabetes_type_labels = label_encoder.inverse_transform(diabetes_type_counts.index)
    
    # Create a pie chart to show the distribution of diabetes types
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].pie(diabetes_type_counts, labels=diabetes_type_labels, autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'])
    axs[0].set_title('Distribution of Diabetes Types')
    
    # Create a bar chart to compare average features for different diabetes types
    feature_means = df_clean.groupby(target)[features].mean()
    feature_means.index = label_encoder.inverse_transform(feature_means.index)
    
    feature_means.plot(kind='bar', ax=axs[1], colormap='viridis')
    axs[1].set_title('Feature Comparison by Diabetes Type')
    axs[1].set_ylabel('Average Values')
    plt.tight_layout()

    # Display the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=6, column=0, columnspan=2)

# Create the Tkinter window
root = tk.Tk()
root.title("Diabetes Prediction")

# Create and place labels and entry boxes for input
tk.Label(root, text="Fasting Plasma Glucose").grid(row=0, column=0, padx=10, pady=5)
entry_fasting_glucose = tk.Entry(root)
entry_fasting_glucose.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="BMI").grid(row=1, column=0, padx=10, pady=5)
entry_bmi = tk.Entry(root)
entry_bmi.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="Systolic BP").grid(row=2, column=0, padx=10, pady=5)
entry_systolic_bp = tk.Entry(root)
entry_systolic_bp.grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="Diastolic BP").grid(row=3, column=0, padx=10, pady=5)
entry_diastolic_bp = tk.Entry(root)
entry_diastolic_bp.grid(row=3, column=1, padx=10, pady=5)

tk.Label(root, text="A1C (percentage)").grid(row=4, column=0, padx=10, pady=5)
entry_a1c = tk.Entry(root)
entry_a1c.grid(row=4, column=1, padx=10, pady=5)

# Add a button to trigger the prediction
predict_button = tk.Button(root, text="Predict Diabetes", command=predict_diabetes)
predict_button.grid(row=5, column=0, columnspan=2, pady=10)

# Add a button to trigger the visualization
visualize_button = tk.Button(root, text="Visualize Data", command=visualize_data)
visualize_button.grid(row=5, column=1, pady=10)

# Start the Tkinter event loop
root.mainloop()
