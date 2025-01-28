import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd

# Load the trained model and encoder
model = joblib.load("model/decision_tree_model.joblib")
encoder = joblib.load("model/encoder.joblib")

# Function to predict fraud
def predict_fraud():
    try:
        # Get user input
        step = float(step_entry.get())
        type_ = type_entry.get()
        amount = float(amount_entry.get())
        oldbalanceOrg = float(oldbalanceOrg_entry.get())
        newbalanceOrig = float(newbalanceOrig_entry.get())
        oldbalanceDest = float(oldbalanceDest_entry.get())
        newbalanceDest = float(newbalanceDest_entry.get())

        # Encode transaction type
        if type_ not in encoder.classes_:
            raise ValueError(f"Invalid type. Choose one of {encoder.classes_}")

        type_encoded = encoder.transform([type_])[0]

        # Prepare input data
        input_data = pd.DataFrame({
            "step": [step],
            "type": [type_encoded],
            "amount": [amount],
            "oldbalanceOrg": [oldbalanceOrg],
            "newbalanceOrig": [newbalanceOrig],
            "oldbalanceDest": [oldbalanceDest],
            "newbalanceDest": [newbalanceDest]
        })

        # Predict using the model
        prediction = model.predict(input_data)

        # Show result in a message box
        result = "Fraudulent Transaction!" if prediction[0] == 1 else "Not Fraudulent Transaction!"
        messagebox.showinfo("Prediction Result", result)

    except ValueError as ve:
        messagebox.showerror("Input Error", str(ve))
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Create the main window
window = tk.Tk()
window.title("Fraud Detection System")

# Set window size and center it on the screen
window.geometry("800x500")
window.eval('tk::PlaceWindow . center')

# Apply background color for the window
window.configure(bg="#000000")

# Configure the root grid for centering the frame
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

# Create a frame to center all widgets
frame = tk.Frame(window, padx=20, pady=20, bg="#ADD8E6")
frame.grid(row=0, column=0, sticky="nsew")

# Configure the frame to center its content
frame.grid_rowconfigure(tuple(range(8)), weight=1)  # Adjust for the number of fields
frame.grid_columnconfigure(0, weight=1)
frame.grid_columnconfigure(1, weight=1)

# Create and place labels and input fields
fields = [
    ("Step", "step_entry"),
    ("Transaction Type", "type_entry"),
    ("Amount", "amount_entry"),
    ("Old Balance (Origin)", "oldbalanceOrg_entry"),
    ("New Balance (Origin)", "newbalanceOrig_entry"),
    ("Old Balance (Destination)", "oldbalanceDest_entry"),
    ("New Balance (Destination)", "newbalanceDest_entry")
]

entries = {}
row = 0

for label_text, entry_name in fields:
    label = tk.Label(frame, text=label_text, font=("Arial", 12, "bold"), bg="#f0f0f0")
    label.grid(row=row, column=0, padx=15, pady=15, sticky="e")  # Align labels to the right

    entry = tk.Entry(frame, font=("Arial", 12), width=40, bd=2, relief="solid")
    entry.grid(row=row, column=1, padx=15, pady=15)  # Place entry fields to the left

    entries[entry_name] = entry
    row += 1

# Assign entries to corresponding variables
step_entry = entries["step_entry"]
type_entry = entries["type_entry"]
amount_entry = entries["amount_entry"]
oldbalanceOrg_entry = entries["oldbalanceOrg_entry"]
newbalanceOrig_entry = entries["newbalanceOrig_entry"]
oldbalanceDest_entry = entries["oldbalanceDest_entry"]
newbalanceDest_entry = entries["newbalanceDest_entry"]

# Create the Predict button
predict_button = tk.Button(
    frame, text="Predict Fraud", font=("Arial", 14, "bold"),
    command=predict_fraud, bg="#4CAF50", fg="white",
    relief="raised", bd=3
)
predict_button.grid(row=row, column=0, columnspan=2, pady=20)

# Run the Tkinter event loop
window.mainloop()
