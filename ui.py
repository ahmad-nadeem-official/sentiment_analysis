# app.py
import tkinter as tk
from tkinter import messagebox
from main import prediction  # Import the prediction function from main.py

def create_ui():
    """Create the user interface for sentiment analysis"""
    
    root = tk.Tk()
    root.title("Sentiment Analysis Tool")
    root.geometry("600x400")
    root.config(bg="#f4f4f4")
    
    # Header
    header_label = tk.Label(root, text="Sentiment Analysis Tool", font=("Helvetica", 20, "bold"), bg="#f4f4f4", fg="#333")
    header_label.pack(pady=20)
    
    # Input Field
    input_label = tk.Label(root, text="Enter your comment:", font=("Helvetica", 14), bg="#f4f4f4", fg="#333")
    input_label.pack(pady=5)
    
    comment_entry = tk.Entry(root, font=("Helvetica", 12), width=50, borderwidth=2, relief="solid")
    comment_entry.pack(pady=10)
    
    # Result Label
    result_label = tk.Label(root, text="Sentiment: ", font=("Helvetica", 14), bg="#f4f4f4", fg="#333")
    result_label.pack(pady=10)
    
    # Prediction Button
    def on_predict_click():
        """Handle button click, get user input, make prediction, and display result"""
        user_input = comment_entry.get()
        
        if not user_input:
            messagebox.showwarning("Input Error", "Please enter a comment for sentiment analysis.")
            return
        
        sentiment = prediction(user_input)  # Using the imported prediction function
        result_label.config(text=f"Sentiment: {sentiment}")
    
    predict_button = tk.Button(root, text="Predict Sentiment", font=("Helvetica", 14), bg="#007BFF", fg="white", command=on_predict_click)
    predict_button.pack(pady=20)
    
    # Footer
    footer_label = tk.Label(root, text="© 2025 Sentiment Analysis Tool", font=("Helvetica", 10), bg="#f4f4f4", fg="#333")
    footer_label.pack(side="bottom", pady=10)
    
    root.mainloop()

# Call the UI function to create the app
create_ui()
