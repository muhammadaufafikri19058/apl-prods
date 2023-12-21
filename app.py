import tkinter as tk
from tkinter import ttk, scrolledtext
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import pandas as pd
import re

# Create Tkinter application
root = tk.Tk()
root.title("Spam Classifier")
root.geometry("1280x720")

# Load the dataset
file_var = tk.StringVar()

# Placeholder for X_train_numeric and y_train
X_train_numeric = None
y_train = None

def load_dataset():
    global X_train_numeric, y_train

    file_path = file_var.get()
    if file_path:
        try:
            df = pd.read_csv(file_path)
            update_df = update_senders(df)
            sender_table = create_sender_table(df)
            updated_df = update_numbers_in_messages(df, sender_table)

            # Split the data into features (X) and target (y)
            X_text = updated_df['message']
            X_numeric = updated_df[['word_total', 'spam_count', 'spam_auto', 'bigram_proportion']]
            y = updated_df['spamminess']

            # Split the data into training and testing sets
            X_train_text, _, X_train_numeric, _, y_train, _ = train_test_split(X_text, X_numeric, y, test_size=0.2, random_state=42)

            classify_and_display(updated_df)
        except Exception as e:
            result_text.config(state=tk.NORMAL)
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, f"Error loading dataset: {str(e)}")
            result_text.config(state=tk.DISABLED)

# Function to classify input and display the result
def classify_and_display(df):
    # Get input values from text boxes
    selected_input = input_var.get()
    
    if selected_input == 'Numeric':
        word_total = float(word_total_entry.get())
        spam_count = float(spam_count_entry.get())
        spam_auto = float(spam_auto_entry.get())
        bigram_proportion = float(bigram_proportion_entry.get())
        input_data = [[word_total, spam_count, spam_auto, bigram_proportion]]
    else:
        message = message_textbox.get("1.0", tk.END)
        vectorizer = CountVectorizer()
        X_message = vectorizer.fit_transform([message])
        input_data = X_message.toarray()

    # Choose the selected classifier
    selected_classifier = classifier_var.get()

    if selected_classifier == 'SVM':
        model = SVC()
    elif selected_classifier == 'Random Forest':
        model = RandomForestClassifier()
    elif selected_classifier == 'KNN':
        model = KNeighborsClassifier()
    elif selected_classifier == 'Naive Bayes':
        model = GaussianNB()
    elif selected_classifier == 'Stacking':
        model = StackingClassifier(estimators=estimators, stack_method='predict', passthrough=True)
    elif selected_classifier == 'Voting':
        voting_weights = {'KNN': 0.4, 'Naive Bayes': 0.1, 'Random Forest': 0.4, 'SVM': 0.1}
        model = VotingClassifier(estimators=estimators, voting='soft', weights=list(voting_weights.values()))

    # Train the selected model
    model.fit(X_train_numeric, y_train)

    # Make prediction
    predictions = model.predict(X_train_numeric)  # Assuming X_train_numeric is used for prediction

    # Display the result in scrolled text widget
    result_text.config(state=tk.NORMAL)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "Message detected as spam:\n")
    for index, row in df.iterrows():
        if predictions[index] == 1:  # Assuming 1 indicates spam
            result_text.insert(tk.END, f"{row['time_sent']}, {row['sender_asli']}, {row['message']}\n")
    result_text.config(state=tk.DISABLED)


# Create input text box for 'Message'
message_textbox = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=10)
message_textbox.grid(row=4, column=0, columnspan=2, padx=10, pady=5)

file_button = ttk.Button(root, text="Load Dataset", command=load_dataset)
file_button.grid(row=5, column=0, columnspan=2, pady=5)

# Create classifier selection dropdown
classifiers = ['SVM', 'Random Forest', 'KNN', 'Naive Bayes', 'Stacking', 'Voting']
classifier_var = tk.StringVar()
classifier_var.set(classifiers[0])
classifier_dropdown = ttk.Combobox(root, textvariable=classifier_var, values=classifiers)
classifier_dropdown.grid(row=6, column=0, columnspan=2, pady=5)

# Create classify button
classify_button = ttk.Button(root, text="Classify", command=lambda: classify_and_display(updated_df))
classify_button.grid(row=8, column=0, columnspan=2, pady=10)

# Create label to display result
result_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=15, state=tk.DISABLED)
result_text.grid(row=9, column=0, columnspan=2, pady=5)

# Run the Tkinter main loop
root.mainloop()
