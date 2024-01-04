import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load
import os
from nltk.util import ngrams
import nltk
nltk.download('punkt')

# Create Tkinter application
root = TkinterDnD.Tk()
root.title("WhatsApp Chat Viewer")
root.geometry("800x600")

# Placeholder for DataFrame and models
df = None
svm_model, rf_model, nb_model, knn_model, stacking_model, voting_model = None, None, None, None, None, None

# Placeholder for selected group and CSV file paths
selected_group = None
group_csv_mapping = {
    'Group 1': 'group1.csv',
    'Group 2': 'group2.csv',
    'Group 3': 'group3.csv',
    'Group 4': 'group4.csv',
}

def load_and_display_dataset(file_path=None, chat_text=None):
    global df

    if file_path:
        try:
            with open(file_path, encoding='utf-8') as f:
                WhatsApp_chat_data = f.read().split('\n')
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return
    elif chat_text:
        WhatsApp_chat_data = chat_text.get("1.0", tk.END).split('\n')
    else:
        print("No input provided.")
        return

    new_chat = []
    regex_pattern = r"\d{1,2}/\d{1,2}/\d{2}, \d{2}:\d{2} - "

    for chat in WhatsApp_chat_data:
        if len(chat) > 0:
            if re.search(regex_pattern, chat):
                new_chat.append(chat)
            elif new_chat:
                new_chat[-1] += '\n' + chat
        else:
            continue

    time_sent = []
    sender = []
    message = []

    for chat in new_chat:
        if len(chat) > 0:
            if chat.count(":") == 1:
                first_dash = chat.find(" - ")
                time_sent.append(chat[:first_dash])
                sender.append("Admin")
                message.append(chat[first_dash+3:])
            elif chat.count(":") > 1:
                first_doublecolon = chat.find(":")
                second_doublecolon = chat.find(":", first_doublecolon+1)
                timesent_sender = chat[:second_doublecolon]
                time_sent_part, sender_part = timesent_sender.split(" - ")
                time_sent.append(time_sent_part)
                sender.append(sender_part)
                message.append(chat[second_doublecolon+1:])
            else:
                message[-1] += '\n' + chat
        else:
            continue

    data = {'time_sent': time_sent, 'sender': sender, 'message': message}
    
    if df is not None:
        # Update the existing DataFrame
        df = pd.concat([df, pd.DataFrame(data)])
    else:
        # Create a new DataFrame
        df = pd.DataFrame(data)

    # Display the DataFrame in a table
    display_dataframe(df)


    disable_load()
    # Enable the classification and group dropdowns
    # Show the dropdowns
    show_dropdowns()

def hide_dropdowns():
    global model_dropdown, group_dropdown
    model_dropdown.pack_forget()
    group_dropdown.pack_forget()

def show_dropdowns():
    labelhasil.pack(pady=10)
    global model_dropdown, group_dropdown, classify_button
    labelklasifikasi.pack(pady=10)
    model_dropdown.pack(pady=10)
    labelgroup.pack(pady=10)
    group_dropdown.pack(pady=10)
    classify_button.pack(pady=10)


def disable_load():
    load_text_button.pack_forget()
    chat_text.pack_forget()
    labelmasukteks.pack_forget()
    


def on_group_selected(event):
    global selected_group
    selected_group = group_var.get()
    print(f"Selected Group: {selected_group}")


def display_dataframe(dataframe):
    # Destroy existing treeview widget (if any)
    for widget in frame.winfo_children():
        widget.destroy()

    # Create a Canvas widget
    canvas = tk.Canvas(frame)
    canvas.pack(side="left", fill="both", expand=True)

    # Create Treeview widget inside the Canvas
    tree = ttk.Treeview(canvas, columns=['time_sent', 'sender', 'message'], show="headings", height=20)

    # Add column headings
    for col in ['time_sent', 'sender', 'message']:
        tree.heading(col, text=col)
        tree.column(col, width=100)  # You can adjust the width as needed

    # Add data to the treeview
    for i, row in dataframe.iterrows():
        tree.insert("", "end", values=list(row))

    # Add treeview to canvas
    tree.pack(side="left", fill="both", expand=True)

    # Add scrollbar to the canvas
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Configure the canvas scrolling
    canvas.create_window((0, 0), window=tree, anchor="nw")
    tree.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))
def train_model(selected_group, selected_model):
    global df, svm_model, rf_model, nb_model, knn_model, stacking_model, voting_model

    if df is not None:
        # Filter DataFrame based on selected group
        group_df= read_csv()


        # Add a new column 'spamminess' based on your criteria (you need to define this)
        # For example, you can set it to 1 if 'spam_auto' is 1, otherwise 0
        group_df['spamminess'] = group_df['spam_auto']

        # Split data into features (X) and target (y)
        X = group_df[['word_total', 'spam_count', 'spam_auto', 'bigram_proportion']]
        y = group_df['spamminess']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize the selected model
        if selected_model == 'SVM':
            model = SVC()
        elif selected_model == 'Random Forest':
            model = RandomForestClassifier()
        elif selected_model == 'Naive Bayes':
            model = MultinomialNB()
        elif selected_model == 'KNN':
            model = KNeighborsClassifier()
        # Add other models as needed

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # Save the trained model
        if selected_model == 'SVM':
            svm_model = model
        elif selected_model == 'Random Forest':
            rf_model = model
        elif selected_model == 'Naive Bayes':
            nb_model = model
        elif selected_model == 'KNN':
            knn_model = model
        # Add other models as needed

        print(f"Model: {selected_model}")
        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{report}")

    else:
        print("No dataset loaded. Please load a dataset first.")


def classify_messages():
    global df
    global selected_group 
    global selected_model

    if df is not None:
        selected_model = model_var.get()  # Declare selected_model as a global variable

        df['word_total'] = 0
        df['spam_count'] = 0
        df['bigram_proportion'] = 0.0
        df['spam_auto'] = 0
        df['spam_bigrams'] = ''

        for index, row in df.iterrows():
            text_message = row['message']  # Update column name to match your DataFrame

            # Calculate word_total
            tokens = nltk.word_tokenize(text_message)
            word_total = len(tokens)
            df.at[index, 'word_total'] = word_total

            # Calculate spam_count
            spam_count = 0
            for token in tokens:
                bi_grams = list(ngrams(token, 2))
                bi_gram_freq = nltk.FreqDist(bi_grams)
                for count in bi_gram_freq.values():
                    if count > 1:
                        spam_count += 1
                        break
            df.at[index, 'spam_count'] = spam_count

            # Calculate bigram_proportion
            bigram_proportion = spam_count / word_total * 100
            df.at[index, 'bigram_proportion'] = bigram_proportion

            # Determine spam_auto
            if bigram_proportion >= 50:
                df.at[index, 'spam_auto'] = 1
            else:
                df.at[index, 'spam_auto'] = 0

            # Calculate spam_bigrams
            spam_bigrams = set()
            for token in tokens:
                bi_grams = list(ngrams(token, 2))
                bi_gram_freq = nltk.FreqDist(bi_grams)
                for bigram, count in bi_gram_freq.items():
                    if count > 1:
                        spam_bigrams.add(''.join(bigram))
            df.at[index, 'spam_bigrams'] = ', '.join(spam_bigrams)
        print(df)
        train_model(selected_group, selected_model)

    else:
        print("No dataset loaded. Please load a dataset first.")



def on_model_selected(event):
    global selected_model
    selected_model = model_var.get()
    print(f"Selected Model: {selected_model}")

# Label untuk teks 'Pilih Grup'
label_text = tk.StringVar()
label_text.set("hasil yang telah di jadikan tabel")
labelhasil = tk.Label(root, textvariable=label_text)
labelhasil.pack_forget()  # Sesuaikan sesuai kebutuhan

# Label untuk teks 'Pilih Grup'
label_text = tk.StringVar()
label_text.set("Masukan Teks")
labelmasukteks = tk.Label(root, textvariable=label_text)
labelmasukteks.pack(pady=10)  # Sesuaikan sesuai kebutuhan

# Enable drag-and-drop functionality
root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', lambda event: load_and_display_dataset(file_path=event.data))

# Create a Text widget for direct input
chat_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=40, height=10)
chat_text.pack(pady=10)

# Create a button to load and display the dataset from text input
load_text_button = ttk.Button(root, text="Load Text", command=lambda: load_and_display_dataset(chat_text=chat_text))
load_text_button.pack(pady=10)

# Label untuk teks 'Pilih Grup'
label_text = tk.StringVar()
label_text.set("Pilih Klasifikasi")
labelklasifikasi = tk.Label(root, textvariable=label_text)
labelklasifikasi.pack_forget()  # Sesuaikan sesuai kebutuhan

# Dropdown for selecting the model
models = ['SVM', 'Random Forest', 'Naive Bayes', 'KNN', 'Stacking', 'Voting']
model_var = tk.StringVar(root)
model_var.set(models[0])  # set the default value
model_dropdown = ttk.Combobox(root, textvariable=model_var, values=models)
model_dropdown.bind("<<ComboboxSelected>>", on_model_selected)
model_dropdown.pack_forget()  # initially hide the dropdown


# Label untuk teks 'Pilih Grup'
label_text = tk.StringVar()
label_text.set("Pilih Grup")
labelgroup = tk.Label(root, textvariable=label_text)
labelgroup.pack_forget()  # Sesuaikan sesuai kebutuhan
# Dropdown for selecting the group
groups = ['Group 1', 'Group 2', 'Group 3', 'Group 4']
group_var = tk.StringVar(root)
group_var.set(groups[0])  # set the default value
group_dropdown = ttk.Combobox(root, textvariable=group_var, values=groups)
group_dropdown.bind("<<ComboboxSelected>>", on_group_selected)
group_dropdown.pack_forget()  # initially hide the dropdown

# Create a button for classification
classify_button = ttk.Button(root, text="Classify", command=lambda: classify_messages())
classify_button.pack_forget()
# Create a frame to hold the Treeview widget
frame = ttk.Frame(root)
frame.pack(expand=True, fill="both", pady=10)  # Added vertical padding

# Run the Tkinter main loop
root.mainloop()