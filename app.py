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
import pickle
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load
from tkinter import filedialog
import os
from nltk.util import ngrams
import nltk
nltk.download('punkt')

# Create Tkinter application
root = TkinterDnD.Tk()
root.title("WhatsApp Chat Viewer")
root.geometry("800x600")
global classify_count,now_count
classify_count=0
now_count=0
# Placeholder for DataFrame and models
df = None
svm_model, rf_model, nb_model, knn_model, stacking_model, voting_model = None, None, None, None, None, None

def load_and_display_dataset(file_path=None, chat_text=None):
    global df,now_count,classify_count
    now_count=classify_count
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
    display_dataframe(df, "Full pesan WA", f" ({len(df)} rows)")


    disable_load()
    # Enable the classification and group dropdowns
    # Show the dropdowns
    show_dropdowns()

def hide_dropdowns():
    global model_dropdown, group_dropdown, classify_button
    klasifikasi_frame.pack_forget()
    labelklasifikasi.pack_forget()
    model_dropdown.pack_forget()
    group_frame.pack_forget()
    labelgroup.pack_forget()
    group_dropdown.pack_forget()
    classify_button.pack_forget()
    back_button.pack_forget()

def show_dropdowns():
    global model_dropdown, group_dropdown, classify_button
    klasifikasi_frame.pack(pady=10)
    labelklasifikasi.pack(side=tk.LEFT,pady=10)
    model_dropdown.pack(side=tk.LEFT,pady=10)
    group_frame.pack(pady=10)
    labelgroup.pack(side=tk.LEFT,pady=10)
    group_dropdown.pack(side=tk.LEFT,pady=10)
    classify_button.pack(pady=10)
    back_button.pack(pady=10)

def disable_dropdowns():
    klasifikasi_frame.pack_forget()
    labelklasifikasi.pack_forget()
    model_dropdown.pack_forget()
    group_frame.pack_forget()
    labelgroup.pack_forget()
    group_dropdown.pack_forget()
    classify_button.pack_forget()
    back_button.pack_forget()
    
def disable_load():
    load_text_button.pack_forget()
    chat_text.pack_forget()
    button_frame.pack_forget()
    labelmasukteks.pack_forget()
    import_file_button.pack_forget()

def show_load():
    labelmasukteks.pack(pady=10)
    chat_text.pack(pady=10)
    button_frame.pack(pady=10)
    load_text_button.pack(side=tk.LEFT,padx=5)   
    import_file_button.pack(side=tk.LEFT,padx=5)


def on_group_selected(event):
    global selected_group
    selected_group = group_var.get()
    print(f"Selected Group: {selected_group}")

def destroy_dataframe():
    # Destroy the existing canvas if it exists
    for widget in frame.winfo_children():
        widget.destroy()

def display_dataframe(dataframe, title, subtitle):
    global classify_count,now_count
    print("udah",classify_count)
    # Destroy the existing canvas if it exists

    for widget in frame.winfo_children():
        widget_name = widget.winfo_name()
        print(widget_name)
        if isinstance(widget, tk.Canvas) and widget_name and widget_name.startswith('!canvas'):
            canvas_number_str = widget_name[len('!canvas'):]
            if canvas_number_str.isdigit():
                canvas_number = int(canvas_number_str)
                if canvas_number >= now_count+2:
                    widget.destroy()

    # Create a new Canvas widget inside the main frame
    canvas = tk.Canvas(frame)
    canvas.pack(side=tk.LEFT, fill="both", expand=True)

    # Create a Treeview widget inside the Canvas
    tree = ttk.Treeview(canvas, columns=['time_sent', 'sender', 'message'], show="headings", height=15)

    # Add column headings
    for col in ['time_sent', 'sender', 'message']:
        tree.heading(col, text=col)
        tree.column(col, width=100)

    # Add data to the treeview
    for i, row in dataframe.iterrows():
        tree.insert("", "end", values=list(row))

    # Add treeview to canvas
    tree.pack(fill="both", expand=True)

    # Add title label
    title_label = tk.Label(canvas, text=title + subtitle)
    title_label.pack()
    classify_count=classify_count+1


def classify_messages():
    global df, selected_group, selected_model,i
    if df is not None:
        selected_group = group_var.get()
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
            if word_total==0:
                word_total=1
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
        
        model_filename = f"{selected_model.lower().replace(' ', '')}_{selected_group.lower().replace(' ', '')}.pkl"
        with open(f"D:/apl prods/model/{model_filename}", 'rb') as file:
          model = pickle.load(file)
        X = df[['word_total', 'spam_count', 'spam_auto', 'bigram_proportion']]
        df['predicted_spam'] = model.predict(X)
        print(df)
        
        # Filter the DataFrame to display only rows where predicted_spam is 1
        df_predicted_spam = df[df['predicted_spam'] == '1']

        # Display the new DataFrame in the second table
        display_dataframe(df_predicted_spam, f"Spam berdasarkan klasifikasi {selected_model} pada {selected_group}", f" ({len(df_predicted_spam)} rows)")
        
    else:
        print("No dataset loaded. Please load a dataset first.")

def import_file():
    file_path = tk.filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        load_and_display_dataset(file_path=file_path)

def on_model_selected(event):
    global selected_model
    selected_model = model_var.get()
    print(f"Selected Model: {selected_model}")

def ganti_data():
    global df
    destroy_dataframe()
    disable_dropdowns()
    show_load()
    df = None

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

# Create a frame for the buttons
button_frame = ttk.Frame(root)
button_frame.pack(pady=10)  # Adjust pady as needed

# Create a button to load and display the dataset from text input
load_text_button = ttk.Button(button_frame, text="Load Text", command=lambda: load_and_display_dataset(chat_text=chat_text))
load_text_button.pack(side=tk.LEFT, padx=5)  # Place the button next to the first button within the frame

# Create a button to import a text file
import_file_button = ttk.Button(button_frame, text="Import File", command=import_file)
import_file_button.pack(side=tk.LEFT, padx=5)  # Place the button on the left within the frame

# Create a frame for the buttons
klasifikasi_frame = ttk.Frame(root)
klasifikasi_frame.pack_forget  # Adjust pady as needed

# Label untuk teks 'Pilih klasifikasi'
label_text = tk.StringVar()
label_text.set("Pilih Klasifikasi : ")
labelklasifikasi = tk.Label(klasifikasi_frame, textvariable=label_text)
labelklasifikasi.pack_forget()  # Sesuaikan sesuai kebutuhan

# Dropdown for selecting the model
models = ['SVM', 'Random Forest', 'Naive Bayes', 'KNN', 'Stacking', 'Voting']
model_var = tk.StringVar(klasifikasi_frame)
model_var.set(models[0])  # set the default value
model_dropdown = ttk.Combobox(klasifikasi_frame, textvariable=model_var, values=models)
model_dropdown.bind("<<ComboboxSelected>>", on_model_selected)
model_dropdown.pack_forget()  # initially hide the dropdown


# Create a frame for the buttons
group_frame = ttk.Frame(root)
group_frame.pack_forget  # Adjust pady as needed

# Label untuk teks 'Pilih Grup'
label_text = tk.StringVar()
label_text.set("Pilih Grup : ")
labelgroup = tk.Label(group_frame, textvariable=label_text)
labelgroup.pack_forget()  # Sesuaikan sesuai kebutuhan

# Dropdown for selecting the group
groups = ['Group 1', 'Group 2', 'Group 3', 'Group 4']
group_var = tk.StringVar(group_frame)
group_var.set(groups[0])  # set the default value
group_dropdown = ttk.Combobox(group_frame, textvariable=group_var, values=groups)
group_dropdown.bind("<<ComboboxSelected>>", on_group_selected)
group_dropdown.pack_forget()  # initially hide the dropdown

# Create a button for classification
classify_button = ttk.Button(root, text="Classify", command=lambda: classify_messages())
classify_button.pack_forget()

# Create a button for classification
back_button = ttk.Button(root, text="ganti data", command=lambda: ganti_data())
back_button.pack_forget()

# Create a frame to hold the Treeview widget
frame = ttk.Frame(root)
frame.pack(expand=True, fill="both", pady=10)  # Added vertical padding

# Run the Tkinter main loop
root.mainloop()