# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Prepare the Dataset (Fault_Issue examples only)
data = {
    'question': [
        "I have an issue",
        "Help me with a fault",
        "Fix my fault",
        "How do I report an issue?",
        "My broadband is not working",
        "I want to fix my PEO TV",
        "My internet stopped working",
        "How do I troubleshoot the fault?"
    ],
    'label': [
        "Fault_Issue",
        "Fault_Issue",
        "Fault_Issue",
        "Fault_Issue",
        "Fault_Issue",
        "Fault_Issue",
        "Fault_Issue",
        "Fault_Issue"
    ]
}

# Step 3: Convert the DataFrame
df = pd.DataFrame(data)

# Step 4: Split Data into Training and Testing Sets
X = df['question']  # Input text (questions)
y = df['label']     # Labels (Fault_Issue)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Convert Text to Numerical Data (Vectorization)
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Step 6: Train a Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Step 7: Evaluate the Model
y_pred = model.predict(X_test_vectorized)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: Allow User to Input Questions
print("\nEnter a question to predict the intent (type 'exit' to quit):")

# Keywords related to Fault_Issue
fault_keywords = ['issue', 'fault', 'problem', 'not working', 'troubleshoot', 'fix', 'repair']

while True:
    user_input = input("You: ")  # Take input from user
    if user_input.lower() == 'exit':  # Exit condition
        print("Exiting... Have a great day!")
        break
    
    # Check if the user input contains any fault-related keywords
    if any(keyword in user_input.lower() for keyword in fault_keywords):
        predicted_label = "Fault_Issue"
    else:
        predicted_label = "Unknown_Intent"
    
    # Output the result
    print(f"Bot: Predicted Intent --> {predicted_label}")
