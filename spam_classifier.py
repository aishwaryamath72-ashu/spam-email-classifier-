import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Sample dataset
data = {
    'message': [
        'Win money now', 'Free prize claim now', 'Earn cash fast',
        'Hello how are you', 'Let us meet tomorrow', 'Good morning',
        'Exclusive offer just for you', 'Call me later',
        'Congratulations you won lottery', 'Are you coming today',
        'Limited time offer', 'See you soon'
    ],
    'label': [1,1,1,0,0,0,1,0,1,0,1,0]
}   

df = pd.DataFrame(data)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MultinomialNB()
model.fit(X_train, y_train)

msg = input("Enter a message: ")
msg_transformed = vectorizer.transform([msg])

prediction = model.predict(msg_transformed)

if prediction[0] == 1:
    print("Spam Message ❌")
else:
    print("Not Spam ✅")

# ✅ Always calculate accuracy
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")