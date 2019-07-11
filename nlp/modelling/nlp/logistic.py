from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

df = pd.read_csv('dataset/imdb.txt', names=['sentence', 'label'], sep='\t')
sentences=df.sentence
y = df.label
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)
vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)
model = LogisticRegression()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print("Accuracy:{}".format(score) )

#save model
pickle.dump(vectorizer, open('vectorizer','wb'))
pickle.dump(model, open('model', 'wb'))

#predict sample
sample=["Movie was awesome","Movie was bad"]
test=vectorizer.transform(sample)
res=model.predict(test)
for i in res:
    if(abs(i)>abs(1-i)):
        print("pos")
    else:
        print("neg")
