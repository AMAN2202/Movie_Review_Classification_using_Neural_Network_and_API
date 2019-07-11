from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from keras.models import Sequential
from keras import layers
df = pd.read_csv('dataset/imdb.txt', names=['sentence', 'label'], sep='\t')
sentences=df.sentence
y = df.label
vectorizer = CountVectorizer(min_df=0,ngram_range=(1,2) ,lowercase=False)
vectorizer.fit(sentences)
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)
vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)
input_dim = X_train.shape[1]  # Number of features
model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(10, activation='sigmoid'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train,epochs=100,verbose=False,validation_data=(X_test, y_test),batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.3f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.3f}".format(accuracy))
#save model
pickle.dump(vectorizer, open('vectorizer','wb'))
pickle.dump(model, open('model', 'wb'))
#predict using sample
sample=["Movie was awesome"]
test=vectorizer.transform(sample)
res=model.predict(test)

if(abs(res[0])>abs(1-res[0])):
    print("pos")
else:
    print("neg")

