import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

f = open("intents.json", mode='r')
data = json.load(f)

training_data = []
training_labels = []
labels = []
response = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        training_data.append(pattern)
        training_labels.append(intent['tag'])
    response.append(intent['response'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

num_of_labels = len(labels)

vocab_size = 100
oov_tokens = '<oov>'
max_len = 20

tokenizer = Tokenizer(vocab_size, oov_token=oov_tokens)
tokenizer.fit_on_texts(training_data)
word_index = tokenizer.word_index
print(word_index)

seq = tokenizer.texts_to_sequences(training_data)
padded_seq = pad_sequences(sequences=seq, padding='post', maxlen=20)

#this is input for our embedding model
print(padded_seq)

label_encoder = LabelEncoder()
label_encoder.fit(training_labels)
targets = label_encoder.transform(training_labels)

#this is output for the corresponding inputs for the model
print(targets)
print(label_encoder.classes_)

model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(num_of_labels, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(padded_seq, np.array(targets), epochs=100)

model.save("chatbot_new_model.h5")

print("model created")

with open('01_tokenizer.pkl', mode='wb') as t:
    pickle.dump(tokenizer, t, protocol=pickle.HIGHEST_PROTOCOL)

with open('lable_encoder.pkl', mode='wb') as encoder:
    pickle.dump(label_encoder, encoder, protocol=pickle.HIGHEST_PROTOCOL)

    






