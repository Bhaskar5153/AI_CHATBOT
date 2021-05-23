import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
from sklearn.preprocessing import LabelEncoder

with open("01_intents.json", mode="r") as f:
    data = json.load(f)

training_set = []
training_labels = []
labels = []
response = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        training_set.append(pattern)
        training_labels.append(intent['tag'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

    response.append(intent['response'])


vocab_size = 1000
embedding_dim = 10
max_len = 20
oov_token = "<oov>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_set)
word_index = tokenizer.word_index
seq = tokenizer.texts_to_sequences(training_set)
padded_seq = pad_sequences(seq, maxlen=max_len, padding='post')

label_encoder = LabelEncoder()
label_encoder.fit(training_labels)
target = label_encoder.transform(training_labels)

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(len(labels), activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

hist = model.fit(x=padded_seq, y=np.array(target), epochs=300)
model.save('04_chatbot_model.h5')

with open("04_tokenizer.pkl", mode='wb') as t:
    pickle.dump(tokenizer, t, protocol=pickle.HIGHEST_PROTOCOL)

with open("04_encoder.pkl", mode="wb") as e:
    pickle.dump(label_encoder, e, protocol=pickle.HIGHEST_PROTOCOL)

print("model printed")




