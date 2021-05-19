import json
import pickle
import random
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


f = open("intents.json", mode="r")
data = json.load(f)

#print(data)


def chat():
    model = load_model("chatbot_new_model.h5")

    with open("01_tokenizer.pkl", mode="rb") as t:
        tokenizer = pickle.load(t)

    with open("lable_encoder.pkl", mode="rb") as e:
        encoder = pickle.load(e)
        #print(encoder.classes_)

    BOT = True
    while BOT:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            BOT = False

        prediction = model.predict(pad_sequences(tokenizer.texts_to_sequences([user_input]),
                                                 truncating='post', maxlen=20))
        prediction = np.argmax(prediction)

        tag = encoder.inverse_transform([prediction])


        for i in data['intents']:
            if i['tag'] == tag:
                print("the tag is ", tag)
                print("BOT: ", np.random.choice(i['response']))


chat()
