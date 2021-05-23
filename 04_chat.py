import tkinter
from tkinter import *
import json
import colorama
from colorama import Fore, Style, Back
import pickle
import numpy as np
from tensorflow import keras

wn = tkinter.Tk()
wn.geometry('500x500')
wn.resizable(width=False, height=False)
wn.title('AutoBot')

chat_log = Text(wn, bd=3, font=('Cambria', 12))
chat_log.config(state=DISABLED)
chat_log.place(x=0, y=0, width=490, height=400)

chat_area = Text(wn, bd=3, font=('Cambria', 12))
chat_area.place(x=0, y=400, width=400, height=100)

with open("01_intents.json") as f:
    data = json.load(f)

model = keras.models.load_model('04_chatbot_model.h5')

with open("04_tokenizer.pkl", mode='rb') as t:
    tokenizer = pickle.load(t)

with open("04_encoder.pkl", mode='rb') as e:
    encoder = pickle.load(e)


def send():
    user_input = chat_area.get(index1='1.0', index2='end-1c')
    chat_area.delete(index1='0.0', index2=END)

    if user_input != "":

        predictions = model.predict(
            keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([user_input]),
                                                       truncating='post', maxlen=20))

        tag = encoder.inverse_transform([np.argmax(predictions)])

        for intent in data['intents']:
            if intent['tag'] == tag:
                res = np.random.choice(intent['response'])

        chat_log.config(state=NORMAL)
        chat_log.insert(END, "You: " + user_input + "\n\n")
        chat_log.insert(END, "AutoBot: " + str(res) + "\n\n")


send_btn = Button(wn, text='Send', command=send, font=('Cambria', 12), bg='sky Blue', fg='Black')
send_btn.place(x=400, y=400, width=100, height=100)

wn.mainloop()
