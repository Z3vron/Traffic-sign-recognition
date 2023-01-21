import PySimpleGUI as sg
import cv2
from pathlib import Path
import numpy as np
from keras.models import load_model
import tensorflow as tf
from tensorflow import keras
import os
from PIL import Image

# Load the model to classify sign
model = load_model('CNN_model.h5')
size = 300
image_to_classify_path = ""

# List of sign names
names = {1: 'Speed limit (20km/h)',
         2: 'Speed limit (30km/h)',
         3: 'Speed limit (50km/h)',
         4: 'Speed limit (60km/h)',
         5: 'Speed limit (70km/h)',
         6: 'Speed limit (80km/h)',
         7: 'Stop',
         8: 'Speed limit (100km/h)',
         9: 'No Entry',
         10: 'No passing',
         11: 'No passing for trucks',
         12: 'Right-of-way at intersection',
         13: 'Priority road',
         14: 'Yield priority',
         15: 'Traffic lights',
         16: 'Pedestrians',
         17: 'Bicycles',
         18: 'Go straight',
         19: 'Turn right',
         20: 'Turn left',
         21: 'Roundabout',
         }


# functions
def classify():
    img = keras.preprocessing.image.load_img(image_to_classify_path, target_size=(30, 30))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    pred = np.argmax(model.predict([img_array]), axis=-1)[0]
    window["SignName"].update(names[pred + 1])

    folder_path_reference_signs = 'Reference_photos'
    image_name_reference_sign = "{0}.png".format(pred)
    image_path_reference_sign = os.path.join(folder_path_reference_signs,image_name_reference_sign)
    try:
       imgage_reference_sign =  Image.open(image_path_reference_sign) 
    except:
        print("cant find image on given path")

    try:
        res, img_to_show = cv2.imencode(".png", cv2.imread(image_path_reference_sign))
     
    except:
        window["Status"].update("Cannot load refence sign image")
        return
    
    window["Result"].update(data=img_to_show.tobytes())

def read_file():
    global image_to_classify_path
    image_to_classify_path = sg.popup_get_file("", no_window=True)
    if image_to_classify_path == "":
        return

    window["Status"].update("Choose image")
    if not Path(image_to_classify_path).is_file():
        window["Status"].update("Image not found")
        return

    try:
        res, image = cv2.imencode(".png", cv2.imread(image_to_classify_path))
    except:
        window["Status"].update("Cannot identify image")
        return

    window["Image"].update(data=image.tobytes())


# gui
gui = [
    [sg.Text("Choose image", expand_x=True, key="Status")],
    [sg.Button("Browse"), sg.Button("Classify")],
    [sg.Image(size=(size, size), key="Image")]
]

# result gui
result_gui = [
    [sg.Text(key="SignName")],
    [sg.Image(size=(size, size), key="Result")]
]

layout = [
    [
        sg.Column(gui, vertical_alignment="center", justification="center"),
        sg.VSeperator(),
        sg.Column(result_gui, vertical_alignment="center", justification="center")
    ]
]

window = sg.Window("Project", layout)

# Event loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    elif event == "Browse":
        read_file()
    elif event == "Classify":
        classify()
window.close()
