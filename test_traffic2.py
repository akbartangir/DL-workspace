import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
from keras.models import load_model

# Load the trained model to classify sign
model = load_model('model2.h5')

# Dictionary to label all traffic signs classes
classes = {1: 'Speed limit (20km/h)',
           2: 'Speed limit (30km/h)',
           3: 'Speed limit (50km/h)',
           4: 'Speed limit (60km/h)',
           5: 'Speed limit (70km/h)',
           6: 'Speed limit (80km/h)',
           7: 'End of speed limit (80km/h)',
           8: 'Speed limit (100km/h)',
           9: 'Speed limit (120km/h)',
           10: 'No passing',
           11: 'No passing veh over 3.5 tons',
           12: 'Right-of-way at intersection',
           13: 'Priority road',
           14: 'Yield',
           15: 'Stop',
           16: 'No vehicles',
           17: 'Veh > 3.5 tons prohibited',
           18: 'No entry',
           19: 'General caution',
           20: 'Dangerous curve left',
           21: 'Dangerous curve right',
           22: 'Double curve',
           23: 'Bumpy road',
           24: 'Slippery road',
           25: 'Road narrows on the right',
           26: 'Road work',
           27: 'Traffic signals',
           28: 'Pedestrians',
           29: 'Children crossing',
           30: 'Bicycles crossing',
           31: 'Beware of ice/snow',
           32: 'Wild animals crossing',
           33: 'End speed + passing limits',
           34: 'Turn right ahead',
           35: 'Turn left ahead',
           36: 'Ahead only',
           37: 'Go straight or right',
           38: 'Go straight or left',
           39: 'Keep right',
           40: 'Keep left',
           41: 'Roundabout mandatory',
           42: 'End of no passing',
           43: 'End no passing vehicle with a weight greater than 3.5 tons'}

# Initialize GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Traffic Sign Classification')
top.configure(background='grey')

# Centralizing widgets
label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
label.pack(side=TOP, pady=10)
sign_image = Label(top)

# Function to preprocess image before classification
def preprocess_image(file_path):
    image = Image.open(file_path)
    image = image.resize((32, 32))  # Resize image to match the input size of the model (32x32)
    image = image.convert('L')  # Convert image to grayscale (if needed)
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=-1)  # Add channel dimension (32, 32, 1)
    image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 32, 32, 1)
    return image

# Function to classify the image and show the result
def classify(file_path):
    image = preprocess_image(file_path)
    pred = model.predict(image)  # Predict the class
    pred_class = np.argmax(pred, axis=1)[0] + 1  # Get the predicted class (0-based index adjusted to 1-based)
    sign = classes.get(pred_class, "Unknown Class")  # Get the traffic sign name
    print(f"Predicted Class: {sign}")
    label.configure(foreground='#011638', text=sign)

# Function to show the "Classify Image" button after an image is uploaded
def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='black', font=('arial', 12, 'bold'))
    classify_b.place(relx=0.5, rely=0.75, anchor='center')  # Centered button

# Function to upload an image
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except Exception as e:
        print(f"Error: {e}")
        pass

# Upload button
upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='black', font=('arial', 14, 'bold'))
upload.pack(side=BOTTOM, pady=20)

# Pack the widgets
sign_image.pack(side=TOP, expand=True, padx=20, pady=20)
label.pack(side=TOP, expand=True)

# Heading
heading = Label(top, text="Traffic Sign Recognition", pady=10, font=('arial', 20, 'bold'))
heading.configure(background='grey', foreground='#364156')
heading.pack()

# Run the Tkinter main loop
top.mainloop()
