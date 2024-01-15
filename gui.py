import os
import tkinter
import numpy as np
import customtkinter
from tkinter import *
from PIL import ImageTk, Image
from tkdocviewer import *
# from image import Image
from starter2 import Starter_2
from starter3 import Starter_3
from main_course_1 import Main_Course_1


#Create root window
root = customtkinter.CTk()
root.title('Fingerprint Image Processing')
# root.iconbitmap('unfpa_logo.ico')
root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
root.resizable(True, True)
root.state('normal')
root.configure(bg="black")

image_1 = Image.open("images/clean_finger.png")
resized_image = image_1.resize((100, 100))
test = ImageTk.PhotoImage(resized_image)

# Reconfigure our rows and columns for grid
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
root.rowconfigure(1, weight=2)
label_img = customtkinter.CTkLabel(root, image=test)
label_img.grid(row=0, column=0, sticky=EW)

button_1 = customtkinter.CTkButton(root, text='Enter')
button_1.grid(row=1, column=0, sticky=S)


root.mainloop()