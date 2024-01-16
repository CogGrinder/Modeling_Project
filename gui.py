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
# Reconfigure our rows and columns for grid
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=2)
root.columnconfigure(2, weight=1)
root.rowconfigure(0, weight=1)
root.rowconfigure(1, weight=1)
root.rowconfigure(2, weight=1)
image_1 = customtkinter.CTkImage(light_image=Image.open("images/uga.png"),
                                  dark_image=Image.open("images/uga.png"),
                                  size=(350, 150))
label_img_1 = customtkinter.CTkLabel(root, image=image_1, text = "")
label_img_1.grid(row=0, column=1)

label_1 = customtkinter.CTkLabel(root, text="Fingerprint Digital Processing", fg_color="transparent",
                               font=('Helvetica', 50, 'bold'))
label_1.grid(row=1, column=1, sticky=N)

image_2 = customtkinter.CTkImage(light_image=Image.open("images/fingerprint.png"),
                                  dark_image=Image.open("images/fingerprint.png"),
                                  size=(150, 80))
label_img_2 = customtkinter.CTkLabel(root, image=image_2, text = "")
label_img_2.grid(row=1, column=1)

label_2 = customtkinter.CTkLabel(root, text="With Bierhoff, Sabin, Sacha and Vincent", fg_color="transparent",
                               font=('Comic Sans MS', 50, 'italic'))
label_2.grid(row=2, column=1, sticky=N)

button_1 = customtkinter.CTkButton(root, text='Start', width=150, height=40, font=('Cambria', 20))
button_1.grid(row=2, column=1)


root.mainloop()