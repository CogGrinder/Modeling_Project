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

class ToplevelWindow(customtkinter.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry("250x150")
        self.resizable(False, False)
        self.state('normal')
        self.configure(bg="black")
        # Reconfigure our rows and columns for grid
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=2)
        self.columnconfigure(2, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)
        self.label = customtkinter.CTkLabel(self, 
                                            text="Pick the section you want to have a snippet of:",
                                            fg_color="transparent", font=('Calibri', 16, 'bold'))
        self.label.grid(column=0, row=0)

        self.combobox = customtkinter.CTkComboBox(self, values=["Starter 1", "Starter 2", "Starter 3",
                                                          "Starter 4", "Starter 5", "Main course 1", "Main course 5"],
                                            command=self.combobox_callback)
        self.combobox.set("Starter 1")
        self.combobox.grid(row=1, column=0)

    def combobox_callback(self, choice):
        print("combobox dropdown clicked:", choice)


class App(customtkinter.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry("{0}x{1}+0+0".format(1100, 700))
        self.title('Fingerprint Image Processing')
        # self.iconbitmap('unfpa_logo.ico')
        self.resizable(False, False)
        self.state('normal')
        self.configure(bg="black")
        # Reconfigure our rows and columns for grid
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=2)
        self.columnconfigure(2, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)
        image_1 = customtkinter.CTkImage(light_image=Image.open("images/uga.png"),
                                        dark_image=Image.open("images/uga.png"),
                                        size=(250, 100))
        self.label_img_1 = customtkinter.CTkLabel(self, image=image_1, text = "")
        self.label_img_1.grid(row=0, column=1)

        self.label_1 = customtkinter.CTkLabel(self, text="Fingerprint Digital Processing", fg_color="transparent",
                                    font=('Helvetica', 30, 'bold'))
        self.label_1.grid(row=1, column=1, sticky=N)

        image_2 = customtkinter.CTkImage(light_image=Image.open("images/fingerprint.png"),
                                        dark_image=Image.open("images/fingerprint.png"),
                                        size=(100, 40))
        self.label_img_2 = customtkinter.CTkLabel(self, image=image_2, text = "")
        self.label_img_2.grid(row=1, column=1)

        self.label_2 = customtkinter.CTkLabel(self, text="With Bierhoff, Sabin, Sacha and Vincent", fg_color="transparent",
                                    font=('Comic Sans MS', 30, 'italic'))
        self.label_2.grid(row=2, column=1, sticky=N)

        self.button_1 = customtkinter.CTkButton(self, text='Start', width=120, height=40, font=('Cambria', 16), command=self.open_toplevel)
        self.button_1.grid(row=2, column=1)

        self.toplevel_window = None


    def open_toplevel(self):
        if self.toplevel_window is None or not self.toplevel_window.winfo_exists():
            self.toplevel_window = ToplevelWindow(self)  # create window if its None or destroyed
        else:
            self.toplevel_window.focus()




app = App()
app.mainloop()