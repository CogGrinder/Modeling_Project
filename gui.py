import os
from tkinter import filedialog
import numpy as np
import customtkinter
from tkinter import *
from PIL import Image as Im
from tkdocviewer import *
from image import Image
from starter2 import Starter_2
from starter3 import Starter_3

from main_course_1 import Main_Course_1

class Starter_4_Window(customtkinter.CTkToplevel):
    # Create a window for starter 4
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title('Starter 4')
        self.geometry("{0}x{1}+0+0".format(self.winfo_screenwidth(), self.winfo_screenheight()))
        self.resizable(True, True)
        self.state('normal')
        self.configure(bg="black")
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)   
        self.rowconfigure(2, weight=1)  
        self.rowconfigure(3, weight=1) 
        self.rowconfigure(4, weight=1)    
        self.label = customtkinter.CTkLabel(self, 
                                            text="Here, you are about to either dilate or erode the binary version of an image." +
                                            "\n You first need to select your image:",
                                            fg_color="transparent", font=('Calibri', 14, 'bold'))
        self.label.grid(row=0, column=0)
        self.button = customtkinter.CTkButton(self, text='Select file', width=120, height=40, 
                                                font=('Cambria', 16), command=self.open_file_dialog)
        self.button.grid(row=1, column=0)

        self.combobox1 = customtkinter.CTkComboBox(self, values=["Square", "Horizontal Rectangle", "Vertical Rectange"],
                                            command=self.struct_element_shape)
        self.combobox1.set("Square")
        self.combobox1.grid(row=2, column=0)

        self.combobox2 = customtkinter.CTkComboBox(self, values=['Dilation', 'Erosion'], 
                                                   command=self.operation_select)
        self.combobox2.set('Dilation')
        self.combobox2.grid(row=3, column=0)

        self.combobox3 = customtkinter.CTkComboBox(self, values=[3, 4, 5, 6, 7],
                                            command=self.struct_element_size)
        self.combobox3.set(3)
        self.combobox3.grid(row=3, column=0)
        
    def open_file_dialog(self):
        global filename
        filename = filedialog.askopenfilename()
    
    def stuct_element_shape(self, choice):
        global shape
        shape = choice
    
    def operation_select(self, choice):
        global operation
        operation = choice

    def struct_element_size(self, choice):
        img = Image(filename)
        global size
        size = choice
        threshold = img.compute_threshold()
        img.binarize(threshold)
        if operation == 'Erosion':
            img.erosion(shape, int(size))
        if operation  == 'Dilation':
            img.dilation(shape, int(size))


        
class SecondWindow(customtkinter.CTkToplevel):
    #Build a new window for section selection
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title('Section Selection')
        self.geometry("250x150")
        self.resizable(False, False)
        self.state('normal')
        self.configure(bg="black")
        # Reconfigure our rows and columns for grid
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=2)
        self.label = customtkinter.CTkLabel(self, 
                                            text="Pick the section you want \n to have a snippet of:",
                                            fg_color="transparent", font=('Calibri', 16, 'bold'))
        self.label.grid(column=0, row=0)
        #Create a list of the different sections we have worked on so far
        self.combobox = customtkinter.CTkComboBox(self, values=["Starter 1", "Starter 2", "Starter 3",
                                                          "Starter 4", "Starter 5", "Main course 1", "Main course 5"],
                                            command=self.combobox_callback)
        self.combobox.set("Starter 1")
        self.combobox.grid(row=1, column=0)

    #This function creates the command that should be executed whenever a section is selected
    # To be updated
    def combobox_callback(self, choice):
        if choice == "Starter 1":
            os.system('starter1.py')
        if choice == "Starter 2":
            os.system('starter2.py')
        if choice == "Starter 3":
            os.system('starter3.py')
        if choice == "Starter 4":
            self.toplevel_window = None
            if self.toplevel_window is None or not self.toplevel_window.winfo_exists():
                self.toplevel_window = Starter_4_Window(self)  # create window if its None or destroyed
            else:
                self.toplevel_window.focus()
        if choice == "Starter 5":
            os.system('starter5.py')
        if choice == "Main course 1":
            os.system('main_course_1.py')
        if choice == "Main vourse 5":
            print("combobox dropdown clicked:", choice)


class App(customtkinter.CTk):
    #Creates the home window
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
        image_1 = customtkinter.CTkImage(light_image=Im.open("images/uga.png"),
                                        dark_image=Im.open("images/uga.png"),
                                        size=(250, 100))
        self.label_img_1 = customtkinter.CTkLabel(self, image=image_1, text = "")
        self.label_img_1.grid(row=0, column=1)

        self.label_1 = customtkinter.CTkLabel(self, text="Fingerprint Digital Processing", fg_color="transparent",
                                    font=('Helvetica', 30, 'bold'))
        self.label_1.grid(row=1, column=1, sticky=N)

        image_2 = customtkinter.CTkImage(light_image=Im.open("images/fingerprint.png"),
                                        dark_image=Im.open("images/fingerprint.png"),
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
            self.toplevel_window = SecondWindow(self)  # create window if its None or destroyed
        else:
            self.toplevel_window.focus()




app = App()
app.mainloop()