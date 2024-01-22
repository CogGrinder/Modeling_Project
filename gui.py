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
        self.geometry("650x500")
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
        self.label.grid(row=0, column=0, sticky=E)
        self.button = customtkinter.CTkButton(self, text='Select file', width=120, height=40, 
                                                font=('Cambria', 16), command=self.open_file_dialog)
        self.button.grid(row=1, column=0)

        self.combobox1 = customtkinter.CTkComboBox(self, values=["Square", "Horizontal Rectangle", "Vertical Rectangle"],
                                            command=self.struct_element_shape)
        self.combobox1.set("Square")
        self.combobox1.grid(row=2, column=0)

        self.combobox2 = customtkinter.CTkComboBox(self, values=['Dilation', 'Erosion'], 
                                                   command=self.operation_select)
        self.combobox2.set('Dilation')
        self.combobox2.grid(row=3, column=0)

        self.combobox3 = customtkinter.CTkComboBox(self, values=["3", "4", "5", "6", "7"],
                                            command=self.struct_element_size)
        self.combobox3.set("3")
        self.combobox3.grid(row=4, column=0)
        
    def open_file_dialog(self):
        global filename
        filename = filedialog.askopenfilename()
    
    def struct_element_shape(self, choice):
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
            img.display()
        if operation  == 'Dilation':
            img.dilation(shape, int(size))
            img.display()
            
            
class Starter_2_Window(customtkinter.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title('Starter 2')
        self.geometry("750x500")
        self.resizable(True, True)
        self.configure(bg="black")
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)
        self.rowconfigure(3, weight=1)
        self.rowconfigure(4, weight=1)
        self.rowconfigure(5, weight=1)
        self.rowconfigure(6, weight=1)
        self.rowconfigure(7, weight=1)
        
        # Variables to store user inputs
        self.selected_file = ""
        self.rotation_angle = DoubleVar()
        self.center_x = IntVar()
        self.center_y = IntVar()
        self.translation_x = IntVar()
        self.translation_y = IntVar()
        self.data_conservation_var = BooleanVar(value=False)
        self.inverse_order_var = BooleanVar(value=False)

        # Label at the top
        self.label = customtkinter.CTkLabel(self, text="Perform Rotation-Translation on a Fingerprint Image",
                              fg_color="transparent", font=('Calibri', 14, 'bold'))
        self.label.grid(row=0, column=0, sticky='w', padx=10, pady=10)

        # Select file button
        self.button = customtkinter.CTkButton(self, text='Select File', width=120, height=40,
                                font=('Cambria', 16), command=self.open_file_dialog)
        self.button.grid(row=1, column=0, pady=(0, 10))

        # Display selected file name
        self.selected_file_label = customtkinter.CTkLabel(self, text="Selected File: None", font=('Calibri', 12))
        self.selected_file_label.grid(row=2, column=0, sticky='w', padx=10)

        # Rotation angle entry
        self.angle_label = customtkinter.CTkLabel(self, text="Enter Rotation Angle (degrees):", font=('Calibri', 12))
        self.angle_label.grid(row=3, column=0, sticky='w', padx=10)
        self.angle_entry = customtkinter.CTkEntry(self, font=('Calibri', 12))
        self.angle_entry.grid(row=4, column=0, padx=10, pady=(0, 10))

        # Center of rotation entry
        self.center_label = customtkinter.CTkLabel(self, text="Enter Center of Rotation (x, y):", font=('Calibri', 12))
        self.center_label.grid(row=5, column=0, sticky='w', padx=10)
        self.center_entry_x = customtkinter.CTkEntry(self, font=('Calibri', 12))
        self.center_entry_x.grid(row=6, column=0, padx=10, pady=(0, 10))
        self.center_entry_y = customtkinter.CTkEntry(self, font=('Calibri', 12))
        self.center_entry_y.grid(row=7, column=0, padx=10, pady=(0, 10))

        # Translation entry
        self.translation_label = customtkinter.CTkLabel(self, text="Enter Translation (x, y):", font=('Calibri', 12))
        self.translation_label.grid(row=8, column=0, sticky='w', padx=10)
        self.translation_entry_x = customtkinter.CTkEntry(self, font=('Calibri', 12))
        self.translation_entry_x.grid(row=9, column=0, padx=10, pady=(0, 10))
        self.translation_entry_y = customtkinter.CTkEntry(self, font=('Calibri', 12))
        self.translation_entry_y.grid(row=10, column=0, padx=10, pady=(0, 10))

        # Checkboxes
        self.data_conservation_checkbox = customtkinter.CTkCheckBox(self, text="Data Conservation", variable=self.data_conservation_var)
        self.data_conservation_checkbox.grid(row=11, column=0, sticky='w', padx=10)
        self.inverse_order_checkbox = customtkinter.CTkCheckBox(self, text="Inverse Rotation and Translation Order", variable=self.inverse_order_var)
        self.inverse_order_checkbox.grid(row=12, column=0, sticky='w', padx=10)

        # Transform button
        self.transform_button = customtkinter.CTkButton(self, text='Transform', width=120, height=40,
                                          font=('Cambria', 16), command=self.transform_image)
        self.transform_button.grid(row=13, column=0, pady=(10, 0))

    def open_file_dialog(self):
        self.selected_file = filedialog.askopenfilename()
        if self.selected_file != "":
            # Create an instance of the Image class with the selected file
            img = Image(self.selected_file)
            # Update the label with the file name and size
            self.selected_file_label.configure(text=f"Selected File: {self.selected_file} (Size: {img.n}x{img.m})")

    def transform_image(self):
        # Retrieve user inputs
        rotation_angle = float(self.angle_entry.get())
        center_x = int(self.center_entry_x.get())
        center_y = int(self.center_entry_y.get())
        translation_x = int(self.translation_entry_x.get())
        translation_y = int(self.translation_entry_y.get())
        data_conservation = self.data_conservation_var.get()
        inverse_order = self.inverse_order_var.get()

        # Create an instance of the Image class with the selected file
        img = Image(self.selected_file)

        # Perform rotation-translation operation
        img.rotate_translate(rotation_angle, (center_x, center_y), (translation_x, translation_y),
                             data_conservation, inverse_order)

        # Display the transformed image
        img.display()
        
        
class Main_1_Simulation_Window(customtkinter.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title('Main Course 1 Simulation')
        self.geometry("650x500")
        self.resizable(True, True)
        self.configure(bg="black")
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)
        self.rowconfigure(3, weight=1)
        self.rowconfigure(4, weight=1)
        self.rowconfigure(5, weight=1)
        self.rowconfigure(6, weight=1)
        
        # Variables to store user inputs
        self.selected_file = ""
        self.center_x = IntVar()
        self.center_y = IntVar()
        self.apply_all_transformations_var = BooleanVar(value=False)

        # Explanation text
        explanation_text = "Perform an operation of simulation of low pressure on any fingerprint."
        self.explanation_label = customtkinter.CTkLabel(self, text=explanation_text, fg_color="transparent", font=('Calibri', 14, 'bold'))
        self.explanation_label.grid(row=0, column=0, sticky='w', padx=10, pady=10)

        # Select file button
        self.button = customtkinter.CTkButton(self, text='Select File', width=120, height=40,
                                font=('Cambria', 16), command=self.open_file_dialog)
        self.button.grid(row=1, column=0, pady=(0, 10))

        # Display selected file name and size
        self.selected_file_label = customtkinter.CTkLabel(self, text="Selected File: None", font=('Calibri', 12))
        self.selected_file_label.grid(row=2, column=0, sticky='w', padx=10)

        # Center coordinates entries
        self.center_label = customtkinter.CTkLabel(self, text="Enter Center of Low Pressure (x, y):", font=('Calibri', 12))
        self.center_label.grid(row=3, column=0, sticky='w', padx=10)
        self.center_entry_x = customtkinter.CTkEntry(self, font=('Calibri', 12))
        self.center_entry_x.grid(row=4, column=0, padx=10, pady=(0, 10))
        self.center_entry_y = customtkinter.CTkEntry(self, font=('Calibri', 12))
        self.center_entry_y.grid(row=5, column=0, padx=10, pady=(0, 10))

        # Checkbox for applying all transformations
        self.apply_all_transformations_checkbox = customtkinter.CTkCheckBox(self, text="Apply All Transformations", variable=self.apply_all_transformations_var)
        self.apply_all_transformations_checkbox.grid(row=6, column=0, sticky='w', padx=10)

        # Reminder text about rotating the image
        reminder_text = "Note: You might want to rotate the image before applying this kind of transformation,\n" \
                        "so the vertical axis of the fingerprint aligns with the vertical axis of the image."
        self.reminder_label = customtkinter.CTkLabel(self, text=reminder_text, fg_color="transparent", font=('Calibri', 10))
        self.reminder_label.grid(row=7, column=0, sticky='w', padx=10, pady=(0, 10))

        # Transform button
        self.transform_button = customtkinter.CTkButton(self, text='Transform', width=120, height=40,
                                          font=('Cambria', 16), command=self.transform_image)
        self.transform_button.grid(row=8, column=0, pady=(10, 0))

    def open_file_dialog(self):
        self.selected_file = filedialog.askopenfilename()
        if self.selected_file != "":
            # Create an instance of the Image class with the selected file
            img = Image(self.selected_file)
            # Update the label with the file name and size
            self.selected_file_label.configure(text=f"Selected File: {self.selected_file} (Size: {img.n}x{img.m})")

    def transform_image(self):
        # Retrieve user inputs
        center_x = int(self.center_entry_x.get())
        center_y = int(self.center_entry_y.get())
        apply_all_transformations = self.apply_all_transformations_var.get()

        # Create an instance of the Image class with the selected file
        img = Image(self.selected_file)

        # Perform low-pressure simulation
        low_pressure_img = Main_Course_1.simulate_low_pressure(img, center_x, center_y, Main_Course_1.c5)

        # Apply additional transformations if requested
        if apply_all_transformations:
            low_pressure_img.binarize(low_pressure_img.compute_threshold())
            low_pressure_img.dilation("Horizontal Rectangle", 3)
            low_pressure_img.blur(3)

        # Display the transformed image
        low_pressure_img.display()


        
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
                                                          "Starter 4", "Starter 5", "Main Course 1 (Simulation)", "Main course 5"],
                                            command=self.combobox_callback)
        self.combobox.set("Starter 1")
        self.combobox.grid(row=1, column=0)

    #This function creates the command that should be executed whenever a section is selected
    # To be updated
    def combobox_callback(self, choice):
        if choice == "Starter 1":
            os.system('starter1.py')
        if choice == "Starter 2":
            self.toplevel_window = None
            if self.toplevel_window is None or not self.toplevel_window.winfo_exists():
                self.toplevel_window = Starter_2_Window(self)  # create window if its None or destroyed
            else:
                self.toplevel_window.focus()
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
        if choice == "Main Course 1 (Simulation)":
            self.toplevel_window = None
            if self.toplevel_window is None or not self.toplevel_window.winfo_exists():
                self.toplevel_window = Main_1_Simulation_Window(self)  # create window if its None or destroyed
            else:
                self.toplevel_window.focus()
        if choice == "Main course 5":
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