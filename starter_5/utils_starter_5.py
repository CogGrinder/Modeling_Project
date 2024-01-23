import numpy as np
import cv2
import matplotlib.pyplot as plt

import math
import sys
import os
sys.path.append(os.getcwd()) #to access current working directory files easily
from image import Image

surface_sampling = 85
default_scheme_step = 0.2

class Utils_starter_5:
    def __init__(self, img1 : Image, img2 : Image):
        self._fixed_img  = img1 #fixed
        self._moving_img = img2 #moving
        return

    def display_all(self) :
        self._fixed_img.display()
        self._moving_img.display()
        return

    def display_warped(self,p,warp,loss_function:callable) :
        fig, ax = plt.subplots(1,3,figsize = (12,12))
        
        print(ax)
        ax[0].set_title("Fixed image")
        ax[1].set_title(f"Warped image, $p=[{p[0]:.2f},{p[1]:.2f}]$")
        ax[2].set_title("Superimposed image")

        ax[0].imshow(self._fixed_img.data,cmap="Blues")
        ax[2].imshow(self._fixed_img.data,cmap="Blues")
        i,j = np.meshgrid(np.arange(self._fixed_img.data.shape[0]),
                           np.arange(self._fixed_img.data.shape[1]),indexing='ij')

        ax[1].imshow(warp(i,j,p),cmap="Oranges",alpha=1)
        ax[2].imshow(warp(i,j,p),cmap="Oranges",alpha=0.5)
        plt.show()
        return

    def get_pix_at_translated(self, x, y, p : list):
        """Optimized translate function

        Args:
            x (int or np.ndarray): numpy matrix containing x of pixel, assumed meshgrid with "i,j" if matrix
            y (int or np.ndarray): numpy matrix containing y of pixel, assumed meshgrid with "i,j" if matrix
            p (list): p is assumed a list of 2 translate parameters

        Returns:
            _type_: _description_
        """
        n,m = self._moving_img.data.shape
        int_px = math.floor(p[0]) #integer part
        int_py = math.floor(p[1])

        decimal_px = p[0] - int_px #decimal part for interpolation, used at return
        decimal_py = p[1] - int_py

        # Optimised conditional statement
        # creates a boolean matrix for each element of x and y
        # verifies that fetched pixel is in the image
        is_in_image = np.logical_and(np.logical_and(int_px<x, x<n+int_px) ,
                                        np.logical_and(int_py<y, y<m+int_py)) #TODO:remove ones and <=
        # equivalent to "if 0<x-int_px<n and 0<y-int_py<m" elementwise

        """
        note : "0<" is to account for fetching neighbours for interpolation,
        resulting in masking the first row and first column of the image
        """

        dummy_x = x.copy()
        dummy_y = y.copy()

        dummy_values = [int_px +1, int_py +1]
        # if False, replace x and y values by dummy values
        dummy_x[np.logical_not(is_in_image)] = dummy_values[0]  # used to nullify the index in the np.where below
        dummy_y[np.logical_not(is_in_image)] = dummy_values[1]

        # TODO add padding
        # translate including dummy values, bilinearly interpolated wrt decimal parts
        dummy_translate = \
        + decimal_px    * decimal_py     *self._moving_img.data[dummy_x-int_px-1,  dummy_y-int_py-1] \
        + decimal_px    * (1-decimal_py) *self._moving_img.data[dummy_x -int_px-1, dummy_y-int_py] \
        + (1-decimal_px)* decimal_py     *self._moving_img.data[dummy_x-int_px,    dummy_y-int_py-1] \
        + (1-decimal_px)* (1-decimal_py) *self._moving_img.data[dummy_x -int_px,   dummy_y-int_py]
        #try7 finally working? inverted 10 and 01

        """
        (1-decimal_px)*( (1-decimal_py) *self._moving_img.data[dummy_x-int_px,   dummy_y-int_py] \
                        + decimal_py    *self._moving_img.data[dummy_x -int_px,  dummy_y-int_py-1]) \
        + decimal_px  *( (1-decimal_py) *self._moving_img.data[dummy_x-int_px-1, dummy_y-int_py] \
                        + decimal_py    *self._moving_img.data[dummy_x -int_px-1,dummy_y-int_py-1])
        #previous version
        """

        """
        + decimal_px    * decimal_py     *self._moving_img.data[dummy_x-int_px-1,  dummy_y-int_py-1] \
        + decimal_px    * (1-decimal_py) *self._moving_img.data[dummy_x -int_px,   dummy_y-int_py-1] \
        + (1-decimal_px)* decimal_py     *self._moving_img.data[dummy_x-int_px-1,  dummy_y-int_py] \
        + (1-decimal_px)* (1-decimal_py) *self._moving_img.data[dummy_x -int_px,   dummy_y-int_py]
        #try6 finally working? added -2 everywhere - no i got mixed up since a long time ago with x and y
        """
        
        """
        + decimal_px    * decimal_py     *self._moving_img.data[dummy_x-int_px,  dummy_y-int_py] \
        + decimal_px    * (1-decimal_py) *self._moving_img.data[dummy_x -int_px+1, dummy_y-int_py] \
        + (1-decimal_px)* decimal_py     *self._moving_img.data[dummy_x-int_px,    dummy_y-int_py+1] \
        + (1-decimal_px)* (1-decimal_py) *self._moving_img.data[dummy_x -int_px+1,   dummy_y-int_py+1]
        #try5 reverting and going reversed in crossed directions, all directions then only 11 and 00
        """
        
        """
        + decimal_px    * decimal_py     *self._moving_img.data[dummy_x-int_px,    dummy_y-int_py] \
        + decimal_px    * (1-decimal_py) *self._moving_img.data[dummy_x -int_px,   dummy_y-int_py-1] \
        + (1-decimal_px)* decimal_py     *self._moving_img.data[dummy_x-int_px-1,  dummy_y-int_py] \
        + (1-decimal_px)* (1-decimal_py) *self._moving_img.data[dummy_x -int_px-1, dummy_y-int_py-1]
        #try4 still discrepancy between int indexes and float ones
        """

        """
          decimal_px    *( decimal_py     *self._moving_img.data[dummy_x-int_px+1,   dummy_y-int_py+1] \
                         + (1-decimal_py) *self._moving_img.data[dummy_x -int_px+1,  dummy_y-int_py]) \
        + (1-decimal_px)*( decimal_py     *self._moving_img.data[dummy_x-int_px, dummy_y-int_py+1] \
                         + (1-decimal_py) *self._moving_img.data[dummy_x -int_px,dummy_y-int_py])
        #try3
        """


        """
        (1-decimal_px)*( (1-decimal_py) *self._moving_img.data[dummy_x-int_px+1,   dummy_y-int_py+1] \
                        + decimal_py    *self._moving_img.data[dummy_x -int_px+1,  dummy_y-int_py]) \
        + decimal_px  *( (1-decimal_py) *self._moving_img.data[dummy_x-int_px, dummy_y-int_py+1] \
                        + decimal_py    *self._moving_img.data[dummy_x -int_px,dummy_y-int_py])
        #try2 probably doesnt work
        """
        
        """
          decimal_px    *( decimal_py     *self._moving_img.data[dummy_x-int_px,   dummy_y-int_py] \
                         + (1-decimal_py) *self._moving_img.data[dummy_x -int_px,  dummy_y-int_py-1]) \
        + (1-decimal_px)*( decimal_py     *self._moving_img.data[dummy_x-int_px-1, dummy_y-int_py] \
                         + (1-decimal_py) *self._moving_img.data[dummy_x -int_px-1,dummy_y-int_py-1])
        #try1
        """

        #1 is white padding
        filtered_translate = np.where(is_in_image,
                                      dummy_translate,
                                      1)

        return filtered_translate
        




    def loss_function_1(self,**kwargs):#(self, p : list, warp : callable = get_pix_at_translated): #[[int,int,list],float]
        p = 0
        warp = self.get_pix_at_translated #default warp function
        for params in kwargs:
            if params == 'p':
                p = kwargs['p']
            if params == 'warp':
                warp = kwargs['warp']
        
        # Old version
        # warped_img2 = np.array([[warp(i,j,p) for j in range(self._img2.data.shape[1])] for i in range(self._img2.data.shape[0])])

        # New version optimized
        i,j = np.meshgrid(np.arange(self._moving_img.data.shape[0]), np.arange(self._moving_img.data.shape[1]),indexing="ij")
        
        warped_moving_image = warp(i,j,p) # uses numpy arrays here
        return np.sum((self._fixed_img.data - warped_moving_image)**2)
    
    def loss_function_2(self,**kwargs):#(self, p : list, warp : callable = get_pix_at_translated):
        p = [0,0]
        warp = self.get_pix_at_translated
        for params in kwargs:
            if params == 'p':
                p = kwargs['p']
            if params == 'warp':
                warp = kwargs['warp']
        
        
        i,j = np.meshgrid(np.arange(self._moving_img.data.shape[0]), np.arange(self._moving_img.data.shape[1]),indexing="ij")
        warped_moving_image = warp(i,j,p) # uses numpy arrays here
        
        moving_average = np.mean(self._fixed_img.data) # TODO testing fixed average
        # print("warped_average ",warped_average)

        fixed_average = np.mean(self._fixed_img.data)
        # print("fixed_average ",fixed_average)
        

        term1 = np.sum((warped_moving_image - moving_average) * (self._fixed_img.data - fixed_average))
        # print("term1 ",term1)
        term2 = np.sum((self._fixed_img.data - fixed_average)**2)
        # print("term2 ",term2)
        term3 = np.sum((warped_moving_image - moving_average)**2)
        # print("term3 ",term3)



        return -term1/np.sqrt(term2*term3) * 10**(4.5) # arbitrary normalisation with the idea of maximising
    
    def make_save_name(self, loss_function:callable) :
        return  self._fixed_img.name + "_" + self._moving_img.name + "_" + str(loss_function.__name__) + ".txt"

    def import_data(self, loss_function:callable) :
        """Imports loss_function data from named .txt file

        Args:
            loss_function (callable): loss function, used for finding the name format of the file

        Returns:
            np.ndarray: meshgrid x indexes for p parameter (px)
            np.ndarray: meshgrid y indexes for p parameter (py)
            np.ndarray: imported data from loss function (loss_grid)
        """

        save_filename = self.make_save_name(loss_function)
        
        if os.path.exists(save_filename) :
            with open(save_filename, "r") as f :
                # first line of the file
                line1 = f.readline().rstrip('\n').split(" ")
                print(line1)
                # used for computing x and y grid
                span_x = 0
                span_y = 0
                step = 1
                # compute parameters from first line of file
                for i, word in enumerate(line1) :
                    if word == "span_x" :
                        span_x = [int(line1[i+1]),
                                    int(line1[i+2])]
                    if word == "span_y" :
                        span_y = [int(line1[i+1]),
                                    int(line1[i+2])]
                    if word == "step":
                        step = int(line1[i+1])

                print(span_x,span_y)
                right_bound_x = span_x[1]
                right_bound_y = span_y[1]


                # Create table and x and y :
                px, py = np.meshgrid(np.linspace(-right_bound_x,right_bound_x, math.floor(right_bound_x/step * 2),\
                                                  endpoint=False).astype(int),
                                     np.linspace(-right_bound_y,right_bound_y, math.floor(right_bound_y/step * 2),\
                                                  endpoint=False).astype(int))
                loss_grid = np.zeros((right_bound_y * 2,right_bound_x * 2) ) # TODO potential error: mixup between x,y and i,j
            

                for i in range(span_x[1]*2):
                    line =  f.readline().rstrip('\n').split(" ")
                    for j, word in enumerate(line) :
                        if word!="":
                            loss_grid[i,j] = float(word)

                return px, py, loss_grid
        else :
            compute = input(f"No data for {loss_function.__name__}. Do you want to compute ? (y/n)")
            while(not compute in ["y","n"]):
                print("Please enter y or n")
                compute = input(f"No data for {loss_function.__name__}. Do you want to compute ? (y/n)")
            if compute == "y":
                return self.compute_and_plot_loss(show=False,span="all")
            else:
                return np.array([]), np.array([]), np.array([])

            
    def compute_and_plot_loss(self, **kwargs) : #TODO : vary the ranges and shapes for p, maybe with relation to loss function and image data
        """Function used to greedily calculate all the loss_function returns for a certain range of p

        Args:
            kwargs :
                loss_function (callable): function treated as loss function with a parameter p
                save : bool, saves as .txt
                show : bool, plots the function
                span : "all" for whole span of p
        
        Returns:
            np.ndarray: meshgrid x indexes for p parameter (px)
            np.ndarray: meshgrid y indexes for p parameter (py)
            np.ndarray: imported data from loss function (loss_grid)
            
        """
        print("test_plot_loss")


        # Default parameter values
        loss_function = self.loss_function_1
        save = True
        show = True
        n,m = self._moving_img.data.shape
        
        
        translate_span_x = 1
        translate_span_y = m//3
        
        #reading kwargs
        for key, value in kwargs.items() :
            if key == "loss_function" :
                loss_function = value
            if key == "save":
                save = bool(value)
            if key == "show":
                show = bool(value)
            if key == "span":
                if value == "all" :
                    translate_span_x = n//2
                    translate_span_y = m//2
                else:
                    raise ValueError("unknown value for span")

        #check for existing file
        save_filename = self.make_save_name(loss_function)
        print(save_filename)
        #if file exists, ask user to confirm overwriting
        compute = "y"
        if os.path.exists(save_filename) :
                compute = input(f"Data exists for {loss_function.__name__}. Do you want to compute and overwrite ? (y/n)")
                while(not compute in ["y","n"]):
                    print("Please enter y or n")
                    compute = input(f"Data exists for {loss_function.__name__}. Do you want to compute and overwrite ? (y/n)")

        if compute == "y":

            px, py = np.meshgrid(np.linspace(-translate_span_x,translate_span_x, translate_span_x * 2, endpoint=False).astype(int),
                                np.linspace(-translate_span_y,translate_span_y, translate_span_y * 2, endpoint=False).astype(int))
            #using "x,y" convention because stored values will be plotted.
            # TODO However, is x and y the x and y from the picture?

            loss_grid = np.zeros(px.shape)
            for i in range(px.shape[0]):
                for j in range(px.shape[1]):
                    p = [px[i,j],py[i,j]]
                    print(p)
                    loss_grid[i][j] = loss_function(p = p, warp = self.get_pix_at_translated ) #pass the px and py values
            
            print("loss computation done")

            if save :
                with open(save_filename, "w") as f :
                    f.write(" ".join(["span_x", str(-translate_span_x), str(translate_span_x),
                            "span_y", str(-translate_span_y), str(translate_span_y),
                            "step", str(1)]) ) #add step
                    f.write("\n")
                    

                    for i in range(loss_grid.shape[0]) :
                        f.write(str(loss_grid[i,0]))
                        for j in range(1,loss_grid.shape[1]) :
                            f.write(" " + str(loss_grid[i,j]))
                        f.write("\n")


        if show :
            ax = plt.figure().add_subplot(projection='3d')
            
            if compute == "y":
                ax.plot_surface(px,py,loss_grid,rcount=surface_sampling,ccount=surface_sampling)
            else :
                px, py, loss_grid = self.import_data(loss_function)
                ax.plot_surface(px,py,loss_grid,rcount=surface_sampling,ccount=surface_sampling)
            
            plt.show()
        else:
            if compute != "y":
                print("compute is y")
                px, py, loss_grid = self.import_data(loss_function)

        
        return px, py, loss_grid

    def greedy_optimization_xy(self, **kwargs) : #TODO : vary the ranges and shapes for p, maybe with relation to loss function and image data
        """greedy brute force strategy to find the optimal value of p_x or of [p_x,p_y]

        Args:
            loss_function (callable): function treated as loss function with a parameter p
            kwargs :
                translate_type : either "xy" or "x"
                plot : default is False - choose wether to plot the loss function
                loss_function : callable function which takes a parameter p
        """
        print("greedy_optimization_xy")
        
        xy_translate = "xy"
        plot = False
        loss_function = self.loss_function_1
        
        # reading kwargs
        for key,value in kwargs.items():
            if key == "loss_function":
                loss_function = value
            if key == "plot":
                if not (type(value) is bool):
                    print(type(value))
                    raise TypeError("plot must be a bool")
                else :
                    plot = value
            if key == "translate_type" :
                if value in ["xy", "x"]:
                    xy_translate = value
                else:
                    raise ValueError("unknown translate_type")
        
        print("~~~~~~~~~~~~")
        print("Parameters :")
        print("~~~~~~~~~~~~")

        for key,value in kwargs.items() :
            print(key,": ",value)
        n,m = self._moving_img.data.shape

        l_min   = sys.float_info.max
        l_list  = np.zeros(n+1) #used to return the loss function for plotting
        
        p_min = [0,0]
        
        if xy_translate == "x" :
            step = 1 #for testing purposes
            list_px = list(np.arange(- math.ceil(n/2), math.floor(n/2) + 1, step))
            l_list = np.zeros(len(list_px))
            for i, p_x in enumerate(list_px) :
                l = loss_function(p=[0,p_x]) #beware the indexation
                
                if l_min > l : #update the min and argmin
                    l_min = l
                    p_min = [0,p_x]
                
                l_list[i] = l
            print("The translation in x that minimizes our loss function is ", p_min[1])
            if plot :
                #p = np.arange(- math.ceil(n/2), math.floor(n/2) + 1) #pre-testing
                plt.plot(list_px,l_list)
                #plt.plot(p,l_list) #pre-testing
                plt.show()
        elif xy_translate == "xy" :
            l_list  = np.zeros((m+1,n+1)) #make the list bigger to accomodate for all translations
            for j, p_x in enumerate(range(- math.ceil(n/2), math.floor(n/2) + 1)) :
                for i, p_y in enumerate(range(- math.ceil(m/2), math.floor(m/2) + 1)) :
                    l = loss_function(p=[p_y,p_x]) #beware the indexation
                    print([p_y,p_x])
                    
                    if l_min > l :
                        l_min = l
                        p_min = [p_y,p_x] #beware the indexation
                    l_list[i][j] = l
            print("The translation in y, x coordinates that minimizes our loss function is ", p_min)
            if plot :
                ax = plt.figure().add_subplot(projection='3d')
                
                px_loss, py_loss, loss_data = self.import_data(loss_function)
                if px_loss != None:
                    ax.plot_surface(px_loss,py_loss,loss_data,rcount=surface_sampling,ccount=surface_sampling)
                    plt.show()

                p_x, p_y = np.meshgrid(np.arange(- math.ceil(n/2), math.floor(n/2) + 1),np.arange(- math.ceil(m/2), math.floor(m/2) + 1))
                ax.plot_surface(p_x,p_y,l_list,rcount=surface_sampling,ccount=surface_sampling)
                plt.show()

        print("min loss and argmin computation done")

        return p_min, l_list


    def coordinate_descent_optimization_xy(self, **kwargs) :
        """non differentiable coordinate descent strategy to find the optimal value of [p_x,p_y]

        Args:
            loss_function (callable): function treated as loss function with a parameter p
            kwargs :
                plot : default is False - choose wether to plot the loss function
                loss_function : callable function which takes a parameter p
                epsilon : stopping level for our loss function decrease
                epsilon2 : stopping level for alpha
                dx : scheme step
                p0 : initial p parameter for loss function
                alpha0 : initial percentage (in direct multiplicative factor form) for adjustment of p


        """
        print("coordinate_descent_optimization_xy")
        

        ### Defaults ###
        # Initial warp parameters
        p0 = [0,0]
        # Initial gradient descent speed
        alpha0 = 0.2
        plot = False
        loss_function = self.loss_function_1
        epsilon = 10 #arbitrary default
        epsilon2 = 0.05 #arbitrary default
        scheme_step = default_scheme_step #scheme step

        warp = self.get_pix_at_translated # new warp function parameter

        if "loss_function" in kwargs:
            loss_function = kwargs["loss_function"]
        if "warp" in kwargs:
            warp = kwargs["warp"]
        if "plot" in kwargs:
            value = kwargs["plot"]
            if not (type(value) is bool):
                print(type(value))
                raise TypeError("plot must be a bool")
            else :
                plot = value
        if "epsilon" in kwargs:
            value = kwargs["epsilon"]
            if value<0:
                raise ValueError("epsilon must be positive")
            else:
                epsilon = value
        if "epsilon2" in kwargs:
            value = kwargs["epsilon2"]
            if value<0:
                raise ValueError("epsilon must be positive")
            else:
                epsilon2 = value
        if "dx" in kwargs:
            value = kwargs["dx"]
            if value<0:
                raise ValueError("scheme step dx must be positive")
            else:
                scheme_step = value
        if "p0" in kwargs:
            p0 = kwargs["p0"]
        if "alpha0" in kwargs:
            alpha0 = kwargs["alpha0"]
        
        print("~~~~~~~~~~~~")
        print("Parameters :")
        print("~~~~~~~~~~~~")

        for key,value in kwargs.items() :
            print(key,": ",value)

        # Descent speed
        alpha = alpha0

        """
        Do While style loop
        """
        p = p0.copy()
        l_previous = loss_function(p=p) # previously sys.float_info.max
        p_list = [p0] #used to return the points for plotting
        l_list = [l_previous] #used to return the loss function for plotting
        
        discrete_gradient = np.array([ (loss_function(p=[p[0] +scheme_step, p[1]], warp = warp) \
                                          - loss_function(p=[p[0] -scheme_step, p[1]], warp = warp) ) /(2*scheme_step), 
                                           (loss_function(p=[p[0], p[1] +scheme_step], warp = warp) \
                                          - loss_function(p=[p[0], p[1] -scheme_step], warp = warp) ) /(2*scheme_step)]) #beware the indexation
        l = loss_function(p=[p[0] - alpha * discrete_gradient[0],
                            p[1] - alpha * discrete_gradient[1]])
        
        print("discrete_gradient: ",discrete_gradient)
        print("alpha: ",alpha)
        print("l_previous,l: ",l_previous,l)

        while abs(l_previous-l) > epsilon and alpha > epsilon2 : #TODO : test change conditional with or
            
            if l < l_previous : # when loss decreases
                l_previous = l

                # update alpha to value used for l
                p[0] -= alpha*discrete_gradient[0] # beware the indexation
                p[1] -= alpha*discrete_gradient[1]
                
                l_list.append(l_previous)
                p_list.append(p.copy())
                alpha *= 1.1 # hardcoded acceleration

            else :
                alpha *= 0.5 # hardcoded slowing

            discrete_gradient = np.array([ (loss_function(p=[p[0] +scheme_step, p[1]], warp = warp) \
                                          - loss_function(p=[p[0] -scheme_step, p[1]], warp = warp) ) /(2*scheme_step), 
                                           (loss_function(p=[p[0], p[1] +scheme_step], warp = warp) \
                                          - loss_function(p=[p[0], p[1] -scheme_step], warp = warp) ) /(2*scheme_step)]) #beware the indexation
            l = loss_function(p=[p[0] - alpha * discrete_gradient[0],
                                 p[1] - alpha * discrete_gradient[1]]) #beware the indexation
            
            print("discrete_gradient: ",discrete_gradient)
            print("alpha: ",alpha)
            print("l_previous,l: ",l_previous,l)
        
        """end of loop
        """

        print("The translation in y, x coordinates that minimizes our loss function is ", p)
        if plot :
            ax = plt.figure().add_subplot(projection='3d')
            
            np.array([])

            px_loss, py_loss, loss_data = self.import_data(loss_function)
            if len(px_loss) != 0:
                ax.plot_surface(px_loss,py_loss,loss_data,alpha=0.3,rcount=surface_sampling,ccount=surface_sampling)    
            p_list_np = np.array(p_list).transpose()
            l_list_np = np.array(l_list)
            ax.plot(p_list_np[0],p_list_np[1],l_list_np)
            plt.show()

        print("min loss and argmin computation done")
        
        return p, l_list



if __name__ == '__main__' :
    
    utils = Utils_starter_5(Image("images/clean_finger.png"),Image("images/tx_finger.png"))
    
    """Testing greedy_optimization_xy with x translation
    """
    if False:
        p_min, l_list = utils.greedy_optimization_xy(translate_type = "x", plot = True)
        # note: can use a floating step to test floating point translation
    

    """Making smaller images for testing greedy_optimization_xy with xy translation
    """
    if False:
        clean_finger_small = Image("images/clean_finger_small.png")
        tx_finger_small = Image("images/tx_finger.png")
        tx_finger_small._data = cv2.resize(tx_finger_small._data, dsize=clean_finger_small._data.shape[::-1], interpolation=cv2.INTER_CUBIC)
        print(clean_finger_small._data.shape,tx_finger_small._data.shape)

        utils = Utils_starter_5(clean_finger_small,tx_finger_small)
        
        p_min, l_list = utils.greedy_optimization_xy(translate_type = "xy", plot = True)

    """Testing coordinate_descent_optimization_xy with small translation
    """
    if False:
        utils = Utils_starter_5(Image("images/clean_finger.png"),Image("images/tx_finger.png"))
        # utils = Utils_starter_5(Image("images/clean_finger.png"),Image("images/txy_finger.png")) # TODO find params

        ### Choose loss function
        loss_function = utils.loss_function_1
        # loss_function = utils.loss_function_2
        utils.compute_and_plot_loss(show = False, loss_function=loss_function,span="all")



        p, l_list = utils.coordinate_descent_optimization_xy(plot = True, alpha0 = 0.1, epsilon = 100, epsilon2 = 0.001, loss_function=loss_function )

        utils.display_warped(p, utils.get_pix_at_translated, loss_function)

        p, l_list = utils.coordinate_descent_optimization_xy(plot = True, alpha0 = 0.01, epsilon = 10,  epsilon2 = 0.0001,  loss_function=loss_function ) #diverge
        
        utils.display_warped(p, utils.get_pix_at_translated, loss_function)
    
    """Testing coordinate_descent_optimization_xy with blur preoptimisation
    """
    if True:
        
        blur_kernel = 8

        blurred_fixed_finger  = Image("images/clean_finger.png")
        blurred_fixed_finger.blur(blur_kernel)
        blurred_fixed_finger.name = "blurred_fixed_finger"
        blurred_fixed_finger.display()
        blurred_moving_finger = Image("images/tx_finger.png")
        blurred_moving_finger.blur(blur_kernel)
        blurred_moving_finger.name = "blurred_moving_finger"
        blurred_moving_finger.display()

        utils = Utils_starter_5(blurred_fixed_finger,blurred_moving_finger)

        ### Choose loss function
        loss_function = utils.loss_function_1
        # loss_function = utils.loss_function_2
        utils.compute_and_plot_loss(show = False, loss_function=loss_function)#,span="all")


        p0, l_list = utils.coordinate_descent_optimization_xy(plot = True, alpha0 = 0.1, epsilon = 100, epsilon2 = 0.001, loss_function=loss_function )

        utils.display_warped(p0, utils.get_pix_at_translated, loss_function)

        p0, l_list = utils.coordinate_descent_optimization_xy(plot = True, alpha0 = 0.01, epsilon = 10,  epsilon2 = 0.0001,  loss_function=loss_function ) #diverge
        
        utils.display_warped(p0, utils.get_pix_at_translated, loss_function)
        


        utils = Utils_starter_5(Image("images/clean_finger.png"),Image("images/tx_finger.png"))
        # utils = Utils_starter_5(Image("images/clean_finger.png"),Image("images/txy_finger.png")) # TODO find params

        utils.compute_and_plot_loss(show = False, loss_function=loss_function,span="all")


        p, l_list = utils.coordinate_descent_optimization_xy(plot = True, p0=p0, alpha0 = 0.1, epsilon = 100, epsilon2 = 0.001, loss_function=loss_function )

        utils.display_warped(p, utils.get_pix_at_translated, loss_function)

        p, l_list = utils.coordinate_descent_optimization_xy(plot = True, p0=p0, alpha0 = 0.01, epsilon = 10,  epsilon2 = 0.0001,  loss_function=loss_function ) #diverge
        
        utils.display_warped(p, utils.get_pix_at_translated, loss_function)
        









