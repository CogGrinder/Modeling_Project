import numpy as np
import cv2
import matplotlib.pyplot as plt

import math
import sys
import os
sys.path.append(os.getcwd())
from image import Image

class Utils_starter_5:
    def __init__(self, img1 : Image, img2 : Image):
        self.__img1 = img1 #fixed
        self.__img2 = img2 #moving
        
        return

    def get_pix_at_translated(self, x : int, y : int, p : list):
        n,m = self.__img2._data.shape
        int_px = math.floor(p[0]) #beware indexation
        int_py = math.floor(p[1]) #beware indexation
        if 1<=x-int_px<n and 1<=y-int_py<m:
            float_px = p[0] - int_px #TODO probably bugs here with indexation
            float_py = p[1] - int_py
            return    (1-float_px)*(1-float_py) *self.__img2._data[x-int_px][y-int_py] \
                    + float_px*(1-float_py)     *self.__img2._data[x-int_px-1][y-int_py] \
                    + (1-float_px)*float_py     *self.__img2._data[x -int_px][y-int_py-1] \
                    + float_px*float_py         *self.__img2._data[x -int_px-1][y-int_py-1] #TODO probably bugs here with indexation

        else:
            return 1 #white padding TODO check this condition when calculating loss

    def loss_function_1(self,**kwargs):#(self, p : list, warp : callable = get_pix_at_translated): #[[int,int,list],float]
        p = 0
        warp = self.get_pix_at_translated #default warp function
        for params in kwargs:
            if params == 'p':
                p = kwargs['p']
            if params == 'warp':
                warp = kwargs['warp']
        warped_img2 = np.array([[warp(i,j,p) for j in range(self.__img2._data.shape[1])] for i in range(self.__img2._data.shape[0])])
        return np.sum((self.__img1._data - warped_img2)**2)
    
    def loss_function_2(self,**kwargs):#(self, p : list, warp : callable = get_pix_at_translated):
        p = 0
        for params in kwargs:
            p = kwargs['p']
        
        """
        warped_img2 = [[warp(self,i,j,p) for j in range(self.__img2.__data.shape[1])] for i in range(self.__img2.__data.shape[0])]
        """
        pass #TODO

    def test_plot_loss(self, loss_function : callable) : #TODO : vary the ranges and shapes for p, maybe with relation to loss function and image data
        """Function used to greedily calculate all the loss_function returns for a certain range of p

        Args:
            loss_function (callable): function treated as loss function with a parameter p
        """
        print("test_plot_loss")

        n,m = self.__img2._data.shape
        
        ax = plt.figure().add_subplot(projection='3d')
        
        translate_span_x = 1
        translate_span_y = m//3

        px, py = np.meshgrid(np.linspace(-translate_span_x,translate_span_x, translate_span_x * 2 + 1).astype(int),np.linspace(-translate_span_y,translate_span_y, translate_span_y * 2 + 1).astype(int))
        #arbitrary range for p, beware the endpoint is True

        loss_grid = np.zeros(px.shape)
        for i in range(px.shape[0]):
            for j in range(px.shape[1]):
                p = [px[i,j],py[i,j]]
                print(p)
                loss_grid[i][j] = loss_function(p = p, warp = self.get_pix_at_translated ) #pass the px and py values
        
        print("loss computation done")

        ax.plot_surface(px,py,loss_grid)
        plt.show()

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

        for key,value in kwargs.items():
            if key == "loss_function":
                if not (type(value) is callable):
                    raise TypeError("loss_function not a function")
                else :
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
        n,m = self.__img2._data.shape

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
                p_x, p_y = np.meshgrid(np.arange(- math.ceil(n/2), math.floor(n/2) + 1),np.arange(- math.ceil(m/2), math.floor(m/2) + 1))
                ax = plt.figure().add_subplot(projection='3d')
                ax.plot_surface(p_x,p_y,l_list)
                plt.show()

        print("min loss and argmin computation done")

        return p_min, l_list


    def coordinate_descent_optimization_xy(self, **kwargs) : #TODO : vary the ranges and shapes for p, maybe with relation to loss function and image data
        """non differentiable coordinate descent strategy to find the optimal value of [p_x,p_y]

        Args:
            loss_function (callable): function treated as loss function with a parameter p
            kwargs :
                plot : default is False - choose wether to plot the loss function
                loss_function : callable function which takes a parameter p
                epsilon : stopping level for our loss function decrease
                p0 : initial p parameter for loss function
                alpha0 : initial percentage (in direct multiplicative factor form) for adjustment of p


        """
        print("coordinate_descent_optimization_xy")
        
        p0 = [0,0]
        alpha0 = 0.1
        plot = False
        loss_function = self.loss_function_1
        epsilon = 1000 #arbitrary default

        for key,value in kwargs.items():
            if key == "loss_function":
                if not (type(value) is callable):
                    raise TypeError("loss_function not a function")
                else :
                    loss_function = value
            elif key == "plot":
                if not (type(value) is bool):
                    print(type(value))
                    raise TypeError("plot must be a bool")
                else :
                    plot = value
            elif key == "epsilon":
                if value<0:
                    raise ValueError("epsilon must be positive")
                else:
                    epsilon = value
            elif key == "p0":
                p0 = value
            elif key == "alpha0":
                alpha0 = value
        
        print("~~~~~~~~~~~~")
        print("Parameters :")
        print("~~~~~~~~~~~~")

        for key,value in kwargs.items() :
            print(key,": ",value)

        
        alpha = alpha0
        l_previous   = sys.float_info.max

        """Do While style loop
        """
        p = p0.copy()
        l = loss_function(p=p)
        p_list = [p.copy()] #used to return the points for plotting
        l_list = [l] #used to return the loss function for plotting
        discrete_gradient = [ (loss_function(p=[p[0] + 1, p[1]]) - (loss_function(p=[p[0] - 1, p[1]])) ) /2, 
                            (loss_function(p=[p[0], p[1] + 1]) - (loss_function(p=[p[0], p[1] - 1])) ) /2] #beware the indexation
        print(discrete_gradient)
        print(alpha)

        while abs(l_previous-l) > epsilon : #TODO : change conditional
            print(l_previous,l)
            
            if l < l_previous : #updates l and alpha
                l_previous = l
                p[0] -= alpha*discrete_gradient[0] #beware the indexation
                p[1] -= alpha*discrete_gradient[1]
                
                l_list.append(l_previous)
                p_list.append(p.copy())
                alpha *= 1.1

            else :
                alpha *= 0.5
            
            discrete_gradient = [ (loss_function(p=[p[0] + 1, p[1]]) - (loss_function(p=[p[0] - 1, p[1]])) ) /2, 
                                (loss_function(p=[p[0], p[1] + 1]) - (loss_function(p=[p[0], p[1] - 1])) ) /2] #beware the indexation
            print(discrete_gradient)
            print(alpha)

            l = loss_function(p=[p[0] - alpha * discrete_gradient[0],
                                 p[1] - alpha * discrete_gradient[1]]) #beware the indexation
        
        """end of loop
        """

        print("The translation in y, x coordinates that minimizes our loss function is ", p)
        if plot :
            ax = plt.figure().add_subplot(projection='3d')
            p_list_np = np.array(p_list).transpose()
            l_list_np = np.array(l_list)
            ax.plot(p_list_np[0],p_list_np[1],l_list_np)
            plt.show()

        print("min loss and argmin computation done")
        
        return p, l_list

def test_loss(*args,**kwargs):#(p : list) :
    """Used as a dummy function to test the plot_loss function

    Args:
        p: coordinates for evaluating function
    Returns:
        float : cone function ie distance to [0,0]
    """
    p = 0
    for params in kwargs:
        p = kwargs['p']

    return np.linalg.norm(np.array(p)-np.array([0,0]))


if __name__ == '__main__' :
    
    utils = Utils_starter_5(Image("images/clean_finger.png"),Image("images/tx_finger.png"))
    
    """Testing loss_function in a test set of translations
    """
    if False:
        utils.test_plot_loss(utils.loss_function_1)
        utils.test_plot_loss(test_loss)
    
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
    if True:
        utils = Utils_starter_5(Image("images/clean_finger.png"),Image("images/tx_finger.png"))

        p, l_list = utils.coordinate_descent_optimization_xy(plot = True)








