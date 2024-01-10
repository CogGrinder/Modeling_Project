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
        # print(x,y,p)
        if 0<=x-p[0]<n and 0<=y-p[1]<m:
            return self.__img2._data[x-p[0]][y-p[1]]
        else:
            return 1 #white padding TODO check this condition when calculating loss
        #if p[0]<x or p[1]<y:

    # @classmethod
    def loss_function_1(self,**kwargs):#(self, p : list, warp : callable = get_pix_at_translated): #[[int,int,list],float]
        p = 0
        warp = self.get_pix_at_translated #default warp function
        for params in kwargs:
            if params == 'p':
                p = kwargs['p']
                # print("got p")
            if params == 'warp':
                warp = kwargs['warp']
                # print("got warp")
        # print((self.__img2).data)
        warped_img2 = np.array([[warp(i,j,p) for j in range(self.__img2._data.shape[1])] for i in range(self.__img2._data.shape[0])])
        # plt.imshow(self.__img2.data)
        # plt.show()
        # plt.imshow(warped_img2)
        # plt.show()
        return np.sum((self.__img1._data - warped_img2)**2)
        #pass
    
    # @classmethod
    def loss_function_2(self,**kwargs):#(self, p : list, warp : callable = get_pix_at_translated):
        p = 0
        for params in kwargs:
            p = kwargs['p']
        
        """
        warped_img2 = [[warp(self,i,j,p) for j in range(self.__img2.__data.shape[1])] for i in range(self.__img2.__data.shape[0])]
        """
        pass #TODO

    def plot_loss(self, loss_function : callable) : #TODO : vary the ranges and shapes for p, maybe with relation to loss function and image data
        """Function used to greedily calculate all the loss_function returns for a certain range of p

        Args:
            loss_function (callable): function treated as loss function with a parameter p
        """
        print("plot_loss")

        n,m = self.__img2._data.shape
        
        ax = plt.figure().add_subplot(projection='3d')
        #fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        
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
        
        #loss_function(p = [1,4])


    def greedy_optimization_xy(self, **kwargs) : #TODO : vary the ranges and shapes for p, maybe with relation to loss function and image data
        """greedy brute force strategy to find the optimal value of p_x

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
            for i, p_x in enumerate(range(- math.ceil(n/2), math.floor(n/2) + 1)) :
                l = loss_function(p=[0,p_x]) #beware the indexation
                
                if l_min > l : #update the min and argmin
                    l_min = l
                    p_min = [0,p_x]
                
                l_list[i] = l
            print("The translation in x that minimizes our loss function is ", p_min[1])
            if plot :
                p = np.arange(- math.ceil(n/2), math.floor(n/2) + 1)
                plt.plot(p,l_list)
                plt.show()
        elif xy_translate == "xy" :
            l_list  = np.zeros((m+1,n+1)) #make the list bigger to accomodate for all translations
            for j, p_x in enumerate(range(- math.ceil(n/2), math.floor(n/2) + 1)) :
                for i, p_y in enumerate(range(- math.ceil(m/2), math.floor(m/2) + 1)) :
                    l = loss_function(p=[p_y,p_x]) #beware the indexation
                    print([p_y,p_x])
                    
                    if l_min > l :
                        l_min = l
                        p_min = [p_y,p_x]
                    l_list[i][j] = l
            print("The translation in y,x that minimizes our loss function is ", p_min)
            if plot :
                p_x, p_y = np.meshgrid(np.arange(- math.ceil(n/2), math.floor(n/2) + 1),np.arange(- math.ceil(m/2), math.floor(m/2) + 1))
                ax = plt.figure().add_subplot(projection='3d')
                ax.plot_surface(p_x,p_y,l_list)
                plt.show()

        print("min loss and argmin computation done")
        
        return p_min, l_list

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
    # p_min, l_list = utils.greedy_optimization_xy(translate_type = "x", plot = True)
    
    clean_finger_small = Image("images/clean_finger_small.png")
    tx_finger_small = Image("images/tx_finger.png")
    tx_finger_small._data = cv2.resize(tx_finger_small._data, dsize=clean_finger_small._data.shape[::-1], interpolation=cv2.INTER_CUBIC)
    print(clean_finger_small._data.shape,tx_finger_small._data.shape)

    utils = Utils_starter_5(clean_finger_small,tx_finger_small)
    p_min, l_list = utils.greedy_optimization_xy(translate_type = "xy", plot = True)

    #utils.plot_loss(utils.loss_function_1)
    #utils.plot_loss(test_loss)
