import numpy as np
import cv2
import matplotlib.pyplot as plt

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
        n,m = self.__img2.data.shape
        # print(x,y,p)
        if 0<=x-p[0]<n and 0<=y-p[1]<m:
            return self.__img2.data[x-p[0]][y-p[1]]
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
        warped_img2 = np.array([[warp(i,j,p) for j in range(self.__img2.data.shape[1])] for i in range(self.__img2.data.shape[0])])
        # plt.imshow(self.__img2.data)
        # plt.show()
        # plt.imshow(warped_img2)
        # plt.show()
        return np.sum((self.__img1.data - warped_img2)**2)
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

    def plot_loss(self, loss_function : callable) :
        print("plot_loss")

        n,m = self.__img1.data.shape
        
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


def test_loss(*args,**kwargs):#(p : list) :
    p = 0
    for params in kwargs:
        p = kwargs['p']

    return np.linalg.norm(np.array(p)-np.array([0,0]))


if __name__ == '__main__' :
    utils = Utils_starter_5(Image("images/clean_finger.png"),Image("images/clean_finger.png"))#,Image("images/tx_finger.png"))
    utils.plot_loss(utils.loss_function_1)
    utils.plot_loss(test_loss)
