import numpy as np
import cv2
import matplotlib.pyplot as plt

import math
import sys
import os
sys.path.append(os.getcwd()) #to access current working directory files easily
from image import Image

from utils_starter_5 import Utils_starter_5

original_background = 1 # set as 0.5 to reveal original background in output


def test_loss(*args,**kwargs): #(p : list) :
    """Used as a dummy function to test the plot_loss function

    Args:
        p: coordinates for evaluating function
    Returns:
        float : cone function ie distance to [0,0]
    """
    p = 0
    center = [0,0]
    for params in kwargs:
        p = kwargs['p']

    return np.linalg.norm(np.array(p)-np.array(center))


def assert_image_dots(image,list_of_points) :
    """checks that list_of_points, a list of black dots, represents the matrix of the image
    """
    # assert_matrix = np.ones(image.data.shape)
    for point in list_of_points:
        assert image.data[point] <= 0.5 #accepts dark grey
    # print(image.data/original_background,assert_matrix,sep="\n")
    # assert (image.data/original_background == assert_matrix).all()

def test_translation():
    """Test on small matrixes that the translation does what it is supposed to
    """
    #shape = (6,6)
    fixed1  = Image( original_background*\
      np.array([[1,1,1,1,1,1],
                [1,1,1,1,1,1],
                [1,1,1,1,1,1],
                [1,0,1,0,1,1],
                [1,1,1,0,1,1],
                [1,1,1,1,0,1]])
    )
    assert_image_dots(fixed1,[(3,1),
                              (3,3),
                              (4,3),
                              (5,4)])

    moving1 = Image( original_background*\
      np.array([[1,1,1,1,1,1],
                [0,1,0,1,1,1],
                [1,1,0,1,1,1],
                [1,1,1,0,1,1],
                [1,1,1,1,1,1],
                [1,1,1,1,1,1]])

    )
    assert_image_dots(moving1,[(1,0),
                               (1,2),
                               (2,2),
                               (3,3)])

    #shape (5,5)
    fixed2  = Image( original_background*\
      np.array([[ 1,.5,1,1,1],
                [ 1,1, 1,1,1],
                [.5,1, 1,1,1],
                [ 1,.5,1,1,1],
                [ 1,1, 1,1,1]])
    )
    assert_image_dots(fixed2,[(2,0)])
    moving2 = Image( original_background*\
      np.array([[1,1,1, 1, 1],
                [1,1,1, 1,.5],
                [1,1,1, 1, 1],
                [1,1,1,.5, 1],
                [1,1,1, 1,.5]])
    )
    assert_image_dots(moving2,[(3,3)])


    print("Test 1")
    print("fixed1:",fixed1.data,sep="\n")
    print("moving1:",moving1.data,sep="\n")
    utils = Utils_starter_5(fixed1,moving1)

    i,j = np.meshgrid(np.arange(6),np.arange(6),indexing="ij") #warning, modify here

    print("moving1 translated by p:")
    p=(1.9,0)
    translated_moving1 = utils.get_pix_at_translated(i,j,p=p)
    print(p,translated_moving1,sep="\n")
    
    p=(0,0.1)
    translated_moving1 = utils.get_pix_at_translated(i,j,p=p)
    print(p,translated_moving1,sep="\n")

    p=(1.9,0.1)
    translated_moving1 = utils.get_pix_at_translated(i,j,p=p)
    print(p,translated_moving1,sep="\n")
    
    p=(2,1)
    translated_moving1 = utils.get_pix_at_translated(i,j,p=p)
    print(p,translated_moving1,sep="\n")
    
    # expected value is fixed1 matrix
    #TODO translated_moving1[3,1] = 0 # this pixel was on the border therefore it was ignored by the filter
    assert (translated_moving1==fixed1.data).all()


    print("Test 2")
    print("fixed2:",fixed2.data,sep="\n")
    print("moving2:",moving2.data,sep="\n")
    utils = Utils_starter_5(fixed2,moving2)

    i,j = np.meshgrid(np.arange(5),np.arange(5),indexing="ij")
    
    print("moving2 translated by p:")
    p=(-0.1,0)
    translated_moving2 = utils.get_pix_at_translated(i,j,p=p)
    print(p,translated_moving2,sep="\n")
    
    p=(0,-2.9)
    translated_moving2 = utils.get_pix_at_translated(i,j,p=p)
    print(p,translated_moving2,sep="\n")

    p=(-1,-3) # should be (-1,-3)
    translated_moving2 = utils.get_pix_at_translated(i,j,p=p)
    print(p,translated_moving2,sep="\n")
    # expected value is fixed2 matrix
    assert (translated_moving2==fixed2.data).all()

def test_image_translate_display():
    pass
    

## Tests
if __name__ == '__main__' :
    
    """Testing compute_and_plot_loss
    """
    utils = Utils_starter_5(Image(np.ones((10,10))),Image(np.ones((10,10))) )
    #TODO uncomment utils.compute_and_plot_loss(loss_function=test_loss, span="all",save=False)
    #Expected : cone shape

    test_translation() # works 22/01/2024
    
    
    # utils.display_warped()
    

