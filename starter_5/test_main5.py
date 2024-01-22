import numpy as np
import cv2
import matplotlib.pyplot as plt

import math
import sys
import os
sys.path.append(os.getcwd()) #to access current working directory files easily
from image import Image

from utils_starter_5 import Utils_starter_5


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


original_background = 1 # set as 0.5 to reveal original background in output

def assert_image_dots(image,list_of_points) :
    """checks that list_of_points, a list of black dots, represents the matrix of the image
    """
    assert_matrix = np.ones(image.data.shape)
    for point in list_of_points:
        assert_matrix[point] = 0
    assert (image.data/original_background == assert_matrix).all()

def test_translation():
    shape = (5,5)
    fixed1  = Image( original_background*\
      np.array([[1,1,1,1,1],
                [1,1,1,1,1],
                [1,0,1,0,1],
                [1,1,1,0,1],
                [1,1,1,1,0]])
    )
    assert_image_dots(fixed1,[(2,1),
                              (2,3),
                              (3,3),
                              (4,4)])

    moving1 = Image( original_background*\
      np.array([[1,1,1,1,1],
                [0,1,0,1,1],
                [1,1,0,1,1],
                [1,1,1,0,1],
                [1,1,1,1,1]])
    )
    assert_image_dots(moving1,[(1,0),
                               (1,2),
                               (2,2),
                               (3,3)])

    fixed2  = Image( original_background*\
      np.array([[1,1,1,1,1],
                [1,1,1,1,1],
                [1,0,1,1,1],
                [1,1,1,1,1],
                [1,1,1,1,1]])
    )
    assert_image_dots(fixed2,[(2,1)])
    moving2 = Image( original_background*\
      np.array([[1,1,1,1,1],
                [1,1,1,1,1],
                [1,1,1,1,1],
                [1,1,1,1,0],
                [1,1,1,1,1]])
    )
    assert_image_dots(moving2,[(3,4)])


    print("Test 1")
    utils = Utils_starter_5(fixed1,moving1)

    i,j = np.meshgrid(np.arange(5),np.arange(5),indexing="ij")
    print(i,j,sep="\n")
    print("fixed1:",fixed1.data,sep="\n")
    print("moving1:",moving1.data,sep="\n")

    print("moving1 translated")
    p=(0.9,0)
    translated_moving1 = utils.get_pix_at_translated(i,j,p=p)
    print(p,translated_moving1,sep="\n")
    
    p=(0,0.1)
    translated_moving1 = utils.get_pix_at_translated(i,j,p=p)
    print(p,translated_moving1,sep="\n")
    
    p=(1,1)
    translated_moving1 = utils.get_pix_at_translated(i,j,p=p)
    print(p,translated_moving1,sep="\n")
    
    # expected value is fixed1 matrix
    # assert (translated_moving1==fixed1.data).all()


    print("Test 2")
    utils = Utils_starter_5(fixed2,moving2)

    i,j = np.meshgrid(np.arange(5),np.arange(5),indexing="ij")
    translated_moving2 = utils.get_pix_at_translated(i,j,p=(3,1)) # should be (3,1)
    print(fixed2.data)
    print(translated_moving2)
    
    # expected value is fixed2 matrix
    assert (translated_moving2==fixed2.data).all()

def test_image_translate_display()
    

## Tests
if __name__ == '__main__' :
    
    # utils = Utils_starter_5(Image(np.ones((10,10))),Image(np.ones((10,10))) )
    """Testing compute_and_plot_loss
    """
    # utils.compute_and_plot_loss(loss_function=test_loss, span="all",save=False)
    #Expected : cone shape

    test_translation()
    
    
    utils.display_warped()
    

