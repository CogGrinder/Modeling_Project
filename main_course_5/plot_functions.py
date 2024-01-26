import matplotlib.pyplot as plt
import numpy as np
from main_course_5 import *


def plot_background(utils,loss_function:callable,title,skip=False,shrink=6):
    """Used to generate plot background

    Args:
        utils (Image_registration_tools): Image_registration_tools object containing import_data function
        loss_function (callable): loss function of which to import data

    Returns:
        Figure: axis on which background was plotted
    """
    print("plot_background")
    ax = plt.figure(figsize = (12,12)).add_subplot(projection='3d')
    ax.set_title(title)
            
    px_loss, py_loss, loss_data = utils.import_data(loss_function,skip=skip)
    if len(px_loss) != 0:
        #restrict size of ax
        n,m = utils._fixed_img.data.shape
        span_x = np.ptp(px_loss)
        span_y = np.ptp(py_loss)
        range_x = [\
            - min(n/shrink,span_x),
              min(n/shrink,span_x)]
        range_y = [\
            - min(m/shrink,span_y),
              min(m/shrink,span_y)]
        in_range = np.logical_and(np.logical_and(range_x[0]<px_loss,px_loss<range_x[1]),
                                  np.logical_and(range_y[0]<py_loss,py_loss<range_y[1]) )

        loss_data = np.where(in_range, loss_data, None)

        ax.set_xlim(range_x)
        ax.set_ylim(range_y)

        surface_sampling = max(span_x,span_y)
        ax.plot_surface(px_loss,py_loss,loss_data,alpha=0.3,rcount=surface_sampling,ccount=surface_sampling)    

    
    return ax

def display_all(utils) :
    """Shorthand for displaying all images from Utils_starter_5 object. Used for testing

    Args:
        utils (Utils_starter_5): object containing the images to display
    """
    utils._fixed_img.display()
    utils._moving_img.display()
    return

def display_warped(utils,p,warp:callable,loss_function:callable) :
    """Display image warped by parameter p, with a superimposed effect

    Args:
        utils (Utils_starter_5): object containing the images to display
        p (list): parameter p
        warp (callable): warp function
        loss_function (callable): loss function
    """
    fig, ax = plt.subplots(1,3,figsize = (12,12))
    
    ax[0].set_title("Fixed image")
    ax[1].set_title(f"Warped image, $p=[{p[0]:.2f},{p[1]:.2f}]$")
    ax[2].set_title("Superimposed image")

    ax[0].imshow(utils._fixed_img.data,cmap="Blues")
    ax[2].imshow(utils._fixed_img.data,cmap="Blues")
    i,j = np.meshgrid(np.arange(utils._fixed_img.data.shape[0]),
                        np.arange(utils._fixed_img.data.shape[1]),indexing='ij')

    ax[1].imshow(warp(i,j,p),cmap="Oranges",alpha=1)
    ax[2].imshow(warp(i,j,p),cmap="Oranges",alpha=0.5)
    plt.show()
    return