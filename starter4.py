import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":

    from image import Image

    img = Image("images/moist_finger.png")
    # img.image_hist()
    img.display()
    print("Image size :", img.n, "x", img.m)
    img.dilation_grayscale()
    img.display()
    # img.binarize(threshold)
    # img.erosion('Square', 3)

    # img.display()
    print("Image size :", img.n, "x", img.m)
    
            
             