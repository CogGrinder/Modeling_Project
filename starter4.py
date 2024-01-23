import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":

    from image import Image

    img = Image("images/clean_finger.png")
    # img.image_hist()
    img.display()
    # print("Image size :", img.n, "x", img.m)
    # img.dilation_grayscale()
    # img.display()
    threshold = img.compute_threshold()
    img.binarize(threshold)
    img.display()
    img.dilation('Cross', 5)
    img.display()
    # print("Image size :", img.n, "x", img.m)
    
            
             