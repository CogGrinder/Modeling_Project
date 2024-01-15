import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":

    from image import Image

    img = Image("images/clean_finger.png")
    img.display()
    print("Image size :", img.n, "x", img.m)

    threshold = img.compute_threshold()
    img.binarize(threshold)
    img.erosion('Square', 3)

    img.display()
    print("Image size :", img.n, "x", img.m)
    
            
             