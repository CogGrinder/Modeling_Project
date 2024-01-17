import numpy as np
import matplotlib.pyplot as plt

# from image import Image
# from starter2 import Starter_2

class Main_Course_1_Reconstruction:
    pass
    
    
if __name__ == "__main__":

    from image import Image
     
    img = Image("images/weak_finger.png")

    patches = img.crop_patches(10)
    for patch in patches:
        plt.imshow(patch, cmap='gray', vmin=0, vmax=1)
        plt.show()

    # create a mask with a small square 30x30 of True values (False otherwise)
    mask = np.full((img.n, img.m), False)
    mask[150 : 150 + 30, 100 : 100 + 30] = True


    # img.display()

    

    
