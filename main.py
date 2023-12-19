import numpy as np
import cv2
import matplotlib.pyplot as plt

from image import Image

# # reads image 'opencv-logo.png' as grayscale
# img = cv2.imread('images/moist_finger.png', 0) 
# print(img.shape)

# # Normalization of the image
# min = np.min(img)
# max = np.max(img)
# normalized_img = (img - min)/(max - min)
# print(normalized_img)

# # Simulating noise with black and white squares
# noisy_img = normalized_img
# noisy_img[50:150, 50:100] = 1
# noisy_img[175:200, 125:175] = 0

# # Converting back to pixel image [0, 255] range
# noisy_img = noisy_img * 256
# noisy_img = noisy_img.astype(int)

# # write image 'noisy_img.png'
# cv2.imwrite('noisy_img.png', noisy_img) 

# noisy_img_symm = np.flip(noisy_img, axis=0)


# plt.imshow(noisy_img_symm, cmap='gray')
# plt.show()

img = Image("images/moist_finger.png")
# img.display()

img.create_rectangle((50,50),100,50,"white")
img.create_rectangle((175,125),25,50,"black")
img.symmetry(0)
img.symmetry(1)

img.save("noisy_img2.png")
img.display()