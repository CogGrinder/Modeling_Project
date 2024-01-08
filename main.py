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

img = Image("images/warp1_finger.png")
# img.display()

img.create_rectangle((50,50),100,50,"white")
img.create_rectangle((175,125),25,50,"black")
# img.symmetry(0)
# img.symmetry(1)
# img.rotate(45,(0.55,0.6))

# img.save("warp1_finger_edit.png")
# img.display()

# img.rotate(-45,(0.5,0.5))
# img.display()

# img.rotate2(45,(0.55,0.6))
img.rotate_translate(-28, (100, 100), (0, 0))

img.save("warp1_finger_edit.png")
#img.display()

img.rotate2(-45,(0.5,0.5))
#img.display()


# Testing for convolution
# img = Image("images/warp1_finger.png")

# kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
# img.conv_2d(kernel)
# img.display()

# img = Image("images/warp1_finger.png")

#testing with black image
img.test_black(5)


# kernel = np.ones((3,3))

# img.conv_2d(kernel)
# img.display()

# Testing 2D FFT
img = Image("images/moist_finger.png")
img_fft = img.fft_2d()
plt.imshow(abs(img_fft))
plt.show()

img_ifft = img.ifft_2d(img_fft)
plt.imshow(abs(img_ifft), cmap="gray")
plt.show()