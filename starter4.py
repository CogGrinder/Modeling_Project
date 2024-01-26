import numpy as np
import matplotlib.pyplot as plt
import cv2


class Starter_4:
    
    @staticmethod
    def image_hist(img):
        """
            Plot the grayscale histogram of the image
        """
        # create the histogram
        hist,bins = np.histogram(img.data.flatten(), bins=256, range=(0, 1))
        # configure and draw the histogram figure
        plt.figure()
        plt.title("Grayscale Histogram")
        plt.xlabel("grayscale value")
        plt.ylabel("pixel count")
        plt.xlim([0.0, 1.0])  # <- named arguments do not work here
        plt.plot(bins[0:-1], hist)  # <- or here
        plt.show()

    @staticmethod
    def binarize(img, threshold):
        """
            Binarize the image(pixels either 1 or 0) given a threshold
        """
        for i in range(img.data.shape[0]):
            for j in range(img.data.shape[1]):
                if img.data[i][j] < threshold:
                    img.data[i][j] = 0
                else:
                    img.data[i][j] = 1

    @staticmethod
    def dilation(img, structuring_element = "Square", size = 3):
        """
            Dilate the binary version of an image
            
            params : 
                structuring element : Defined the shape of the structuring_element(geometrical shape) used to probe the image
                    Possible values : Square, Horizontal Rectangle, Vertical Horizontal, Cross

                size : Defined the size of the structuring element
        """
        if structuring_element=='Square':
            kernel = np.ones((size, size), np.uint8)
            orig_shape = img.data.shape
            pad_width = size - 2 

            # pad the image with pad_width
            image_pad = np.pad(array=img.data, pad_width=pad_width, mode='constant')
            pimg_shape = image_pad.shape
            h_reduce, w_reduce = (pimg_shape[0] - orig_shape[0]), (pimg_shape[1] - orig_shape[1])
            
            # obtain the submatrices according to the size of the kernel
            flat_submatrices = np.array([image_pad[i:(i + size), j:(j + size)]
                                         for i in range(pimg_shape[0] - h_reduce) for j in range(pimg_shape[1] - w_reduce)])
            
            # replace the values either 1 or 0 by dilation condition
            image_dilate = np.array([1 if (i == kernel).any() else 0 for i in flat_submatrices])
            # obtain new matrix whose shape is equal to the original image size
            img.data = image_dilate.reshape(orig_shape)
        
        if structuring_element=='Horizontal Rectangle':
            kernel = np.ones((2, size), np.uint8)
            orig_shape = img.data.shape
            pad_width = size - 1

            # pad the image with pad_width
            image_pad = np.pad(array=img.data, pad_width=pad_width, mode='constant')
            image_pad = image_pad[pad_width-1:image_pad.shape[0]-pad_width, :image_pad.shape[1]-pad_width]
            pimg_shape = image_pad.shape
            # obtain the submatrices according to the size of the kernel
            flat_submatrices = np.array([image_pad[i:(i + 2), j:(j + size)]
                                         for i in range(pimg_shape[0] - 1) for j in range(pimg_shape[1] - size + 1)])
            
            # replace the values either 1 or 0 by dilation condition
            image_dilate = np.array([1 if (i == kernel).any() else 0 for i in flat_submatrices])
            # obtain new matrix whose shape is equal to the original image size
            img.data = image_dilate.reshape(orig_shape)
        
        if structuring_element=='Vertical Rectangle':
            kernel = np.ones((size, 2), np.uint8)
            orig_shape = img.data.shape
            pad_width = size - 1

            # pad the image with pad_width
            image_pad = np.pad(array=img.data, pad_width=pad_width, mode='constant')
            image_pad = image_pad[:image_pad.shape[0]-pad_width, pad_width-1:image_pad.shape[1]-pad_width]
            pimg_shape = image_pad.shape
            
            
            # obtain the submatrices according to the size of the kernel
            flat_submatrices = np.array([image_pad[i:(i + size), j:(j + 2)]
                                         for i in range(pimg_shape[0] - size + 1) for j in range(pimg_shape[1] - 1)])
            
            # replace the values either 1 or 0 by dilation condition
            image_dilate = np.array([1 if (i == kernel).any() else 0 for i in flat_submatrices])
            # obtain new matrix whose shape is equal to the original image size
            img.data = image_dilate.reshape(orig_shape)

        if structuring_element=='Cross':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size,size))
            orig_shape = img.data.shape
            pad_width = size - 2 

            # pad the image with pad_width
            image_pad = np.pad(array=img.data, pad_width=pad_width, mode='constant')
            pimg_shape = image_pad.shape
            h_reduce, w_reduce = (pimg_shape[0] - orig_shape[0]), (pimg_shape[1] - orig_shape[1])
            
            # obtain the submatrices according to the size of the kernel
            flat_submatrices = np.array([image_pad[i:(i + size), j:(j + size)]
                                         for i in range(pimg_shape[0] - h_reduce) for j in range(pimg_shape[1] - w_reduce)])
            # replace the values either 1 or 0 by dilation condition
            image_dilate = np.array([1 if (np.max(i + kernel)== 2) else 0 for i in flat_submatrices])
            # obtain new matrix whose shape is equal to the original image size
            img.data = image_dilate.reshape(orig_shape)
    
    @staticmethod    
    def erosion(img, structuring_element = "Square", size = 3):
        """
            Erode the binary version of an image
            
            params : 
                structuring element : Defined the shape of the structuring_element(geometrical shape) used to probe the image
                    Possible values : Square, Horizontal Rectangle, Vertical Horizontal, Cross

                size : Defined the size of the structuring element
        """
        if structuring_element=='Square':
            kernel = np.ones((size, size), np.uint8)
            orig_shape = img.data.shape
            pad_width = size - 2 

            # pad the image with pad_width
            image_pad = np.pad(array=img.data, pad_width=pad_width, mode='constant', constant_values=1)
            pimg_shape = image_pad.shape
            h_reduce, w_reduce = (pimg_shape[0] - orig_shape[0]), (pimg_shape[1] - orig_shape[1])
            
            # obtain the submatrices according to the size of the kernel
            flat_submatrices = np.array([image_pad[i:(i + size), j:(j + size)]
                                         for i in range(pimg_shape[0] - h_reduce) for j in range(pimg_shape[1] - w_reduce)])
            
            # replace the values either 1 or 0 by erosion condition
            image_erode = np.array([0 if (i != kernel).any() else 1 for i in flat_submatrices])
            # obtain new matrix whose shape is equal to the original image size
            img.data = image_erode.reshape(orig_shape)

        if structuring_element=='Horizontal Rectangle':
            kernel = np.ones((2, size), np.uint8)
            orig_shape = img.data.shape
            pad_width = size - 1

            # pad the image with pad_width
            image_pad = np.pad(array=img.data, pad_width=pad_width, mode='constant', constant_values=1)
            image_pad = image_pad[pad_width-1:image_pad.shape[0]-pad_width, :image_pad.shape[1]-pad_width]
            pimg_shape = image_pad.shape
            # obtain the submatrices according to the size of the kernel
            flat_submatrices = np.array([image_pad[i:(i + 2), j:(j + size)]
                                         for i in range(pimg_shape[0] - 1) for j in range(pimg_shape[1] - size + 1)])
            
            # replace the values either 1 or 0 by dilation condition
            image_erode = np.array([0 if (i != kernel).any() else 1 for i in flat_submatrices])
            # obtain new matrix whose shape is equal to the original image size
            img.data = image_erode.reshape(orig_shape)
        
        if structuring_element=='Vertical Rectangle':
            kernel = np.ones((size, 2), np.uint8)
            orig_shape = img.data.shape
            pad_width = size - 1

            # pad the image with pad_width
            image_pad = np.pad(array=img.data, pad_width=pad_width, mode='constant', constant_values=1)
            image_pad = image_pad[:image_pad.shape[0]-pad_width, pad_width-1:image_pad.shape[1]-pad_width]
            pimg_shape = image_pad.shape
            
            
            # obtain the submatrices according to the size of the kernel
            flat_submatrices = np.array([image_pad[i:(i + size), j:(j + 2)]
                                         for i in range(pimg_shape[0] - size + 1) for j in range(pimg_shape[1] - 1)])
            
            # replace the values either 1 or 0 by dilation condition
            image_erode = np.array([0 if (i != kernel).any() else 1 for i in flat_submatrices])
            # obtain new matrix whose shape is equal to the original image size
            img.data = image_erode.reshape(orig_shape)

        if structuring_element=='Cross':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size,size))
            orig_shape = img.data.shape
            pad_width = size - 2 

            # pad the image with pad_width
            image_pad = np.pad(array=img.data, pad_width=pad_width, mode='constant', constant_values=1)
            pimg_shape = image_pad.shape
            h_reduce, w_reduce = (pimg_shape[0] - orig_shape[0]), (pimg_shape[1] - orig_shape[1])
            
            # obtain the submatrices according to the size of the kernel
            flat_submatrices = np.array([image_pad[i:(i + size), j:(j + size)]
                                         for i in range(pimg_shape[0] - h_reduce) for j in range(pimg_shape[1] - w_reduce)])
            # replace the values either 1 or 0 by dilation condition
            image_erode = np.array([1 if (np.sum(kernel + i) >= 2*np.sum(kernel)) else 0 for i in flat_submatrices])
            # obtain new matrix whose shape is equal to the original image size
            img.data = image_erode.reshape(orig_shape)
        
    




if __name__ == "__main__":

    from image import Image

    img = Image("images/clean_finger.png")
    img.image_hist()
    threshold = img.compute_threshold()
    img.binarize(threshold)
    img.display()
    img.erosion()
    img.display()
    
            
             