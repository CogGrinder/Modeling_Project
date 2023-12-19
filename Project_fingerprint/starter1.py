import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def maximum_value(img):
    ''' Return the maximum value of the pixels of the image im '''
    maxi = max(img[0])
    for i in range(1, len(img)):
        if max(img[i]) > maxi:
            maxi = max(img[i])
    return maxi

def minimum_value(img):
    ''' Return the minimum value of the pixels of the image im '''
    mini = min(img[0])
    for i in range(1, len(img)):
        if min(img[i]) < mini:
            mini = min(img[i])
    return mini


def normalize(img):
    ''' Create and return a normalized version of the image img
        The minimum value of img become 0 and its maximum value become 1 '''
    img_nm = np.copy(img)
    maxi = maximum_value(img)
    mini = minimum_value(img)
    return (img_nm - mini) / (maxi - mini)

def denormalize(img, original_min=-1, original_max=-1):
    ''' Create and return the non normalized version (with pixels value living in [0;255], not in [0;1]) of the normalized image img
        Parameters original_min and original_max allows to not lose any information in comparison with the original version
        Those parameters are initially set to -1, which means that teh denormalized image will pixel values in the full interval [0;255]
        (0 will be transformed into 0, 1 into 255) '''
    img_dnm = np.copy(img)
    if original_max == -1 and original_min == -1:
        return img_dnm * 255 
    else:
        return img_dnm * (original_max - original_min) + original_min



def create_white_rect(img, origin, width, lenght, value=1):
    ''' Create and return a version of the image, where has been included a rectangle of origin (top-left corner) origin (tuple of 2 int values, coordinates
    of the origin), of width witdth and lenght lenght, with pixel value value (normalized) '''
    img_new = np.copy(img)
    for w in range(width):
        for l in range(lenght):
            img_new[origin[0]+w][origin[1]+l] = value
    return img_new

def symetry_yaxis(img):
    ''' Create and return the symetric of img with respect to the y axis '''
    img_sym_y = np.zeros_like(img)
    n = len(img)
    m = len(img[0])
    for x in range(n):
        for y in range(m):
            img_sym_y[x][y] = img[x][(m-1)-y]

    return img_sym_y

def symetry_xaxis(img):
    ''' Create and return the symetric of img with respect to the x axis '''
    img_sym_x = np.zeros_like(img)
    n = len(img)
    m = len(img[0])
    for x in range(n):
        for y in range(m):
            img_sym_x[x][y] = img[(n-1)-x][y]

    return img_sym_x
        


def saving_image(img, file_name, directory='./'):
    ''' Save the image img under the name {file_name}.png (no need to add the extension .png at the end of the file name) in the directory directory '''
    # Ensure it is in the range [0, 255] and of type int
    if maximum_value(img) > 255:
        print("Warning : max value of the image", file_name," above 255 when trying to save it")
    if minimum_value(img) < 0:
        print("Warning : min value of the image", file_name," below 0 when trying to save it")
    # Force the type to be int (truncature) and bring the eventual values out of [0;255] to 0 if the value is <0, or to 255 if value is >255
    img = np.clip(img, 0, 255).astype(np.uint8)
    # Save the processed image as a PNG file
    cv2.imwrite(directory+file_name+'.png', img)

def print_image(img):
    ''' Print the image img '''
    plt.imshow(img, cmap='gray')
    plt.show()


# reads image 'opencv-logo.png' as grayscale

dir = os.path.abspath("images")

print(type(cv2.imread))
img = cv2.imread(dir + "\\moist_finger_small.png", 0)
print(type(img))
# print(img.shape)
# print_image(img)

original_max = maximum_value(img)
original_min = minimum_value(img)
img_nm = normalize(img)
# img_nm_rect = create_white_rect(img_nm, (100, 75), 10, 20, 0)
img_nm_sym_x = symetry_xaxis(img_nm)
img_dnm_sym_x = denormalize(img_nm_sym_x, original_min, original_max)

# print_image(img_dnm_sym_x)


# saving_image(img_dnm, "modified_image", "images/")
# img_dnm = cv2.imread('images/modified_image.png', 0) 
# print_image(img_dnm)


####starter_2
def img_rotation(img, p, center_normalized):
    ''' 2D rotation of the img matrix in a p angle '''
    center = np.array([img.shape[0]//(1/center_normalized[0]),img.shape[1]//(1/center_normalized[1])]).astype(int)
    print(center)
    p_radian = p * np.pi/180
    n, m = img.shape
    rotation_matrix = np.array([[np.cos(p_radian), -np.sin(p_radian)],[np.sin(p_radian), np.cos(p_radian)]])
    
    n_rotated = n
    m_rotated = m
    offset_x = 0
    offset_y = 0
    for i_c in [0,n-1] :
        for j_c in [0,m-1] :
            i = i_c-center[0]
            j = j_c-center[1]
            index_centered = np.array([i, j])
            rotated_ind_centered = np.floor(rotation_matrix @ index_centered).astype(int)
            rotated_ind = rotated_ind_centered + center
            
            if rotated_ind[0]>n_rotated :
                n_rotated = rotated_ind[0]
            elif rotated_ind[0] < -offset_x:
                offset_x = -rotated_ind[0]
            
            if rotated_ind[1]>m_rotated :
                m_rotated = rotated_ind[1]
            elif rotated_ind[1] < -offset_y:
                offset_y = -rotated_ind[1]

    rotated_img = np.ones((n_rotated + offset_x, m_rotated + offset_y))
    print(offset_x,offset_y)
    print(n_rotated,m_rotated)



    for i_c in range(0,n_rotated + offset_x):
        for j_c in range(0,m_rotated + offset_y):
            i = i_c-center[0] - offset_x
            j = j_c-center[1] - offset_y
            index_centered = np.array([i, j])
            rotated_ind = np.floor(rotation_matrix @ index_centered).astype(int)
            # print(rotated_ind)
            if (0 <= rotated_ind[0] + center[0] < n) and (0 <=rotated_ind[1] + center[1]  < m):
                rotated_img[i_c][j_c] = img[rotated_ind[0] + center[0] ][rotated_ind[1] + center[1]]
    return rotated_img

# print_image(img)
rotated_img_nm = img_rotation(img_nm, 70, (0.5,0.5))
rotated_img = denormalize(rotated_img_nm, original_min, original_max)
print_image(rotated_img)

def img_rotation(img, p):
    p_radian = p * np.pi/180
    rotation_matrix = np.array([[np.cos(p_radian), -np.sin(p_radian)],[np.sin(p_radian), np.cos(p_radian)]])#.transpose()
    n, m = img.shape 
    rotated_img = np.ones((2*n, 2*m))
    for i in range(n):
        for j in range(m):
            initial_ind = np.array([i, j]) 
            rotated_ind = np.dot(rotation_matrix, initial_ind).astype(np.uint8)
            if (0 <= rotated_ind[0] < n) and (0 <=rotated_ind[1] < m)   :
                rotated_img[i + n//2][j + m//2] = img[rotated_ind[0]][rotated_ind[1]]
            else:
                rotated_img[i][j] = 0
    return rotated_img


rotated_img_nm = img_rotation(img_nm, 90)
rotated_img = denormalize(rotated_img_nm, original_min, original_max)
#print_image(rotated_img)


