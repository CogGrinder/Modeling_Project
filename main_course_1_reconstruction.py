import numpy as np
import matplotlib.pyplot as plt

# from image import Image
# from starter2 import Starter_2

class Main_Course_1_Reconstruction:

    @staticmethod
    def frobenius_dist(A, B):
        """
            Method that computes the Frobenius distance between two matrices A and B

            Parameters:
                - A, B: two square matrices of size n x n

            Returns:
                - A float value: dist(A, B)
        """
        # https://math.stackexchange.com/questions/507742/distance-similarity-between-two-matrices

        # Frobenius distance
        return np.sqrt(np.trace(np.transpose(A-B) * (A-B)))

        # Euclidean distance
        # dist = 0
        # for i in range(A.shape[0]):
        #     for j in range(A.shape[1]):
        #         dist += (A[i][j] - B[i][j]) ** 2
        # return np.sqrt(dist)
    
    @staticmethod
    def restauration(img, patches, mask, p_size=9):
        """
            Apply the given method in the course that reconstruct losses in an image from patches cropped
            randomly. The loss is simulated using a binary mask where True pixels are those which are missing.
            For each of those (x,y) True pixels, we restaure its intensity by computing the nearest patch cropped with
            respect to the patch which has (x,y) as its center.

            Parameters:
                - img: Instance of Image class, image to be restaured
                - patches: list of several cropped patches
                - mask: np.array(), binary mask
                - p_size: odd size of each patches, default value: 9

            Returns:
                - img: The same instance of Image class where its attribute data contains the restaured image.
        """

        # offset related to the center pixel coordinates of a patch
        offset = p_size // 2
        
        # For each pixels in the mask
        for x in range(1, mask.shape[0]):
            for y in range(1, mask.shape[1]):

                # Process only true pixels
                if mask[x][y] == True:
                    # print("Processing (", x, ",", y, ") pixel")

                    # Crop the surrounding patch p in the fingerprint at coordinates (x,y)
                    top_left_coord_x, top_left_coord_y = x - offset, y - offset
                    p = img.data[top_left_coord_x : top_left_coord_x + p_size, top_left_coord_y : top_left_coord_y + p_size]
                    # print(p)

                    # Compute the Frobenius distance d(p, P) for all P in patches
                    # and deduce the closest patch p_s (minimum distance to p)
                    p_s = np.array([])
                    dist_min = 1000
                    for P in patches:

                        # Get the pixels from P which are not in the mask
                        # i.e. coords (i,j) s.t. mask[i][j] == False
                        # P_outmask = []
                        # for i in range(P.shape[0]):
                        #     for j in range(P.shape[1]):
                        #         if P[i][j] == False:
                        #             P_outmask.append(P[i][j])
                        # P_outmask = np.array(P_outmask)

                        dist = Main_Course_1_Reconstruction.frobenius_dist(p, P)
                        if dist < dist_min:
                            dist_min = dist
                            p_s = P

                    # Copy paste the middle pixel value of P into the fingerprint image 
                    # at coordinates (x,y)
                    img.data[x][y] = p_s[offset][offset]

        # Return the restaured image
        return img

if __name__ == "__main__":

    from image import Image
    
    # Open the weak finger image
    img = Image("images/weak_finger.png")
    img2 = Image("images/clean_finger.png")
    # img.display()

    # Crop the image into several patches
    patch_size = 3
    patches = img.crop_patches(2000, patch_size)

    # Plot the patches for debug purpose
    # for patch in patches:
    #     plt.imshow(patch, cmap='gray', vmin=0, vmax=1)
    #     plt.show()

    # Create a mask with a small square of True values (False otherwise)
    mask_size = 60
    mask = np.full((img2.n, img2.m), False)
    mask[150 : 150 + mask_size, 100 : 100 + mask_size] = True

    # Apply the restauration algorithm onto the image
    img2 = Main_Course_1_Reconstruction.restauration(img2, patches, mask, patch_size)
    img2.display()

    

    
