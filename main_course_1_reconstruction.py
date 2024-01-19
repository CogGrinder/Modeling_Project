import numpy as np
import matplotlib.pyplot as plt

# from image import Image
# from starter2 import Starter_2

class Main_Course_1_Reconstruction:

    @staticmethod
    def eucl_dist_matrices(A, B):
        """
            Method that computes the euclidean distance between two matrices A and B

            Parameters:
                - A, B: two square matrices of size n x n

            Returns:
                - A float value: dist(A, B)
        """
        # https://math.stackexchange.com/questions/507742/distance-similarity-between-two-matrices

        # Frobenius distance
        # return np.sqrt(np.trace(np.transpose(A-B) * (A-B)))

        # Euclidean distance
        dist = 0
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                dist += (A[i][j] - B[i][j]) ** 2
        return np.sqrt(dist)
    
    @staticmethod
    def restauration(img, patches, mask):
        
        # For each pixels in the mask
        for x in range(1, mask.shape[0]):
            for y in range(1, mask.shape[1]):

                # Process only true pixels
                if mask[x][y] == True:

                    # Crop the surrounding patch p in the fingerprint at coordinates (x,y)
                    top_left_coord_x, top_left_coord_y = x - 4, y - 4
                    p = img.data[top_left_coord_x : top_left_coord_x + 9, top_left_coord_y : top_left_coord_y + 9]
                    # print(p)

                    # Compute the euclidean distance d(p, P) for all P in patches
                    # and deduce the closest patch p_s (minimum distance to p)
                    p_s = np.array([])
                    dist_min = 1000
                    for P in patches:

                        # Get the pixels from P which are not in the mask
                        # i.e. coords (i,j) s.t. mask[i][j] == False
                        P_outmask = []
                        for i in range(P.shape[0]):
                            for j in range(P.shape[1]):
                                if P[i][j] == False:
                                    P_outmask.append(P[i][j])
                        P_outmask = np.array(P_outmask)

                        dist = Main_Course_1_Reconstruction.eucl_dist_matrices(p, P_outmask)
                        if dist < dist_min:
                            dist_min = dist
                            p_s = P

                    # Copy paste the middle pixel value of P into the fingerprint image 
                    # at coordinates (x,y)
                    img.data[x][y] = p_s[1][1]

        # Return the restaured image
        return img

if __name__ == "__main__":

    from image import Image
    
    # Open the weak finger image
    img = Image("images/weak_finger.png")
    img.display()

    # Crop the image into several patches
    patches = img.crop_patches(1000)

    # Plot the patches for debug purpose
    # for patch in patches:
    #     plt.imshow(patch, cmap='gray', vmin=0, vmax=1)
    #     plt.show()

    # Create a mask with a small square 30x30 of True values (False otherwise)
    mask = np.full((img.n, img.m), False)
    mask[150 : 150 + 30, 100 : 100 + 30] = True

    # Apply the restauration algorithm onto the image
    img = Main_Course_1_Reconstruction.restauration(img, patches, mask)
    img.display()

    

    
