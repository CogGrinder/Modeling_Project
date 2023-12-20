## Pseudo code for convolution

2D convolution :
- matrix N x N with odd N
- need padding of image of size (N-1)/2
- original image : size n x m 

Algorithm : convolution by a matrix
Input     : matrix ``ker``, ``image_padded``

```python

result_image <- double list of size n x m
for i from 0 to n :
    for j from 0 to m :
        result <- 0
        for N_i from -(N-1)/2 to (N-1)/2 + 1:
            for N_j from -(N-1)/2 to (N-1)/2 + 1:
                result <- result + ker[N_i + (N-1)/2 ][N_j + (N-1)/2] * image_padded[i+N_i][j+N_j]
        result_image[i][j] <- result
return result_image
```