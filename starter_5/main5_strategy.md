## Ideas for main 5

### New essential features
- [x] at the end of optimization, plot the best warped image next to the fixed image
    - side by side for testing
    - show on top with red and blue (with truncated sum of images)
- [x] test all functions on (5,5) - tested most functions
    - [x] on report: say that for simplicity (time constraint) and optimization, the translate function removes information from first column and first row
        note: the proper way is by adding a padding
    - [x] for test add constructor from numpy array in image.py
- [ ] support new loss function
    - [x] normalized arbitrarily
    - [ ] normalize better
- [ ] support txy_finger.png


- [ ] bug hunting
    - [ ] make use of testing file more clear
    - [ ] fix translate function by adding first line and firt row manually at the end
    - [ ] fix gradient by making a more accurate "signature"

### From subject
#### to answer : how could you use this algo for a broader warp like txy_finger.png ?
- [ ] add support for bigger images and bigger warps
    - [ ] put the image with most loss as fixed
    - [x] blur approach
-  add more detail? definitions?



### Better code structure

- [ ] add legend to 3d plot
- [ ] add interpolation function and update rotate translate function with Sabin

better
- [ ] clarify i,j x,y (y in shown data is actually "x" in image ? look at meshgrid)
- [ ] use common functions/skeleton for different optimisation

doc
- [x] add doc for get_pix_at
- [ ] add doc for loss_function_1 (warn assuming warp supports np.ndarray)
- [ ] add doc for make_save_name


### Better quality algorithm
- [ ] add interpolation function and update rotate translate function with Sabin

- [ ] support for rotation
    - [ ] add warp derivative - trivial 1 in the case of translate but requires restructuring code


- [ ] apply pixelize blur on both images and optimize, then use result as p0
    - [x] first apply blur to image
    - [x] test algorithm and display
    - [x] now save p0 and reinitialize image, redo the same algorithm
    - [ ] figure out the parameters

- [ ] choose the next step with second derivative and parabolic (maybe cubic) regression/interpolation and take theoretical minimum

- choose the best quality image as moving so it is adjusted to the fixed image
    - error: actually take the lower quality image

- apply successive warps
    - support new warp:
        first rotation with Sabin



### LaTeX part
- [ ] need better labels in 3d plots

- [ ] use x translate graph


#### content
- explain that we use fixed average to avoid getting a null average when translate is too big in loss_function_2, and that it does not support non isometric warps like dilation and shrinking
- [ ] add interpretation for ridges and bumps in loss_function
#### cosmetic details
- [ ] remove the ``l_list`` in the returns