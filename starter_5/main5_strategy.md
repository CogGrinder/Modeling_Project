## Ideas for main 5

### New essential features
- [x] at the end of optimization, plot the best warped image next to the fixed image
    - side by side for testing
    - show on top with red and blue (with truncated sum of images)
- [ ] support new loss function
    - [ ] add warp derivative - trivial 1 in the case of translate

### From subject
#### to answer : how could you use this algo for a broader warp like txy_finger.png ?
- [ ] add support for bigger images and bigger warps
    - [ ] put the image with most loss as fixed
    - [ ] blur approach
-  add more detail? definitions?



### Better code structure
- [x] add doc for get_pix_at
- [ ] add doc for loss_function_1 (warn assuming warp supports np.ndarray)
- [ ] add doc for make_save_name

- [ ] use common functions/skeleton for different optimisation

- [ ] clarify i,j x,y (y in shown data is actually "x" in image ? look at meshgrid)
- [ ] add legend to 3d plot 
- [ ] add interpolation function and update rotate translate function with Sabin

### Better quality algorithm
- apply pixelize blur on both images and optimize, then use result as p0
- choose the next step with second derivative and parabolic (maybe cubic) regression/interpolation and take theoretical minimum

- choose the best quality image as moving so it is adjusted to the fixed image

- apply successive warps
    - support new warp:
        first rotation with Sabin



### LaTeX corrections
- [ ] remove the ``l_list`` in the returns
- [ ] add interpretation for ridges and bumps in loss_function