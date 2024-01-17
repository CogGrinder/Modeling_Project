## Ideas for main 5

### New essential features
- [ ] at the end of optimization, plot the best warped image next to the fixed image
    - show on top with red and blue (with truncated sum of images)
    - side by side otherwise
- [ ] support new loss function
-  add more detail? definitions? 


### Better code structure
- [x] add doc for get_pix_at
- [ ] add doc for loss_function_1 (warn assuming warp supports np.ndarray)
- [ ] add doc for make_save_name
- [ ] clarify i,j x,y (y in shown data is actually "x" in image ? look at meshgrid)
- [ ] add legend to 3d plot 
- [ ] add interpolation function and update rotate translate function with Sabin

### Better quality algorithm
- apply blur and optimize, then use result as p0
- choose the next step with second derivative and parabolic (maybe cubic) interpolation and take theoretical minimum

- choose the best quality image as moving so it is adjusted to the fixed image

- apply successive warps (if multp) and 



### LaTeX corrections
- [ ] remove the ``l_list`` in the returns
- [ ] add interpretation for ridges and bumps in loss_function