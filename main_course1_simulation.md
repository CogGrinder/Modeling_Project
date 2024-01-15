1) functions which monotonically decrease as r → ∞, with c(0) = 1 and lim _r → ∞c(r) = 0 :
    - c(r) = 1 / (r+1)
    - c(r) = exp(-r)
    - c(r) = 1/2 (1 + erf((x - u)/(s*sqrt(2)))) , with u = 5, s² = 0.2 (here we have lim r-->0 c(r) = 1 and not c(0) = 1)
    I like this formulation of the function c, because u corresponds to the size in pixels of the radius of data almost inchanged, and s² corresponds to the width of the blured border, around the circle (link with density and cdf function of N(u,s²) quite logic in my sense) 

isotropy : invariance of the physical properties in a specific field as a function of direction
anisotropy : variance of the physical properties as a function of the direction in a given field

We can make this observation of anisotropy in the case of the simulation of low pressure on a fingerprint since it's clearly visible, that the pressure is not lowering with the same intensity when you move away from the center of pressure.
Indeed, the pressure drops fastly on the horizontal axis (of a fingerprint, left to right axis of a finger print) than on teh vertical axis of a fingerprint.
The result of a low pressure looks more like an ellipse oriented towards the vertical axis of the fingerprint than a circle.
The big challenge here, beside creating a function c(x, y) that depends on the direction in additino to the distance to the center of low pressure (I have some ideas to deal with it), will be to create a function that will spot the orientation of the vertical axis of the fingerprint on a given image
