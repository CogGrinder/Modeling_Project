- Rotation of 45° to the left + a translation
- w could be the function of rotation-translation, of angle p, center c=(cx, cy), translation (offset_x, offset_y)
- Many methods exist, we are using a bilinear interpolation, but we could have implemented a cubic interpolation (possible way to improve) 
- the parameter of this motion model is the angle p, angle of rotation, the center of rotation and the translation parameters (on x and on y)
- the rotation function is of complexity O(n²)
- to double check our algorithm, we could apply the exact inverse rotation-translation to the image obtained by a given one. 
We observe that we are loosing some information, firstly due to the fact that the rotation make some information "going out" of the frame of the image (lost of information between the transformed image and the inverse of the transformed image, which should be the original one). When we add a translation to the rotation, we are loosing some information in a transformation itself, since we are doing firstly a rotation, and then a translation. 
We could have the objective to improve all that by having a bigger frame to work with during the transformations.