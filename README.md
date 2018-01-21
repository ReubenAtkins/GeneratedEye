# GeneratedEye
Uses: Numpy, Caseman's Noise (https://github.com/caseman/noise), math, random, and PIL 
(with a few other random libraries thrown in for experimentation). 

Program to automatically generate a random eye using fractal simplex noise.

There are some color options that generate multiple brownian noise flowfields in order to color the eye. 

Some black and white images are added as well, as I wasn't very succcessful in finding a good way to color the generated eye, but
I have been working on some other flowfields to devise a way to color.

Similarly, if interested, Inconvergent has a very interesting coloring method, but seeing as this program is already a bit bloated
(takes a while to run and generate the multiple fields even with multiprocessing), I'll leave this as something to think about for anyone
who might seek to pick this up and build on it. 
https://github.com/inconvergent/sand-spline

Out10.png is also somewhat edited in photoshop to increase the colors and change them around a little bit to see what could be with 
some improvements to coloring and flowfields. 

