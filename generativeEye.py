# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:36:14 2018

@author: Reuben
"""

import noise
import numpy as np
import matplotlib.pyplot as plt
from math import atan2, cos, sin, pi, radians, floor
import random
from sklearn.preprocessing import normalize
from PIL import Image
from scipy.ndimage.filters import gaussian_filter

randStart = 10000*random.random()

def brownnoise(x, y, scl):
    scl = 1/scl
    x+=randStart
    y+=randStart
    n = noise.snoise2(x*scl, y*scl, octaves=8, lacunarity=2.5)
    n+=1
    n/=2
    return n
def dist(x0, y0, x1, y1):
    return abs( ((x1-x0)**2 + (y1-y0)**2)**.5)
def lerp(x0, x1, n):
    n = max(min(1, n), 0)
    return (1-n)*x0 + n*x1
#BELOW FUNCTION: Thanks to http://robotics.usc.edu/~ampereir/wordpress/?p=626
    #To save matplotlib to an image, usually just better to use PIL And NP
    
def SaveFigureAsImage(fileName,fig=None,**kwargs):
    ''' Save a Matplotlib figure as an image without borders or frames.
       Args:
            fileName (str): String that ends in .png etc.
 
            fig (Matplotlib figure instance): figure you want to save as the image
        Keyword Args:
            orig_size (tuple): width, height of the original image used to maintain 
            aspect ratio.
    '''
    fig_size = fig.get_size_inches()
    w,h = fig_size[0], fig_size[1]
    fig.patch.set_alpha(0)
    if 'orig_size' in kwargs: # Aspect ratio scaling if required
        w,h = kwargs['orig_size']
        w2,h2 = fig_size[0],fig_size[1]
        fig.set_size_inches([(w2/w)*w,(w2/w)*h])
        fig.set_dpi((w2/w)*fig.get_dpi())
    a=fig.gca()
    a.set_frame_on(False)
    a.set_xticks([]); a.set_yticks([])
    plt.axis('off')
    plt.xlim(0,h); plt.ylim(w,0)
    fig.savefig(fileName, transparent=True, bbox_inches='tight', \
                        pad_inches=0)    

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def mult(self, m):
        self.x *=m
        self.y *=m
    def heading(self):
        return atan2(self.y, self.x)
    def copy(self):
        return Vector(self.x, self.y)
    def fromAngle(a):
        return Vector(cos(a), sin(a))
    def lerp(v0, v1, n):
        n = max(min(1, n), 0)
        x = (1-n)*v0.x+n*v1.x
        y = (1-n)*v0.y+n*v1.y
        return Vector(x, y)
    
        
        
class Particle:
    
    def __init__(self, x, y):
        self.loc = Vector(x, y)
        self.angle = atan2(y, x)
        self.loc.y *= 1-np.random.normal(0, .02, 1)[0].item()
        self.loc.x*=1-np.random.normal(0, .02, 1)[0].item()
        
        theta = pi/2
        self.lowerbound = (self.angle-theta)
        self.upperbound = (self.angle+theta)
        self.destination = Vector.fromAngle(self.angle)
        self.destination.mult(.11)
        self.destination.x *=1- np.random.normal(0, .1, 1)[0].item()
        self.destination.y *=1- np.random.normal(0, .1, 1)[0].item()
    def update(self, bn, pic):
        radius = 400
        ploc = self.loc.copy()
        w = pic.shape[0]-1
        h = pic.shape[1]-1
        max0 = bn.shape[0]
        repetitions = int(w/10)
        for i in range(repetitions):
            ploc = Vector.lerp(self.loc, self.destination, i/repetitions)
            indx = max(min(max0-1, floor(ploc.x*(max0/2)+(max0/2))), 0)
            indy = max(min(max0-1, floor(ploc.y*(max0/2)+(max0/2))), 0)
            lerpval = bn[indx,indy].item()
            d = dist(ploc.x, ploc.y, 0, 0)
            result = Vector.fromAngle(lerp(self.lowerbound, self.upperbound, lerpval))
            result.mult(d)
            result.x = max(min(w, result.x * radius + w/2), 0)
            result.y = max(min(h, result.y * radius + h/2), 0)

            spike = abs(abs((d-.55))-.25)
            pic[round(result.x), round(result.y)] += (spike*.001+.001)**4


            #TO-DO: Do something with result, place it in an array, somehow save values

if __name__ == "__main__":
    pics = []
    result = np.zeros((1000, 1000, 3))
    for j in range(3):
        W = 1000
        H = 1000
        pic = np.zeros((W, H))
        bn = np.zeros((1000, 1000))
        it = np.nditer(bn, flags=['multi_index'], op_flags=['writeonly'])
        randStart = random.random() * 10000
        while not it.finished:
            it[0] = brownnoise(it.multi_index[1], it.multi_index[0], 10000)
            #TO DO : Experiment with changing this back to 1000, and why it becomes alll fuzzy
            it.iternext()
        rands = random.random()*100000
        
        particles = [Particle(cos(brownnoise(1000-i+rands, i+rands, 1500)*2*pi*20), sin(brownnoise(1000-i+rands, i+rands, 1500)*2*pi*20)) for i in range(100000)]
        for p in particles:
            p.update(bn, pic)
    #    pic = normalize(pic)
        #pic *= 255.0/pic.max()
        pic *= 1/pic.max()
        #pic = gaussian_filter(pic, sigma=3)
        pics.append(pic)
#        plt.figure(figsize=(10, 10), dpi=100)
#        plt.imshow(pic, cmap="gist_gray")
#        SaveFigureAsImage('out'+str(10+j)+'.png',plt.gcf(), orig_size=(W-1, H-1))
        print("Finished")
    result[..., 0] = pics[0]*255
    result[..., 1] = pics[1]*255
    result[..., 2] = (pics[2]/40)*255
    print("Done")

    im = Image.fromarray(np.uint8(result))
    im.save('eye.png')


            
            
            
            
        