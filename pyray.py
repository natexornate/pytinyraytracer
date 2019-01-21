from PIL import Image
import math
import sys
import numpy as np
from numpy import linalg as LA
import time
from numba import jitclass
from numba import njit
import numba

class Material:
    def __init__(self, color):
        self.color = color
        self.difuse_color = color


sphereSpec = [
    ('center', numba.float64[:]),               # a simple scalar field
    ('radius', numba.float64),          # an array field
]

#@jitclass(sphereSpec)
class Sphere:
    def __init__(self, center, radius, material):
        self.center = np.array(center)
        self.radius = float(radius)
        self.material = material

    def ray_intersect(self, orig, dir):
        t0 = 0.0
        L = np.subtract(self.center, orig)
        tca = np.sum(L*dir)
        d2 = np.sum(L*L) - (tca*tca)
        if d2 > (self.radius * self.radius):
            return (False, t0)
        thc = math.sqrt((self.radius * self.radius) - d2)
        t0 = tca - thc
        t1 = tca + thc
        if t0 < 0:
            t0 = t1
        if t0 < 0:
            return (False, t0)

        return (True, t0)

def scene_intersect(orig, dir, spheres):
    spheres_dist = sys.float_info.max
    N = None
    material = None
    for s in spheres:
        (intersect, dist) = s.ray_intersect(orig, dir)
        if intersect:
            if dist < spheres_dist:
                spheres_dist = dist
                hit = orig + (dir * dist)
                N_vec = (hit - s.center)
                N = N_vec/LA.norm(N_vec)
                material = s.material
    
    return (spheres_dist, material, N)

#@njit
def cast_ray(orig, dir, spheres):
    (spheres_dist, material, N) = scene_intersect(orig, dir, spheres)
    if material is not None and spheres_dist < 1000:
        return material.difuse_color
    return (0.2, 0.7, 0.8)

#@njit
def getFB(width, height, spheres):
    fov = int(math.pi / 2)
    tanfovo2 = math.tan(fov/2)
    tanfovo2timeswidthdivheight = tanfovo2*width/float(height)
    fwidth = float(width)
    fheight = float(height)
    origin = np.array([0.,0.,0.])
    framebuffer = []

    for j in range(height):
        for i in range(width):
            x =  (2*(i + 0.5)/fwidth  - 1)*tanfovo2timeswidthdivheight
            y = -(2*(j + 0.5)/fheight - 1)*tanfovo2
            p = np.array([x,y,-1])
            dir = p/LA.norm(p)
            framebuffer.append(cast_ray(origin, dir, spheres))
        if j % 32 == 0:
            print("Done row")

    return framebuffer

def render(sphere):
    print("Hello, World!\n")
    width = 1024
    height = 768
    img = Image.new('RGB', (width, height))

    framebuffer = getFB(width, height, sphere)

    data = img.load()

    for j in range(height):
        for i in range(width):
            pix = framebuffer[i+j*width]
            data[i,j] = (int(pix[0] * 255), int(pix[1] * 255), int(pix[2] * 255))

    img.save('out.png')

if __name__ == "__main__":
    ivory = Material([0.4, 0.4, 0.3])
    red_rubber = Material([0.3, 0.1, 0.1])

    s = []
    s.append(Sphere([-3.0,  0.0,    -16.0], 2.0, ivory))
    s.append(Sphere([-1.0,  -1.5,   -12.0], 2.0, red_rubber))
    s.append(Sphere([1.5,   -0.5,   -18.0], 3.0, red_rubber))
    s.append(Sphere([7.0,   5.0,    -18.0], 4.0, ivory))

    start = time.time()
    render(s)
    stop = time.time()

    print('Elapsed time: {}'.format(stop - start))


