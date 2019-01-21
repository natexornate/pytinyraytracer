from PIL import Image
import math
import numpy as np
from numpy import linalg as LA
import time
from numba import jitclass
from numba import njit
import numba

spec = [
    ('center', numba.float64[:]),               # a simple scalar field
    ('radius', numba.float64),          # an array field
]

@jitclass(spec)
class Sphere:
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = float(radius)

    def ray_intersect(self, orig, dir):
        L = np.subtract(self.center, orig)
        tca = np.sum(L*dir)
        d2 = np.sum(L*L) - (tca*tca)
        if d2 > (self.radius * self.radius):
            return False
        thc = math.sqrt((self.radius * self.radius) - d2)
        t0 = tca - thc
        t1 = tca + thc
        if t0 < 0:
            t0 = t1
        if t0 < 0:
            return False

        return True

@njit
def cast_ray(orig, dir, sphere):
    if sphere.ray_intersect(orig, dir):
        return (0.4, 0.4, 0.3)
    return (0.2, 0.7, 0.8)

@njit
def getFB(width, height, sphere):
    fov = int(math.pi / 2)
    tanfovo2 = math.tan(fov/2)
    origin = np.array([0.,0.,0.])
    framebuffer = []

    for i in range(height):
        for j in range(width):
            x =  (2*(i + 0.5)/float(width)  - 1)*tanfovo2*width/float(height)
            y = -(2*(j + 0.5)/float(height) - 1)*tanfovo2
            p = np.array([x,y,-1])
            dir = p/LA.norm(p)
            framebuffer.append(cast_ray(origin, dir, sphere))
        if i % 32 == 0:
            print("Done row")

    return framebuffer

def render(sphere):
    print("Hello, World!\n")
    width = 1024
    height = 768
    img = Image.new('RGB', (width, height))

    framebuffer = getFB(width, height, sphere)

    data = img.load()

    for i in range(width):
        for j in range(height):
            pix = framebuffer[i+j*width]
            data[i,j] = (int(pix[0] * 255), int(pix[1] * 255), int(pix[2] * 255))

    img.save('out.png')





if __name__ == "__main__":
    s = Sphere([-3.0,0.0,-16.0], 2.0)

    start = time.time()
    render(s)
    stop = time.time()

    print('Elapsed time: {}'.format(stop - start))


