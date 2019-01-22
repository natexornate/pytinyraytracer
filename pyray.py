from PIL import Image
import math
import sys
import numpy as np
from numpy import linalg as LA
import time
from numba import jitclass, njit, deferred_type
import numba

lightSpec = [
    ('position', numba.float64[:]),
    ('intensity', numba.float64),
]
@jitclass(lightSpec)
class Light:
    def __init__(self, position, intensity):
        self.position = np.array(position)
        self.intensity = float(intensity)

matSpec = [
    ('color', numba.float64[:]),
    ('difuse_color', numba.float64[:]),
]

@jitclass(matSpec)
class Material:
    def __init__(self, color):
        self.color = np.array(color)
        self.difuse_color = self.color

#material_type = deferred_type()
#material_type.define(Material.class_type.instance_type)

sphereSpec = [
    ('center', numba.float64[:]),
    ('radius', numba.float64),
    ('radius2', numba.float64),
#    ('material', material_type),
]

@jitclass(sphereSpec)
class Sphere:
    def __init__(self, center, radius, material):
        self.center = np.array(center)
        self.radius = float(radius)
        self.radius2 = self.radius * self.radius
        self.material = material

    def ray_intersect(self, orig, dir):
        t0 = 0.0
        L = np.subtract(self.center, orig)
        #tca = np.sum(L*dir)
        tca_t = L*dir
        tca = tca_t[0] + tca_t[1] + tca_t[2]
        #d2 = np.sum(L*L) - (tca*tca)
        ll_t = L*L
        ll = ll_t[0] + ll_t[1] + ll_t[2]
        d2 = ll - (tca*tca)
        if d2 > self.radius2:
            return (False, t0)
        thc = math.sqrt(self.radius2 - d2)
        t0 = tca - thc
        t1 = tca + thc
        if t0 < 0:
            t0 = t1
        if t0 < 0:
            return (False, t0)

        return (True, t0)

@njit
def scene_intersect(orig, dir, spheres):
    spheres_dist = 1e308
    N = None
    material = None
    hit = None
    for s in spheres:
        (intersect, dist) = s.ray_intersect(orig, dir)
        if intersect:
            if dist < spheres_dist:
                spheres_dist = dist
                hit = orig + (dir * dist)
                N_vec = np.subtract(hit, s.center)
                N = N_vec/LA.norm(N_vec)
                material = s.material
    
    return (spheres_dist, material, N, hit)

@njit
def cast_ray(orig, dir, spheres, background, lights):
    (spheres_dist, material, N, point) = scene_intersect(orig, dir, spheres)
    if material is not None and spheres_dist < 1000:
        diffuse_light_intensity = 0.0
        for l in lights:
            light_dir = np.subtract(l.position, point)
            light_dir = light_dir/LA.norm(light_dir)
            diffuse_light_intensity += l.intensity * max(0.0, np.sum(light_dir*N))
        return material.difuse_color * diffuse_light_intensity
    return background.difuse_color

@njit
def getFB(width, height, spheres, background, lights):
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
            framebuffer.append(cast_ray(origin, dir, spheres, background, lights))
        if j % 32 == 0:
            print("Done row")

    return framebuffer

def render(sphere, background, lights):
    print("Hello, World!\n")
    width = 1024
    height = 768
    img = Image.new('RGB', (width, height))

    framebuffer = getFB(width, height, sphere, background, lights)

    data = img.load()

    for j in range(height):
        for i in range(width):
            pix = framebuffer[i+j*width]
            data[i,j] = (int(pix[0] * 255), int(pix[1] * 255), int(pix[2] * 255))

    img.save('out.png')

if __name__ == "__main__":
    ivory = Material([0.4, 0.4, 0.3])
    red_rubber = Material([0.3, 0.1, 0.1])
    background = Material([0.2, 0.7, 0.8])

    s = []
    s.append(Sphere([-3.0,  0.0,    -16.0], 2.0, ivory))
    s.append(Sphere([-1.0,  -1.5,   -12.0], 2.0, red_rubber))
    s.append(Sphere([1.5,   -0.5,   -18.0], 3.0, red_rubber))
    s.append(Sphere([7.0,   5.0,    -18.0], 4.0, ivory))

    l = []
    l.append(Light([-20.0,   20.0,    20.0], 1.5))

    start = time.time()
    render(s, background, l)
    stop = time.time()

    print('Elapsed time: {}'.format(stop - start))


