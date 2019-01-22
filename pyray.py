from PIL import Image
import math
import os
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
    ('albedo', numba.float64[:]),
    ('difuse_color', numba.float64[:]),
    ('specular_exponent', numba.float64),
]

@jitclass(matSpec)
class Material:
    def __init__(self, albedo, color, spec):
        self.albedo = np.array(albedo)
        self.difuse_color = np.array(color)
        self.specular_exponent = float(spec)

if os.environ['NUMBA_DISABLE_JIT'] == '1':
    material_type = None
else:
    material_type = deferred_type()
    material_type.define(Material.class_type.instance_type)

sphereSpec = [
    ('center', numba.float64[:]),
    ('radius', numba.float64),
    ('radius2', numba.float64),
    ('material', material_type),
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
def reflect(I, N):
    return I - N * 2.0 * np.sum(I * N)

@njit
def scene_intersect(orig, dir, spheres):
    spheres_dist = 1e308
    N = np.array([0.0,0.0,0.0])
    material = None
    hit = np.array([0.0,0.0,0.0])
    for s in spheres:
        (intersect, dist) = s.ray_intersect(orig, dir)
        if intersect:
            if dist < spheres_dist:
                spheres_dist = dist
                hit = orig + (dir * dist)
                N_vec = hit - s.center
                N = N_vec/LA.norm(N_vec)
                material = s.material
    
    return (spheres_dist, material, N, hit)

@njit
def cast_ray(orig, dir, spheres, background, lights):
    (spheres_dist, material, N, point) = scene_intersect(orig, dir, spheres)
    if material is not None and spheres_dist < 1000:
        diffuse_light_intensity = 0.0
        specular_light_intensity = 0.0
        for l in lights:
            light_dir = l.position - point
            light_dir = light_dir/LA.norm(light_dir)
            diffuse_light_intensity += l.intensity * max(0.0, np.sum(light_dir*N))
            base = max(0.0, np.sum(-reflect(-light_dir, N) * dir))
            l_spec = np.power(base, material.specular_exponent) * l.intensity
            specular_light_intensity += l_spec
        return material.difuse_color * diffuse_light_intensity * material.albedo[0] + np.array([1.,1.,1.])*specular_light_intensity*material.albedo[1]
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
            maximum = max(pix)
            if maximum > 1:
                pix = pix*(1/maximum)
            data[i,j] = (int(pix[0] * 255), int(pix[1] * 255), int(pix[2] * 255))

    img.save('out.png')

if __name__ == "__main__":
    ivory = Material([0.6, 0.3], [0.4, 0.4, 0.3], 50.)
    red_rubber = Material([0.9, 0.1], [0.3, 0.1, 0.1], 10.)
    background = Material([1., 0.], [0.2, 0.7, 0.8], 0.)

    s = []
    s.append(Sphere([-3.0,  0.0,    -16.0], 2.0, ivory))
    s.append(Sphere([-1.0,  -1.5,   -12.0], 2.0, red_rubber))
    s.append(Sphere([1.5,   -0.5,   -18.0], 3.0, red_rubber))
    s.append(Sphere([7.0,   5.0,    -18.0], 4.0, ivory))

    l = []
    l.append(Light([-20.0,  20.0,    20.0], 1.5))
    l.append(Light([30.0,   50.0,    -25.0], 1.8))
    l.append(Light([30.0,   20.0,    30.0], 1.7))

    start = time.time()
    render(s, background, l)
    stop = time.time()

    print('Elapsed time: {}'.format(stop - start))


