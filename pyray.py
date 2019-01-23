#!/usr/bin/env python3
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
    ('refractive_index', numba.float64),
]

@jitclass(matSpec)
class Material:
    def __init__(self, refractive_index, albedo, color, spec):
        self.albedo = np.array(albedo)
        self.difuse_color = np.array(color)
        self.specular_exponent = float(spec)
        self.refractive_index = float(refractive_index)

if os.environ.get('NUMBA_DISABLE_JIT') == '1':
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
        tca = np.sum(L*dir)
        d2 = np.sum(L*L) - (tca*tca)
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
def normalize(vec):
    return vec/LA.norm(vec)

@njit
def refract(I, N, refractive_index):
    cosi = -max(-1., min(1., np.sum(I*N)))
    etai = 1
    etat = refractive_index
    if cosi < 0:
        cosi = -cosi
        n = -N
        etai, etat = etat, etai
    else:
        n = N
    
    eta = etai / etat
    k = 1 - eta*eta*(1 - cosi*cosi)

    if k < 0:
        return np.array([0.,0.,0.])
    else:
        return I*eta + n*(eta * cosi - math.sqrt(k))

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
                N = normalize(hit - s.center)
                material = s.material
    
    checkerboard_dist = 1e308
    if abs(dir[1]) > 1e-3:
        d = -(orig[1] + 4)/dir[1]
        pt = orig + dir*d
        if d > 0 and abs(pt[0]) < 10 and pt[2] < -10 and pt[2] > -30 and d < spheres_dist:
            checkerboard_dist = d
            hit = pt
            N = np.array([0., 1., 0.])
            square = (int(.5*hit[0]+1000) + int(.5*hit[2]))
            material = Material(1.0, [1., 0., 0., 0.], [.3,.2,.1], 0.)
            if square & 1:
                material.difuse_color = np.array([.3,.3,.3])

    return (min(spheres_dist, checkerboard_dist)<1000, material, N, hit)

@njit
def cast_ray(orig, dir, spheres, background, lights, depth):
    if depth > 4:
        return background.difuse_color
    (intersect, material, N, point) = scene_intersect(orig, dir, spheres)
    if intersect:
        reflect_dir = normalize(reflect(dir, N))
        if np.sum(reflect_dir*N) < 0:
            reflect_orig = point - N*1e-3
        else:
            reflect_orig = point + N*1e-3
        reflect_color = cast_ray(reflect_orig, reflect_dir, spheres, background, lights, depth+1)

        refract_dir = normalize(refract(dir, N, material.refractive_index))
        if np.sum(refract_dir * N) < 0:
            refract_orig = point - N*1e-3
        else:
            refract_orig = point + N*1e-3
        refract_color = cast_ray(refract_orig, refract_dir, spheres, background, lights, depth+1)

        diffuse_light_intensity = 0.0
        specular_light_intensity = 0.0
        for l in lights:
            light_dir = l.position - point
            light_distance = LA.norm(light_dir)
            light_dir = light_dir/light_distance

            if np.sum(light_dir*N) < 0:
                shadow_orig = point - N*1e-3
            else:
                shadow_orig = point + N*1e-3
            
            (shad_intersect, tmpmat, shadow_N, shadow_pt) = scene_intersect(shadow_orig, light_dir, spheres)
            if shad_intersect:
                shadow_dist = LA.norm(shadow_pt - shadow_orig)
                if shadow_dist < light_distance:
                    continue
            
            diffuse_light_intensity += l.intensity * max(0.0, np.sum(light_dir*N))
            base = max(0.0, np.sum(-reflect(-light_dir, N) * dir))
            l_spec = np.power(base, material.specular_exponent) * l.intensity
            specular_light_intensity += l_spec
        return material.difuse_color * diffuse_light_intensity * material.albedo[0] + np.array([1.,1.,1.])*specular_light_intensity*material.albedo[1] + reflect_color*material.albedo[2] + refract_color*material.albedo[3]
    return background.difuse_color

@njit
def getFB(width, height, spheres, background, lights):
    fov = math.pi / 3.
    origin = np.array([0.,0.,0.])
    framebuffer = []

    for j in range(height):
        for i in range(width):
            x =  (2*(i + 0.5)/float(width)  - 1)*math.tan(fov/2.)*width/float(height)
            y = -(2*(j + 0.5)/float(height) - 1)*math.tan(fov/2.)
            p = np.array([x,y,-1])
            dir = p/LA.norm(p)
            framebuffer.append(cast_ray(origin, dir, spheres, background, lights, 0))
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

    start = time.time()
    for j in range(height):
        for i in range(width):
            pix = framebuffer[i+j*width]
            maximum = max(pix)
            if maximum > 1:
                pix = pix*(1/maximum)
            data[i,j] = (int(pix[0] * 255), int(pix[1] * 255), int(pix[2] * 255))
    data_end = time.time()
    img.save('out.png')
    save_end = time.time()
    print("Data time: {} \tSaving time: {}".format(data_end - start, save_end - data_end))

if __name__ == "__main__":
    ivory =         Material(1.0, [0.6, 0.3, 0.1, 0.0],     [0.4, 0.4, 0.3], 50.)
    glass =         Material(1.5, [0.0, 0.5, 0.1, 0.8],     [0.6, 0.7, 0.8], 125.)
    red_rubber =    Material(1.0, [0.9, 0.1, 0.0, 0.0],     [0.3, 0.1, 0.1], 10.)
    mirror =        Material(1.0, [0.0, 10.0, 0.8, 0.0],    [1.0, 1.0, 1.0], 1425.)
    background =    Material(1.0, [1., 0., 0., 0.],         [0.2, 0.7, 0.8], 0.)

    s = []
    s.append(Sphere([-3.0,  0.0,    -16.0], 2.0, ivory))
    s.append(Sphere([-1.0,  -1.5,   -12.0], 2.0, glass))
    s.append(Sphere([1.5,   -0.5,   -18.0], 3.0, red_rubber))
    s.append(Sphere([7.0,   5.0,    -18.0], 4.0, mirror))

    l = []
    l.append(Light([-20.0,  20.0,    20.0], 1.5))
    l.append(Light([30.0,   50.0,    -25.0], 1.8))
    l.append(Light([30.0,   20.0,    30.0], 1.7))

    start = time.time()
    render(s, background, l)
    stop = time.time()

    print('Elapsed time: {}'.format(stop - start))


