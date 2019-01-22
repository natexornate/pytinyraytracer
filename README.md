# pytinyraytracer
This is my fun implementation of the cool TinyRayTracer course here: https://github.com/ssloy/tinyraytracer

I tried implementing the RayTracer in Python for fun. I'm not a Python expert, so I probably did it wrong. It's slow, but it mostly works!

For math/vectors I used Numpy, for image saving I used Pillow to save directly to PNG. I still think this is in the spirit of the original course in regards to third party libraries.

I'm currently working on step 6: Shadows. I'm running into issues with the graphics looking bad and not matching the original C++ implementation. I'll work through that and publish more when I figure it out.

In an attempt to speed things up, I used the Numba library which JITs the Python code using LLVM. See here for more details: https://numba.pydata.org/

Requirements:
* Python (I used 3.7.2)
* Numpy
* Numba
* Pillow (replacement for PIL)

```
$ pip install numba Pillow numpy
$ python pyray.py
```
