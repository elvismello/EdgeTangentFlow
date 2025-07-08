#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Original source https://github.com/naotokimura/EdgeTangentFlow

import os
import numpy as np
import multiprocessing as mp
from PIL import Image, ImageOps
from scipy import ndimage


KERNEL = 5
COLOUR_OR_GRAY = 0
input_img = "test_images/Lenna.png"
SIZE = (512, 512, 3)
CPU_COUNT = mp.cpu_count()

flowField = np.zeros(SIZE, dtype = np.float32)
gradientMag = np.zeros(SIZE, dtype = np.float32)
phiPreCalc = np.zeros(SIZE, dtype = np.float32)

####################
# Generate ETF 
####################
#memo
 #cv2.normalize(src[, dst[, alpha[, beta[, norm_type[, dtype[, mask]]]]]])：正規化
 #cv2.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]])：微分
 #cv2.magnitude(x, y[, magnitude])：2次元ベクトルの大きさ



def initial_ETF(input_img, size):
    global flowField
    global gradientMag

    src = np.array(ImageOps.exif_transpose(Image.open(input_img)).convert("L"))
    src_n = np.array(src, dtype=np.float32) / 255
    print(src.shape)

    #Generate grad_x and grad_y
    grad_x = np.array(ndimage.sobel(src_n, axis=1))
    grad_y = np.array(ndimage.sobel(src_n, axis=0))

    #Compute gradient
    gradientMag = np.sqrt(grad_x**2.0 + grad_y**2.0)
    gradientMag = gradientMag / np.max(gradientMag)

    flowField = np.stack((grad_x, grad_y), axis=-1)
    norm = np.linalg.norm(flowField, axis=2, keepdims=True)
    norm[norm == 0] = 1
    flowField = flowField / norm

    rotateFlow(flowField, 90.0)



def rotateFlow(src, theta):
    global flowField

    theta = theta / 180.0 * np.pi

    rx = src[:, :, 0] * np.cos(theta) - src[:, :, 1] * np.sin(theta)
    ry = src[:, :, 1] * np.cos(theta) + src[:, :, 0] * np.sin(theta)

    flowField = np.stack((rx, ry, np.zeros_like(rx)), axis=-1)



def refine_ETF(kernel):
    global flowField

    h_f, w_f = flowField.shape[:2]
    args = []
    for r in range(h_f):
        for c in range(w_f):
            args.append([c, r, kernel])
            #computeNewVector(c, r, kernel)

    with mp.Pool() as pool:
        results = pool.starmap(computeNewVector, args)

    #print(results)
    for y, x, vec in results:
        #print(y, x, vec)
        flowField[int(y), int(x), :] = vec



def process_pixel(x, y, kernel):
    result_vec = computeNewVector(x, y, kernel)
    return (y, x, result_vec)



#Paper's Eq(1)
def computeNewVector(x, y, kernel):
    global flowField, gradientMag

    h, w = flowField.shape[:2]

    y_min = max(0, y - kernel)
    y_max = min(h, y + kernel + 1)
    x_min = max(0, x - kernel)
    x_max = min(w, x + kernel + 1)

    # Central vector
    t_cur_x = flowField[y, x]
    gradmag_x = gradientMag[y, x]
    pos_xy = np.array([x, y])

    # Getting neighborhoods
    patch_flow = flowField[y_min:y_max, x_min:x_max]  # (Hk, Wk, 3)
    patch_mag = gradientMag[y_min:y_max, x_min:x_max] # (Hk, Wk)

    # Neighborhood coords
    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
    patch_pos = np.stack((xx, yy), axis=-1)  # (Hk, Wk, 2)

    phi = np.sign(np.sum(patch_flow * t_cur_x, axis=-1))  # (Hk, Wk) # Eq(5)
    dist = np.linalg.norm(patch_pos - pos_xy, axis=-1)
    ws = (dist < kernel).astype(np.float32) # Eq(2)
    wm = (1 + np.tanh(patch_mag - gradmag_x)) / 2 # Eq(3)
    wd = np.abs(np.sum(patch_flow * t_cur_x, axis=-1)) # Eq(4)

    weight = phi * ws * wm * wd  # (Hk, Wk)

    # Applying to vectors
    weighted = patch_flow * weight[..., np.newaxis]  # (Hk, Wk, 3)

    # Total sum
    t_new = np.sum(weighted, axis=(0, 1))

    norm = np.linalg.norm(t_new)
    if norm > 0:
        t_new /= norm
    else:
        t_new = np.zeros_like(t_new)

    #refinedETF[y, x] = t_new
    return y, x, t_new



#Paper's Eq(5)
def computePhi(x, y):
    if np.dot(x, y) > 0:
        return 1
    else:
        return -1
    


#Paper's Eq(2)
def computeWs(x, y, r):
    if np.linalg.norm(x-y) < r:
        return 1
    else:
        return 0



#Paper's Eq(3)
# TODO implement eta
# wm = (1 + np.tanh(eta(gradmag_y - gradmag_x))) / 2
def computeWm(gradmag_x, gradmag_y):
    wm = (1 + np.tanh(gradmag_y - gradmag_x)) / 2
    return wm



#Paper's Eq(4)
def computeWd(x, y):
    return abs(x.dot(y))



# save image
def save_ETF(count, kernel):
    global flowField
    global img_name

    img_name = os.path.splitext(os.path.basename(input_img))[0]
    directory = os.path.join("frames", img_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    np.save(directory + f"/etf_kernel_iter_{kernel}_{count}.npy", flowField)



if __name__ == '__main__':
    print("Making initial ETF...", flush=True)
    initial_ETF(input_img, SIZE)
    print("Starting refinement...", flush=True)
    for i in range(10):
        print(f"Iteration {i:2d}")
        refine_ETF_mp(KERNEL)
        save_ETF(i, KERNEL)
