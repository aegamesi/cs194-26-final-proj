import numpy as np
import skimage.io as skio
import skimage as sk
import IPython
import PIL.Image
import IPython.display
import scipy.ndimage
import scipy
import skimage.transform
import matplotlib.pyplot as plt
import cv2

import tempfile
import io
import os
from multiprocessing import Pool
import random

def load_images_from_directory(directory, scale=1.0):
    filenames = sorted(os.listdir(directory))
    filenames = [x for x in filenames if any(x.endswith(y) for y in ('jpg', 'jpeg', 'png'))]
    filenames = ["{}/{}".format(directory, x) for x in filenames]
    print("{} images".format(len(filenames)))

    imgs = []
    for i, filename in enumerate(filenames):
        print("Loading {} / {}...".format(i + 1, len(filenames)))
        imgs.append(imread(filename, scale))
    return imgs

def imread(filename, scale=1.0, as_float=True):
    im = skio.imread(filename)
    if as_float:
        im = sk.img_as_float(im)
    if scale != 1.0:
        im = sk.transform.rescale(im, scale)
    return im

def imsave(filename, im):
    if np.max(im) > 50:
        im = im / 255.0
    im = np.clip(im, -1, 1)
    skio.imsave(filename, im)

def showimage(im, clip=True, quality=80):
    if np.max(im) > 50:
        im = im / 255.0
    if clip:
        im = np.clip(im, 0, 1)
    im = sk.img_as_ubyte(im)
    f = io.BytesIO()
    PIL.Image.fromarray(im).save(f, 'jpeg' if quality < 100 else 'png', quality=quality)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))

def videosave(filename, ims, fps=60):
    with tempfile.TemporaryDirectory() as d:
        with Pool() as p:
            args = [(os.path.join(d, "%010d.jpg" % i), im) for i, im in enumerate(ims)]
            p.starmap(imsave, args)

        cmd = "ffmpeg -framerate {} -i '{}/%10d.jpg' -pix_fmt yuv420p -r {} -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' {}"
        cmd = cmd.format(fps, d, fps, filename)
        print(cmd)
        if os.path.exists(filename):
            os.remove(filename)
        os.system(cmd)

def vecdist(a, b):
    return np.sum((a - b) ** 2.0) ** 0.5

# ffmpeg -framerate 60 -i '%2d.png' -pix_fmt yuv420p -r 60 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" out.mp4

#######################################################################

# RANSAC

def warp_point(H, pt):
    x, y, p = H @ (pt[0], pt[1], 1.0)
    return (x / p, y / p)

def get_homography(pts_0, pts_1, max_iterations=1000, dist_threshold=5):
    assert len(pts_0) == len(pts_1)
    m = len(pts_0)
    best_inliers = None
    for i in range(max_iterations):
        rand_inds = random.sample(range(m), 4)
        pts0 = [pts_0[x] for x in rand_inds]
        pts1 = [pts_1[x] for x in rand_inds]
        H = compute_hom_rigid(pts0, pts1, inv=True)
        inliers = []
        for j in range(m):
            pt0_warp = warp_point(H, pts_0[j])
            pt1 = pts_1[j]
            d = vecdist(pt0_warp, pt1)
            if d < dist_threshold:
                inliers.append(j)

        if best_inliers is None or len(inliers) > len(best_inliers):
            best_inliers = inliers

    pts0 = [pts_0[x] for x in best_inliers]
    pts1 = [pts_1[x] for x in best_inliers]
    H = compute_hom_rigid(pts0, pts1, inv=False)
    print("RANSAC returned", len(best_inliers), "/", m)
    return H

def compute_hom_rigid(im1_pts, im2_pts, m=1.0, inv=True):
    # http://nghiaho.com/?p=2208
    # ax - by + 1c + 0d= x'
    # ay + by + 0c + 1d = y'
    assert len(im1_pts) == len(im2_pts)
    
    A = []
    b = []
    for i in range(len(im1_pts)):
        if inv:
            x, y = im1_pts[i]
            X, Y = im2_pts[i]
        else:
            y, x = im1_pts[i]
            Y, X = im2_pts[i]
        x, y, X, Y = x * m, y * m, X * m, Y * m
        A.append([x, -y, 1.0, 0.0])
        A.append([y,  x, 0.0, 1.0])
        b.append(X)
        b.append(Y)
        
    x, _, _, _  = np.linalg.lstsq(np.float32(A), np.float32(b), rcond=None)
    H = np.float32([[x[0], -x[1], x[2]], [x[1], x[0], x[3]], [0.0, 0.0, 1.0]])
    return H
