# stdlib
import io
import importlib
import functools
import math
import os
from multiprocessing import Pool
import json
import pickle
import argparse
import sys

# first party
import common
importlib.reload(common)

# third party
from contexttimer import Timer
from tqdm import tqdm

# SciPy / OpenCV
import numpy as np
import cv2
from matplotlib import pyplot as plt
import skimage.io as skio
import skimage as sk
import IPython
import PIL.Image
import IPython.display
import scipy.ndimage
import scipy
import skimage.transform
import matplotlib.pyplot as plt


## argparse
parser = argparse.ArgumentParser()
parser.add_argument('images', type=str, help='directory containing images to load')
parser.add_argument('output', type=str, help='save the output video to here')

parser.add_argument('--crop', type=str, default="0,0,0,0", help='"left,right,top,bottom" pixels to crop the images by')
parser.add_argument('--offset', type=int, default=0, help='start with the N-th image in the directory')
parser.add_argument('--count', type=int, default=0, help='only use N images')

parser.add_argument('--stabilization', type=float, default=20.0, help='sigma to use with vanish point stabilization')
parser.add_argument('--speed', type=float, default=1.0, help='average this many frames per input image')
parser.add_argument('--draw-distance', type=int, default=100, help='how many images to draw per frame')
parser.add_argument('--mask-radius', type=int, default=32, help='how blurred the mask should be')

parser.add_argument('--color', action='store_true', help='use color images')
parser.add_argument('--parallel', action='store_true', help='try to speed it up with parallelism')
parser.add_argument('--homography-cache', type=str, default=None, help='where to cache the homographies')
args = parser.parse_args()


## utilities
def lerp(a, b, t):
    return (b - a) * t + a

def combine(a, b):
    return np.multiply(a, (1 - np.ma.masked_where(b != 0, b).mask)) + b

def combine_mask(a, b, mask):
    return np.multiply(a, 1 - mask) + np.multiply(b, mask)

def combine_many(ims):
    return functools.reduce(lambda a, b: combine(b, a), ims[::-1])

def transform_pts(pts, M):
    """
    pts is like np.float32([[x, y], [x1, y1], [x2, y2]]...)
    
    """
    trans = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), M)
    return trans.reshape(pts.shape)

def get_vanishing_pt(H, h, w):
    corners = np.float32([[w * 0.5, h * 0.5]])
    vanishing, = transform_pts(corners, H)
    return vanishing

def find_homography(inputs):
    im1, im2 = inputs
    sift = cv2.xfeatures2d.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    good = []
    for m,n in matches:
        if m.distance < 0.9*n.distance:
            good.append(m)
            
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,2)
    
    
    #M = common.get_homography(src_pts, dst_pts)
    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 20.0)
    M = cv2.estimateRigidTransform(src_pts, dst_pts, False)
    if M is None:
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 20.0)
        print("***")
    else:
        M = np.vstack([M, np.float32([[0, 0, 1]])])
    
    return M

def lerpiform(w, h, M, t):
    """
    Computes A_(b -> a, t) from T_(b -> a)
    
    Computes the matrix needed to transform an image warped from the viewport (0 -> w, 0 -> h)
    and then lerp it back to the viewport with parameter t.
    """
    
    # transform corners to the inner image's corners
    corners = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0] ])
    inner_corners = transform_pts(corners, M)

    # lerp inner corners towards the outer corners
    lerped = lerp(inner_corners, corners, t)

    # find a new matrix to go from outer corners to lerped inner corners
    M2 = cv2.getPerspectiveTransform(inner_corners, lerped)

    return M2

def lerpywarp(im_inner, im_outer, M, t):
    h, w = im_inner.shape
    M2 = lerpiform(w, h, M, t)

    im_inner_2 = cv2.warpPerspective(im_inner, M2 @ M, (w, h))
    im_outer_2 = cv2.warpPerspective(im_outer, M2, (w, h))

    return im_inner_2 #(im_outer_2 + im_inner_2) * 0.5#combine(im_outer_2, im_inner_2)

def pairwise_homographies(imgs):
    """
    Takes in a list of images, computes homographies between each image
    and its successive image, the first one is always the identity matrix.
    """
    inputs = [(imgs[i + 1], imgs[i]) for i in range(len(imgs) - 1)]

    if args.parallel:
        with Pool() as p:
            homs = list(tqdm(p.imap(find_homography, inputs), total=len(inputs)))
    else:
        homs = list(tqdm((find_homography(x) for x in inputs), total=len(inputs)))

    return [np.identity(3)] + homs

def _compute_frame(args):
    t, context = args
    imgs, Ts, target_vanishing_pts, mask, w, h, draw_distance = context
    lower = int(math.floor(t))
    A = lerpiform(w, h, Ts[lower + 1], t - lower)
    
    Mvanish = functools.reduce(np.dot, [A] + Ts[lower + 1:])
    vanishing = get_vanishing_pt(Mvanish, h, w)
    target = target_vanishing_pts[lower]
    X = np.eye(3)
    X[0][2] = target[0] - vanishing[0]
    X[1][2] = target[1] - vanishing[1]
    #A = A @ X

    transformed_images = []
    im = None
    for j in range(lower, min(len(imgs), lower + draw_distance)):
        M = functools.reduce(np.dot, [A] + Ts[lower + 1:j + 1])

        im_trans = cv2.warpPerspective(imgs[j], M, (w, h))
        mask_trans = cv2.warpPerspective(mask, M, (w, h))
        
        if im is None:
            im = im_trans
        else:
            im = combine_mask(im, im_trans, mask_trans)
            
    return im
    
def final_video(imgs, Ts, target_vanishing_pts, speed=1.0, draw_distance=30, mask_r=32):
    h, w = imgs[0].shape[:2]
    
    # compute mask
    mask = np.zeros((h, w))
    mask[mask_r:-mask_r,mask_r:-mask_r] = 1.0
    mask = scipy.ndimage.gaussian_filter(mask, sigma=(mask_r * 0.5))
    if len(imgs[0].shape) == 3:
        mask = np.dstack([mask] * imgs[0].shape[-1])
    
    # compute timestamps to make frames at
    dets = [min(1.0, np.linalg.det(x)) for x in Ts]
    logdets = [-math.log(x) for x in dets]
    cdf = [sum(logdets[:i+1]) for i in range(len(logdets))]
    numframes = speed * len(imgs)
    
    timestamps = []
    for y in np.linspace(cdf[0], cdf[-1], int(numframes)):
        ind = max(0, np.searchsorted(cdf, y, side='right') - 1)
        if ind + 1 >= len(cdf):
            break
        lval, rval = cdf[ind], cdf[ind + 1]
        percent = (y - lval) / (rval - lval)
        t = ind + percent
        timestamps.append(t)

    context = (imgs, Ts, target_vanishing_pts, mask, w, h, draw_distance)
    inputs = [(t, context) for t in timestamps]
    if args.parallel:
        with Pool() as p:
            frames = p.imap(_compute_frame, inputs, chunksize=20)
            frames = list(tqdm(frames, total=len(inputs)))
    else:
        frames = (_compute_frame(x) for x in inputs)
        frames = list(tqdm(frames, total=len(inputs)))
        
    return frames

######

print("Loading images...")
filenames = sorted([x for x in os.listdir(args.images) if any(x.endswith(e) for e in ('.png', '.jpg'))])
num_imgs = args.count if args.count else len(filenames)
#filenames = filenames[::3]
filenames = filenames[args.offset:args.offset+num_imgs]
filenames = [os.path.join(args.images, f) for f in filenames]
if args.color:
    imgs = list(tqdm((common.imread(f, as_float=False) for f in filenames), total=len(filenames)))
else:
    imgs = list(tqdm((cv2.imread(f, 0) for f in filenames), total=len(filenames)))
print()

print("Cropping images...")
# crop images
crop_left, crop_right, crop_top, crop_bottom = [int(x) for x in args.crop.split(",")]
crop_bottom = imgs[0].shape[0] - crop_bottom
crop_right = imgs[0].shape[1] - crop_right
imgs = [x[crop_top:crop_bottom,crop_left:crop_right] for x in imgs]
print()

print("Computing homographies...")
if args.homography_cache and os.path.exists(args.homography_cache):
    with open(args.homography_cache, 'rb') as f:
        Ts = pickle.load(f)
else:
    with Timer() as t:
        Ts = pairwise_homographies(imgs)
    print("computed {} homographies in {}".format(len(Ts), t.elapsed))
    print()
    if args.homography_cache:
        with open(args.homography_cache, 'wb') as f:
            pickle.dump(Ts, f)

smooth_sigma = args.stabilization
vanishings = []
h, w = imgs[0].shape[0:2]
for i in range(len(Ts)):
    H = functools.reduce(np.dot, Ts[i:])
    vanishings.append(get_vanishing_pt(H, h, w))
vanishing_pts = np.vstack(vanishings)
kernel = scipy.signal.gaussian(int(smooth_sigma * 7), std=smooth_sigma)
kernel /= np.sum(kernel)
target_vanishing_pts = scipy.ndimage.filters.convolve1d(vanishing_pts, kernel, axis=0, mode='nearest')

#sys.exit(0)

print("Compositing frames...")
with Timer() as t:
    frames = final_video(imgs, Ts, target_vanishing_pts, speed=args.speed, draw_distance=args.draw_distance, mask_r=args.mask_radius)
print("computed {} frames in {}".format(len(frames), t.elapsed))
print()

print("Saving video...")
common.videosave(args.output, frames, fps=60)