import os

import numpy as np
from numpy.linalg.linalg import norm
import tensorflow as tf


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def poses_avg(poses):
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    camera2world = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    return camera2world


def recenter_poses(poses):
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    camera2world = poses_avg(poses)
    camera2world = np.concatenate([camera2world[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)
    poses = np.linalg.inv(camera2world) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.0])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array(
                [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) *
            rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def load_data(basedir, factor=8, bound_factor=0.75):
    poses_array = np.load(os.path.join(basedir, 'poses_bounds.npy'))

    # for M images, poses -> [3, 5, M]
    poses = poses_array[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])

    # bounds -> [2, M]
    bounds = poses_array[:, -2:].transpose([1, 0])

    print('Loaded', basedir, bounds.min(), bounds.max())

    imgdir = os.path.join(basedir, 'images')

    images = []
    sh = None
    for f in sorted(os.listdir(imgdir)):
        if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png'):
            image_path = os.path.join(imgdir, f)
            image = tf.io.decode_image(tf.io.read_file(image_path))
            images.append(image)
            if sh is None:
                sh = np.array([image.shape[0], image.shape[1]]) / 8
    poses[:2, 4, :] = sh[:2].reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor
    images = np.stack(images, -1)

    poses = np.concatenate(
        [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    images = np.moveaxis(images, -1, 0).astype(np.float32)
    bounds = np.moveaxis(bounds, -1, 0).astype(np.float32)

    scale = 1. if bound_factor is None else 1. / (bounds.min() * bound_factor)
    poses[:, :3, 3] *= scale
    bounds *= scale

    poses = recenter_poses(poses)

    camera2world = poses_avg(poses)
    print('recentered', camera2world.shape)
    print(camera2world[:3, :4])

    up = normalize(poses[:, :3, 1].sum(0))

    close_depth, inf_depth = bounds.min() * 0.9, bounds.max() * 0.5
    dt = 0.75
    mean_dz = 1 / ((1 - dt) / close_depth + dt / inf_depth)
    focal = mean_dz

    shrink_factor = 0.8
    zdelta = close_depth * 0.2
    tt = poses[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0)
    c2w_path = camera2world
    N_views = 120
    N_rots = 2

    render_poses = render_path_spiral(c2w_path,
                                      up,
                                      rads,
                                      focal,
                                      zdelta,
                                      zrate=.5,
                                      rots=N_rots,
                                      N=N_views)

    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bounds.shape)

    dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)
    i_test = np.argmin(dists)

    return images, poses, bounds, render_poses, i_test


load_data(
    '/Users/yanjia.li/Snapchat/Dev/deep-vision/NeRF/tensorflow/data/frames')
