import os
import shutil
import subprocess

import click
import numpy as np
import tensorflow as tf

from colmap import load_colmap_data, run_colmap, save_poses


def get_blur_score(image_path):
    """Calculate bluriness score with FFT
    https://www.pyimagesearch.com/2020/06/15/opencv-fast-fourier-transform-fft-for-blur-detection-in-images-and-video-streams/
    """
    size = 60
    encoded = tf.io.read_file(image_path)
    image = tf.io.decode_image(encoded)
    image = tf.image.resize(image, (500, 500), preserve_aspect_ratio=True)
    image = tf.squeeze(tf.image.rgb_to_grayscale(image), -1).numpy()
    h = image.shape[0]
    w = image.shape[1]
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    magnitude = 20 * np.log(np.abs(recon).clip(min=1e-8))
    mean = np.mean(magnitude)

    return mean, image_path


def filter_blurry_images(frames_dir, output_dir, threshold):
    images = os.listdir(frames_dir)
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    results = []
    for image_name in images:
        if image_name.endswith('.jpg') or image_name.endswith('.png'):
            image_path = os.path.join(frames_dir, image_name)
            results.append(get_blur_score(image_path))

    cnt = 0
    for score, image_path in results:
        if score < threshold:
            cnt += 1
            continue
        output_path = os.path.join(images_dir, os.path.basename(image_path))
        shutil.copy(image_path, output_path)
    print(
        f'Filtered out {cnt} blurry images out of {len(results)} total images.'
    )


def generate_poses(basedir, colmap_location, match_type='exhaustive_matcher'):

    files_needed = [
        '{}.bin'.format(f) for f in ['cameras', 'images', 'points3D']
    ]
    if os.path.exists(os.path.join(basedir, 'sparse/0')):
        files_had = os.listdir(os.path.join(basedir, 'sparse/0'))
    else:
        files_had = []
    if not all([f in files_had for f in files_needed]):
        print('Need to run COLMAP')
        run_colmap(basedir, colmap_location, match_type)
    else:
        print('Don\'t need to run COLMAP')

    print('Post-colmap')

    poses, pts3d, perm = load_colmap_data(basedir)

    save_poses(basedir, poses, pts3d, perm)

    print('Done with imgs2poses')

    return True


def extract_raw_frames(video_path, frames_dir, fps):
    subprocess.run([
        'ffmpeg', '-i', video_path, '-r', f'{fps}/1',
        f'{frames_dir}/frame%03d.png'
    ])


@click.command()
@click.option('--video', help='Video path.')
@click.option('--output-dir', help='Directory for output images.')
@click.option('--threshold', default=15, help='Bluriness threshold.')
@click.option('--fps', default=5, help='Frames per second.')
@click.option('--colmap',
              default='/Applications/COLMAP.app/Contents/MacOS/colmap',
              help='The path to COLMAP executable.')
def main(video, output_dir, colmap, threshold, fps):
    frames_dir = './temp/frames'
    os.makedirs(frames_dir, exist_ok=True)

    try:
        extract_raw_frames(video, frames_dir, fps)
        filter_blurry_images(frames_dir, output_dir, threshold)
        generate_poses(output_dir, colmap)
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)
        os.rmdir(frames_dir)


if __name__ == "__main__":
    main()