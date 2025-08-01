import os, sys
import numpy as np
from skimage.io import imread, imsave
from skimage import img_as_float, img_as_ubyte
from scipy.fft import fft2, ifft2
from tqdm import tqdm


def main(img_dir):
    """
    Normalize all images to have the same mean pixel intensity and RMS contrast.
    RMS contrast = std / mean
    """
    normalize_dir_recursive(img_dir)

    """
    Created Fourier-phase scrambled versions of each image.
    These scrambled images maintain the frequency spectrum and colours of the original image,
    but the content cannot be perceived.
    """
    scramble_dir_recursive()


def get_img_paths(root_folder):
    img_paths = list()
    
    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            if fname.lower().endswith('.jpg'):
                full_path = os.path.join(dirpath, fname)
                img_paths.append(full_path)
    
    return img_paths


def compute_global_stats(img_paths):
    means, stds = [], []

    for path in img_paths:
        img = img_as_float(imread(path))
        means.append(np.mean(img))
        stds.append(np.std(img))

    global_mean = np.mean(means)
    global_std = np.mean(stds)

    # Standard RMS contrast: RMS = std / mean
    desired_rms_contrast = global_std / global_mean
    # Solve for target std: std = mean * desired_rms
    target_std = global_mean * desired_rms_contrast

    return global_mean, target_std


def normalize_img(img, target_mean, target_std):
    current_mean = np.mean(img)
    current_std = np.std(img)
    
    if current_std == 0:
        return np.full_like(img, target_mean)

    normalized = (img - current_mean) / current_std
    normalized = normalized * target_std + target_mean
    
    return np.clip(normalized, 0, 1)


def normalize_dir_recursive(input_dir):
    output_dir = "normalized_images"
    os.makedirs(output_dir, exist_ok=True)

    # 1: Collect all image paths
    img_paths = get_img_paths(input_dir)
    # 2: Compute global mean and std
    target_mean, target_std = compute_global_stats(img_paths)

    # 3: Normalize each image and save to corresponding subfolder in output
    for path in tqdm(img_paths, desc="Normalizing images"):
        img = img_as_float(imread(path))
        normalized_img = normalize_img(img, target_mean, target_std)

        # Reconstruct relative path to preserve folder structure
        rel_path = os.path.relpath(path, input_dir)
        output_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        imsave(output_path, img_as_ubyte(normalized_img))


def match_intensity_contrast(img, ref_img):
    mean_ref = np.mean(ref_img)
    std_ref = np.std(ref_img)
    img_matched = (img - np.mean(img)) / np.std(img) * std_ref + mean_ref
    
    return np.clip(img_matched, 0, 1)


def scramble_img(img):
    img = img_as_float(img)
    h, w, _ = img.shape

    # Use the same random phase for all channels
    random_phase = np.angle(fft2(np.random.rand(h, w)))

    scrambled_channels = []
    for c in range(3):  # R, G, B channels
        channel = img[:, :, c]

        # FFT
        F = fft2(channel)
        amplitude = np.abs(F)

        # Combine amplitude with shared random phase
        scrambled_F = amplitude * np.exp(1j * random_phase)

        # Inverse FFT
        scrambled_channel = np.real(ifft2(scrambled_F))

        # Match intensity and contrast with original channel
        scrambled_matched = match_intensity_contrast(scrambled_channel, channel)
        scrambled_channels.append(scrambled_matched)

    scrambled_img = np.stack(scrambled_channels, axis=2)
    return np.clip(scrambled_img, 0, 1)


def scramble_dir_recursive():
    input_dir = "normalized_images"
    output_dir = "scrambled_images"
    os.makedirs(output_dir, exist_ok=True)

    # 1: Collect all image paths
    img_paths = get_img_paths(input_dir)

    # 2: Scramble each image
    for path in tqdm(img_paths, desc="Scrambling images"):
        img = img_as_float(imread(path))
        scrambled_img = scramble_img(img)

        # Reconstruct relative path to preserve folder structure
        rel_path = os.path.relpath(path, input_dir)
        output_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        imsave(output_path, img_as_ubyte(scrambled_img))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Incorrect usage.\n" \
        "Correct usage: python normalize_and_scramble <stimuli_images_directory>")
        sys.exit(1)
    
    STIM_DIR = sys.argv[1]

    if not os.path.exists(STIM_DIR):
        print(f"The directory \"{STIM_DIR}\" does not exist in current path.")
        sys.exit(1)

    main(STIM_DIR)
