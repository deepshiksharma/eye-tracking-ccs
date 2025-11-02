import sys, ast
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Check command line args
if len(sys.argv) != 3:
    print("Incorrect usage.")
    print("Usage: python generate_gaze_heatmap.py <path_to_eye_tracker_data.csv> <path_to_overlay_image>")
    sys.exit(1)


def plot_gaze_heatmap(df, image_path, save_path="./heatmap.png", display_size=(1920, 1080),
                      blur_sigma=50, cmap='jet', overlay_alpha=0.8, background_visible=0.6):
    """
    Generate gaze heatmap overlay on image.
    
    Args:
        - df (pd.Dataframe):  Dataframe containing eye-tracking data.
        - image_path (str):   Path to image on which to overlay heatmap (what the subject was viewing).
        - save_path (str, optional):  Path to save heatmap. Defaults to "./heatmap.png".
        - display_size (tuple(screen_width_px, screen_height_px), optional):  Size of the display monitor used, in pixels.
                                                                              Defaults to (1920, 1080).
        - blur_sigma (int, optional):  Defines the standard deviation of Gaussian kernel used to blur the heatmap.
                                       Larger values → broader, smoother heat spots; Smaller values → sharper, more localized gaze points.
                                       Defaults to 50; passed to scipy.ndimage.gaussian_filter
        - cmap (str, optional. Example values: 'jet'|'viridis'|'plasma'|'hot'): Defines color mapping used to visualize gaze intensity.
                                                                                Defaults to 'jet'; passed to matplotlib.cm.get_cmap
        - overlay_alpha (float 0..1, optional):  Controls transparency of the heatmap overlay. Higher value → Heatmap more opaque, colors stronger.
                                                 Defaults to 0.8
        - background_visible (float 0..1, optional):  Controls how much of the original image brightness is kept before overlay.
                                                      1 → background fully visible; 0 → background completely blacked out. Defaults to 0.6
    """
    
    # helpers
    def parse_point(val):
        if pd.isna(val):
            return None
        if isinstance(val, (tuple, list)) and len(val) >= 2:
            try:
                return (float(val[0]), float(val[1]))
            except Exception:
                return None
        if isinstance(val, (int, float)):
            return None
        if isinstance(val, str):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, (tuple, list)) and len(parsed) >= 2:
                    return (float(parsed[0]), float(parsed[1]))
            except Exception:
                s = val.strip().strip("()[]")
                parts = [p.strip() for p in s.split(",")]
                if len(parts) >= 2:
                    try:
                        return (float(parts[0]), float(parts[1]))
                    except Exception:
                        return None
        return None

    def valid_flag(v):
        if v is None or (pd.isna(v)):
            return True
        try:
            s = str(v).strip()
            if s.isdigit():
                return bool(int(s))
            return bool(v)
        except Exception:
            return bool(v)

    # load image
    width, height = display_size
    img = Image.open(str(image_path)).convert("RGBA")
    if img.size != (width, height):
        img = img.resize((width, height), resample=Image.LANCZOS)

    # gather gaze points (pixel co-ordinates)
    points = []
    for _, row in df.iterrows():
        l = parse_point(row.get('left_gaze_point_on_display_area'))
        r = parse_point(row.get('right_gaze_point_on_display_area'))
        lv = valid_flag(row.get('left_gaze_point_validity', None))
        rv = valid_flag(row.get('right_gaze_point_validity', None))

        pt = None
        if l and r and lv and rv:
            pt = ((l[0] + r[0]) / 2.0, (l[1] + r[1]) / 2.0)
        elif l and lv:
            pt = l
        elif r and rv:
            pt = r

        if pt is None:
            continue

        px, py = pt
        x_px = px * width
        y_px = py * height

        xi = int(round(x_px))
        yi = int(round(y_px))
        if 0 <= xi < width and 0 <= yi < height:
            points.append((xi, yi))

    if len(points) == 0:
        raise ValueError("No valid gaze points found in dataframe.")

    # raw accumulator
    heat = np.zeros((height, width), dtype=np.float32)
    for x, y in points:
        heat[y, x] += 1.0

    # smooth: try scipy, fallback to cv2, final numeric blur
    smoothed = gaussian_filter(heat, sigma=blur_sigma, mode='constant')

    # normalize to 0..1
    if smoothed.max() > 0:
        heat_norm = smoothed / smoothed.max()
    else:
        heat_norm = smoothed

    # build colored heatmap RGBA (alpha = intensity * overlay_alpha)
    import matplotlib.cm as cm
    cmap_obj = cm.get_cmap(cmap)
    colored = cmap_obj(heat_norm)  # RGBA floats 0..1
    colored[..., 3] = np.clip(heat_norm * overlay_alpha, 0.0, 1.0)
    heatmap_img = Image.fromarray((colored * 255).astype(np.uint8), mode='RGBA')

    # prepare background: dim/blend toward black so heatmap stands out
    black = Image.new('RGBA', img.size, (0, 0, 0, 255))
    # Image.blend(A,B,alpha) => A*(1-alpha) + B*alpha
    dim_alpha = 1.0 - background_visible
    base_dim = Image.blend(img, black, alpha=dim_alpha)

    # composite: heatmap foreground over dimmed background
    heatmap_resized = heatmap_img.resize(base_dim.size, resample=Image.BILINEAR)
    composite = Image.alpha_composite(base_dim, heatmap_resized)

    # save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(str(save_path))

    # plt.figure(figsize=(10, 6))
    plt.imshow(composite)
    plt.axis('off')
    plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    data = pd.read_csv(sys.argv[1])
    plot_gaze_heatmap(data, sys.argv[2])
