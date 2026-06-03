import numpy as np
from PIL import Image, ImageDraw

def save_png(img, path):
    Image.fromarray(np.flipud(img).astype(np.uint8)).save(path)

def label_frame(img, text):
    im = Image.fromarray(np.asarray(img).astype(np.uint8))
    ImageDraw.Draw(im).text((5, 5), text, fill=(255, 255, 0))
    return np.asarray(im)

def write_video(frames, path_noext, fps):
    import imageio
    try:
        path = path_noext + ".mp4"
        imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
        return path
    except Exception as e:
        path = path_noext + ".gif"
        imageio.mimsave(path, frames, fps=fps)
        print(f"mp4 unavailable ({type(e).__name__}), wrote gif")
        return path
