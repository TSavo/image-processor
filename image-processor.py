import cv2
import numpy as np
import argparse
import warnings
import os
from os import path
import tqdm
warnings.filterwarnings("ignore")
from glob import glob
import numpy as np
import oxipng
import math


def black_border(image, size):
    h, w = image.shape[:2]
    top, bottom = 0, 0
    left, right = 0, 0
    if h < size:
        top = size // 2 - h // 2
        bottom = size - h - top
    elif w < size:
        left = size // 2 - w // 2
        right = size - w - left
    else:
        top = (h - size) // 2
        bottom = h - size - top
        left = (w - size) // 2
        right = w - size - left
    # add a black border using cv2.BORDER_CONSTANT
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return image

def center_crop(image, size):
    h, w = image.shape[:2]
    y = h//2-(size//2)
    x = w//2-(size//2)
    return image[y:y+size, x:x+size]

def resize(image, new_length):
    h, w = image.shape[:2]
    if h > w:
        # resize the image by specifying the new height
        image = cv2.resize(image, (int(w * new_length / h), new_length), interpolation=cv2.INTER_NEAREST)
    else:
        # resize the image by specifying the new width
        image = cv2.resize(image, (new_length, int(h * new_length / w)), interpolation=cv2.INTER_NEAREST)
    return image

from sklearn.cluster import KMeans
def find_mean_dim_color(image, num_clusters=15, subsample=.25):
    #subsample the image
    image = cv2.resize(image, (0,0), fx=subsample, fy=subsample)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clt = KMeans(n_clusters = num_clusters, n_init=20)
    clt.fit_predict(image)    
    dominant_color = clt.cluster_centers_[np.argmin(clt.cluster_centers_.mean(axis=1))]

    return max(25, min(40,dominant_color.mean().astype(int) + 3))


def mask_edges(image, num_pixels, blur=0):
    height, width, _ = image.shape
    x = num_pixels
    # Create a black image with the same dimensions as the original image
    mask = np.zeros((height, width), dtype="uint8")
    # Draw a white rectangle on the black image to create a black border around the edges
    cv2.rectangle(mask, (x, x), (width - x, height - x), 255, -1)
    # Apply a Gaussian blur to the mask to soften the edges
    if blur > 0:
        if blur % 2 == 0:
            blur += 1
        mask = cv2.GaussianBlur(mask, (blur,blur), blur * 2000000)
    #invert the mask
    mask = cv2.bitwise_not(mask)
    # Convert the mask to 3 channels
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.imwrite("mask.png", mask)
    # Apply the mask to the image
    masked_image = np.maximum(cv2.subtract(image.astype(np.int16), mask.astype(np.int16)), (0,0,0)).astype(np.uint8)
    return masked_image


def vignette(image, level = 1): 
    height, width = image.shape[:2]
    x_resultant_kernel = cv2.getGaussianKernel(width, width/level)
    y_resultant_kernel = cv2.getGaussianKernel(height, height/level)
    kernel = y_resultant_kernel * x_resultant_kernel.T
    mask = kernel / kernel.max()
    image = image * mask[:, :, np.newaxis]
    return image

def vignette_sides(image, dim_factor=0.5):
    height, width = image.shape[:2]

    # Create the vignette mask using absolute horizontal distance
    x = np.linspace(-1, 1, width)
    vignette_mask = 1 - np.abs(np.tile(x, (height, 1)))

    # Apply the mask to the image
    image_vignette = np.copy(image)
    # Multiply the vignette mask with dim factor and add 1 - dim_factor
    vignette_mask = vignette_mask * dim_factor + (1 - dim_factor)

    # Multiply the image with the vignette mask and apply maximum with 0
    image_vignette = np.maximum(image_vignette * vignette_mask[:,:,np.newaxis], 0)
    return image_vignette

def vignette_corners(image, dim_factor=0.5):
    height, width = image.shape[:2]

    # Create the vignette mask using absolute distance
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    x, y = np.meshgrid(x, y)
    vignette_mask = 1 - np.sqrt(x * x + y * y)

    # Apply the mask to the image
    image_vignette = np.copy(image)
    for i in range(3):
        image_vignette[:,:,i] = np.maximum(image_vignette[:,:,i] * (vignette_mask * dim_factor + (1 - dim_factor)), 0)
    return image_vignette

def adjust_brightness_contrast(image, brightness, contrast):
    # Calculate the actual brightness and contrast values
    alpha = contrast * 2 + 1
    beta = brightness * 255

    # Scale and shift the pixel values of the image
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Clamp the pixel values to the valid range
    adjusted_image = np.clip(adjusted_image, 0, 255)

    # Return the adjusted image
    return adjusted_image

def compute_transparency_mask(image, threshold):
    # Convert the image to grayscale
    grayscaled = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # Apply a threshold to the grayscaled image
    _, threshold = cv2.threshold(grayscaled, threshold, 255, cv2.THRESH_BINARY)

    # Return the mask
    return threshold

def blur_mask(mask, sigmaX, sigmaY):
    # Blur the mask
    mask = cv2.GaussianBlur(mask, (0,0), sigmaX=sigmaX, sigmaY=sigmaY, borderType = cv2.BORDER_DEFAULT)

    # Return the blurred mask
    return mask

def apply_transparency_mask(image, mask):
    alpha = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    alpha[:,:,3] = mask
    return alpha

def apply_black_mask(image, mask):
    image = cv2.bitwise_and(image, image, mask=mask)
    return image

def sharpen(image):
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    return image_sharp


def adjust_saturation(image, saturation):
    # Convert the image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Scale the saturation
    hsv[:,:,1] = np.clip(hsv[:,:,1] * saturation, 0, 255)

    # Convert the image back to BGR
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Return the image
    return image
import svgwrite


def compress(image_file, image, progress, target_size = 49 * 1024 * 1024):
    #encode the image as a png
    encoded = cv2.imencode('.png', image)[1]
    progress.write(f"Original size: {humanbytes(len(encoded.tobytes()))} bytes")
    #optimize the image with oxipng
    progress.write("Optimizing image with level " + ((str(6) + " (this may take a while)") if len(encoded.tobytes()) > target_size else str(2)) + "...")
    optimized_data = oxipng.optimize_from_memory(encoded.tobytes(), level = 6 if len(encoded.tobytes()) > target_size else 2)
    #use humanbytes() to format the bytes
    progress.write(f"Optimized size: {humanbytes(len(optimized_data))} bytes")
    while len(optimized_data) > target_size:
        #compute the optimal reduction factor
        reduction_factor = math.sqrt(target_size / len(optimized_data)) - 0.005
        progress.write(f"Optimized image is too big, resizing by {reduction_factor * 100:.2f}%")
        image = cv2.resize(image, (0,0), fx=reduction_factor, fy=reduction_factor, interpolation=cv2.INTER_AREA)
        progress.write(f"Resizing to {image.shape[1]}x{image.shape[0]} pixels")
        #encode the image as a png
        encoded = cv2.imencode('.png', image)[1]
        progress.write(f"Original size: {humanbytes(len(encoded.tobytes()))} bytes")
        #optimize the image with oxipng
        progress.write("Optimizing image with level 6 (this may take a while)...")
        optimized_data = oxipng.optimize_from_memory(encoded.tobytes(), level=6)        
        progress.write(f"Optimized size: {humanbytes(len(optimized_data))} bytes")
    #if the directory doesn't exist, create it
    if not os.path.exists(os.path.dirname(image_file)):
        os.makedirs(os.path.dirname(image_file))
    #save the image
    with open(image_file, "wb") as f:
        f.write(optimized_data)


#write out bytes as human readable
def humanbytes(B):
    #return the given bytes as a human friendly KB, MB, GB, or TB string
    B = float(B)
    KB = float(1024)
    MB = float(KB ** 2) # 1,048,576
    GB = float(KB ** 3) # 1,073,741,824
    TB = float(KB ** 4) # 1,099,511,627,776
    PB = float(KB ** 5) # 1,125,899,906,842,624
    EB = float(KB ** 6) # 1,152,921,504,606,846,976
    ZB = float(KB ** 7) # 1,180,591,620,717,411,303,424
    YB = float(KB ** 8) # 1,208,925,819,614,629,174,706,176
    if B < KB:
        return '{0} {1}'.format(B,'Bytes' if 0 == B > 1 else 'Byte')
    elif KB <= B < MB:
        return '{0:.2f} KB'.format(B/KB)
    elif MB <= B < GB:
        return '{0:.2f} MB'.format(B/MB)
    elif GB <= B < TB:
        return '{0:.2f} GB'.format(B/GB)
    elif TB <= B < PB:
        return '{0:.2f} TB'.format(B/TB)
    elif PB <= B < EB:
        return '{0:.2f} PB'.format(B/PB)
    elif EB <= B < ZB:
        return '{0:.2f} EB'.format(B/EB)
    elif ZB <= B < YB:
        return '{0:.2f} ZB'.format(B/ZB)
    elif YB <= B:
        return '{0:.2f} YB'.format(B/YB)

def process_image(image_file, progress, args):
    # Read the image
    image = cv2.imread(image_file, cv2.IMREAD_COLOR)
    progress.write(f"Working on {image_file}, which is {image.shape[1]}x{image.shape[0]} pixels and {humanbytes(os.path.getsize(image_file))} bytes")
    if args.resize > 0:
        progress.write("Resizing...")
        image = resize(image, args.resize)
        progress.write(f"Resized to {image.shape[1]}x{image.shape[0]} pixels")

    #if crop is set, crop the image
    if args.crop > 0:
        h, w = image.shape[:2]
        progress.write("Cropping...")
        if h < args.crop and w < args.crop:
            image = black_border(image, args.crop)
        else:
            image = center_crop(image, args.crop)
        progress.write(f"Cropped to {image.shape[1]}x{image.shape[0]} pixels")

    #if vignette is set, apply a vignette
    if args.vignette > 0:
        progress.write("Vignetting...")
        image = vignette(image, args.vignette)
        progress.write(f"Vignette applied")

    #if vignette_sides is set, apply a vignette to the sides
    if args.vignette_sides > 0:
        progress.write("Vignetting sides...")
        image = vignette_sides(image, args.vignette_sides)
        progress.write(f"Vignette applied")

    #if vignette_corners is set, apply a vignette to the corners
    if args.vignette_corners > 0:
        progress.write("Vignetting corners...")
        image = vignette_corners(image, args.vignette_corners)
        progress.write(f"Vignette applied")

    #if sharpen is set, sharpen the image
    if args.sharpen:
        progress.write("Sharpening...")
        image = sharpen(image)
        progress.write(f"Sharpened")

    #if brightness or contrast is set, adjust the brightness and contrast
    if args.brightness != 0 or args.contrast != 0:
        progress.write("Adjusting brightness and contrast...")
        image = adjust_brightness_contrast(image, args.brightness, args.contrast)
        progress.write(f"Brightness and contrast adjusted")

    #if saturation is set, adjust the saturation
    if args.saturation != 0:
        progress.write("Adjusting saturation...")
        image = adjust_saturation(image, args.saturation)
        progress.write(f"Saturation adjusted")

    #if threshold is not set, compute the transparency mask
    threshold = args.threshold
    if args.threshold == -1:
        progress.write("Computing background color threshold...")
        threshold = find_mean_dim_color(image)
        progress.write(f"The suggested threshold value color is: {threshold}")
    
    #if mask is set, manually mask the edges
    if args.mask > 0:
        progress.write("Manually masking edges...")
        image = mask_edges(image, args.mask, args.blur_mask)
        
    progress.write("Computing black mask...")
    mask = compute_transparency_mask(image, threshold)
 
    #if blur is set, blur the mask
    if args.blur > 0:
        progress.write("Blurring mask...")
        mask = blur_mask(mask, args.blur, args.blur)


    progress.write("Applying black mask...")
    black = apply_black_mask(image, mask)
    dirname = path.dirname(image_file)  + "/black"
    filename = path.basename(image_file).replace(".jpg", ".png").replace(".jpeg", ".png")
    progress.write(f"Saving black image to {dirname}/black-{filename}...")
    # Save the transparent image
    compress(dirname + "/black-" + filename, black, progress)

    progress.write("Applying transparency mask...")
    transparent = apply_transparency_mask(image, mask)
    dirname = path.dirname(image_file)  + "/transparent"
    progress.write(f"Saving transparent image to {dirname}/transparent-{filename}...")
    # Save the transparent image
    compress(dirname + "/transparent-" + filename, transparent, progress)


if __name__ == "__main__":
    TOP = 0
    BOTTOM = 1
    LEFT = 2
    RIGHT = 3
    
    # Create a parser object
    parser = argparse.ArgumentParser()

    # Add a required image_file argument
    parser.add_argument("--input_file", "-i", help="the name of the image file to process", type=str)

    # Add a required output_file argument
    parser.add_argument("--output_file", "-o", help="the name of the output file", type=str, default="output.png")

    # Add an optional threshold argument with a default value of None
    parser.add_argument("--threshold", "-t", help="the threshold value to use for transparency, between 0 and 255 (0 being only black and 255 being everything, try 6)", type=int, default=-1)

    # Add an optional vignetting_factor argument with a default value of 2
    parser.add_argument("--vignette", "-v", help="the amount of vignetting to apply", type=float, default=0)

    parser.add_argument("--vignette_sides", "-vs", help="the amount of side vignetting to apply", type=float, default=0)

    parser.add_argument("--vignette_corners", "-vc", help="the amount of corner vignetting to apply", type=float, default=0)

    #Add an optional contrast argument with a default value of 1.75
    parser.add_argument("--contrast", "-c", help="the amount of contrast to apply", type=float, default=0)
    parser.add_argument("--brightness", "-b", help="the amount of brightness to apply", type=float, default=0)
    parser.add_argument("--mask", "-m", help="the number of pixels to mask off the edges of the image", type=int, default=0)
    parser.add_argument("--blur_mask", "-bm", help="the amount of blur to apply to the mask", type=int, default=0)
    parser.add_argument("--blur", "-u", help="the amount of blur to apply to the transparency mask", type=int, default=0)
    parser.add_argument("--sharpen", "-s", help="To sharpen or not, default False", type=bool, default=False)
    parser.add_argument("--crop", "-cr", help="The amount to clip off the edges of the image", type=int, default=0)
    parser.add_argument("--resize", "-r", help="The amount to resize the image by", type=int, default=0)
    parser.add_argument("--saturation", "-sa", help="The amount to saturate the image by", type=float, default=0)
    # Parse the command line arguments
    args = parser.parse_args()
    if path.isdir(args.input_file):
        files = glob(args.input_file + "/*", recursive=True)
    else:
        files = [args.input_file]
    to_process = tqdm.tqdm(files)
    for image_file in to_process:
        if path.isdir(image_file) or "\\transparent\\" in image_file or "\\black\\" in image_file:
            continue
        to_process.write(f"Processing {image_file}...")
        process_image(image_file, to_process, args)

