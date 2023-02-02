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
import shutil
from sklearn.cluster import MiniBatchKMeans
from upscaler import TargetUpscaler
import sys

def count_dim_pixels(image, threshold=50):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.sum(image < threshold)

def count_bright_pixels(image, threshold=230):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.sum(image > threshold)


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


def find_mean_color(img, is_black, num_clusters=8, subsample=.25):
    # Subsample the image if specified
    if subsample < 1:
        img = cv2.resize(img, (0, 0), fx=subsample, fy=subsample, interpolation=cv2.INTER_NEAREST)
    # Convert the image to a single-channel, floating point representation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray= gray.reshape(-1,1)
    # Reshape the image into a 2D array of shape (m * n, 1)
    # Apply K-Means clustering to the data, specifying the number of clusters to be 2

    kmeans = MiniBatchKMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(gray)
    # Create an image that will be used to visualize the clusters
    clustered_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    colors = []
    for i in range(num_clusters):
        h = i / num_clusters
        s = 1.0
        v = 1.0
        color = np.uint8([[[h * 180, s * 255, v * 255]]])
        color = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)
        color = color[0][0].tolist()
        colors.append(color)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            clustered_image[i, j] = colors[labels[i * img.shape[1] + j]]
    cv2.imwrite("clustered.png", clustered_image.reshape(img.shape[0], img.shape[1], 3))
    # Determine which cluster represents the background
    if is_black:
        dominant_cluster = np.argmin(kmeans.cluster_centers_.mean(axis=1))
    else:
        dominant_cluster = np.argmax(kmeans.cluster_centers_.mean(axis=1))

    pixel_indices = np.where(labels == dominant_cluster)
    print(f"Found {len(pixel_indices[0])} pixels in dominant cluster")
    dominant_pixels = gray[pixel_indices].astype(np.float)
    mean_color = dominant_pixels.max().astype(np.uint8)
    return mean_color

def mask_edges(image, num_pixels, is_black, blur=0):
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
    # Apply the mask to the image
    if is_black:
        masked_image = np.maximum(cv2.subtract(image.astype(np.int16), mask.astype(np.int16)), (0,0,0)).astype(np.uint8)
    else:
        masked_image = np.minimum(cv2.add(image.astype(np.int16), mask.astype(np.int16)), (255,255,255)).astype(np.uint8)
    return masked_image

def vignette(image, level = 1, is_black=True): 
    height, width = image.shape[:2]
    x_resultant_kernel = cv2.getGaussianKernel(width, width/level)
    y_resultant_kernel = cv2.getGaussianKernel(height, height/level)
    kernel = y_resultant_kernel * x_resultant_kernel.T
    mask = kernel / kernel.max()
    if is_black:
        image = image * mask[:, :, np.newaxis]
    else:
        image = image * (1 - mask[:, :, np.newaxis])
    return image

def vignette_sides(image, dim_factor=0.5, is_black=True):
    height, width = image.shape[:2]

    # Create the vignette mask using absolute horizontal distance
    x = np.linspace(-1, 1, width)
    vignette_mask = 1 - np.abs(np.tile(x, (height, 1)))

    # Apply the mask to the image
    image_vignette = np.copy(image)
    if is_black:
        vignette_mask = vignette_mask * dim_factor + (1 - dim_factor)
        image_vignette = np.maximum(image_vignette * vignette_mask[:,:,np.newaxis], 0)
    else:
        vignette_mask = vignette_mask * (1 - dim_factor)
        image_vignette = np.minimum(image_vignette * vignette_mask[:,:,np.newaxis], 255)
    return image_vignette

def vignette_corners(image, dim_factor=0.5, is_black = True):
    height, width = image.shape[:2]

    # Create the vignette mask using absolute horizontal and vertical distance
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    x, y = np.meshgrid(x, y)
    vignette_mask = 1 - np.sqrt(x**2 + y**2)

    # Apply the mask to the image
    image_vignette = np.copy(image)
    if is_black:
        vignette_mask = vignette_mask * dim_factor + (1 - dim_factor)
        image_vignette = np.maximum(image_vignette * vignette_mask[:,:,np.newaxis], 0)
    else:
        vignette_mask = vignette_mask * (1 - dim_factor)
        image_vignette = np.minimum(image_vignette * vignette_mask[:,:,np.newaxis], 255)
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

def compute_transparency_mask(image, threshold, is_black=True):
    # Convert the image to grayscale
    grayscaled = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # Apply a threshold to the grayscaled image
    if is_black:
        _, threshold = cv2.threshold(grayscaled, threshold, 255, cv2.THRESH_BINARY)
    else:
        _, threshold = cv2.threshold(grayscaled, threshold, 255, cv2.THRESH_BINARY_INV)

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

def apply_mask(image, mask, is_black=True):
    # Apply the mask to the image
    masked_image = np.copy(image)
    if is_black:
        masked_image[mask == 0] = 0
    else:
        masked_image[mask == 0] = 255
    return masked_image
    
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

def adjust_hue(image, hue):
    # Convert the image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Scale the hue
    hsv[:,:,0] = np.clip(hsv[:,:,0] * hue, 0, 255)

    # Convert the image back to BGR
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Return the image
    return image

def adjust_gamma(image, gamma):
    # Build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # Apply gamma correction using the lookup table
    image = cv2.LUT(image, table)

    # Return the adjusted image
    return image

def adjust_sharpness(image, sharpness):
    # Create the sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1, 9 + sharpness, -1],
                       [-1, -1, -1]])

    # Apply the kernel to the image
    image = cv2.filter2D(image, -1, kernel)

    # Return the sharpened image
    return image

def adjust_blur(image, blur):
    # Create the blurring kernel
    kernel = np.ones((blur, blur), np.float32) / (blur * blur)

    # Apply the kernel to the image
    image = cv2.filter2D(image, -1, kernel)

    # Return the blurred image
    return image

def adjust_noise(image, noise):
    # Add noise to the image
    image = image + noise * np.random.randn(*image.shape)

    # Return the noisy image
    return image

def adjust_grain(image, grain):
    # Add grain to the image
    image = image + grain * np.random.randn(*image.shape)

    # Return the grainy image
    return image


def adjust_contrast(image, contrast):
    # Calculate the mean of the image
    mean = np.mean(image)

    # Apply the contrast to the image
    image_contrast = np.copy(image)
    for i in range(3):
        image_contrast[:,:,i] = np.clip((image_contrast[:,:,i] - mean) * contrast + mean, 0, 255)

    # Return the contrast image
    return image_contrast

def adjust_brightness(image, brightness):
    # Apply the brightness to the image
    image_brightness = np.copy(image)
    for i in range(3):
        image_brightness[:,:,i] = np.clip(image_brightness[:,:,i] * brightness, 0, 255)

    # Return the brightness image
    return image_brightness
def adjust_exposure(image, exposure):
    # Apply the exposure to the image
    image_exposure = np.copy(image)
    for i in range(3):
        image_exposure[:,:,i] = np.clip(image_exposure[:,:,i] * exposure, 0, 255)

    # Return the exposure image
    return image_exposure

def adjust_temperature(image, temperature):
    # Calculate the temperature
    temperature = temperature / 100

    # Calculate the red and blue channels
    red = temperature * 255
    blue = (1 - temperature) * 255

    # Create the temperature lookup table
    table = np.array([[red, 0, blue] for i in np.arange(0, 256)]).astype("uint8")

    # Apply the temperature to the image
    image_temperature = cv2.LUT(image, table)

    # Return the temperature image
    return image_temperature

def adjust_saturation(image, saturation):
    # Convert the image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Scale the saturation
    hsv[:,:,1] = np.clip(hsv[:,:,1] * saturation, 0, 255)

    # Convert the image back to BGR
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Return the image
    return image

def rotate(image):
    # Rotate the image
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    # Return the rotated image
    return image


def compress(image_file, image, progress, target_size = 49 * 1024 * 1024):
    #encode the image as a png
    encoded = cv2.imencode('.png', image)[1]
    original_size = len(encoded.tobytes())
    progress.write(f"Original size: {humanbytes(len(encoded.tobytes()))} bytes")
    
    #optimize the image with oxipng
    progress.write("Optimizing image with level " + ((str(6) + " (this may take a while)") if len(encoded.tobytes()) > target_size else str(2)) + "...")
    optimized_data = oxipng.optimize_from_memory(encoded.tobytes(), level = 6 if len(encoded.tobytes()) > target_size else 2)

    #use humanbytes() to format the bytes
    progress.write(f"Optimized size: {humanbytes(len(optimized_data))} bytes")
    while len(optimized_data) > target_size + (target_size * 0.01):
        #compute the right amount to reduce the image by in resizing as a percentage between 0 and
        #1, so that the new image is just under the target size
        reduction_factor = (target_size / len(optimized_data)) * 0.99
        progress.write(f"Optimized image is too big, resizing by {100 - (reduction_factor * 100):.2f}%")
        image = cv2.resize(image, (0,0), fx=reduction_factor, fy=reduction_factor, interpolation=cv2.INTER_AREA)
        progress.write(f"Resizing to {image.shape[1]}x{image.shape[0]} pixels")
        #encode the image as a png
        encoded = cv2.imencode('.png', image)[1]
        progress.write(f"Original size: {humanbytes(len(encoded.tobytes()))} bytes")
        #if its now bigger, reize it again
        if len(encoded.tobytes()) > original_size:
            progress.write("Resized image is now bigger, resizing...")
            continue
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


def process_image(image_file, progress, upscaler, args):
    
   # Read the image
    image = cv2.imread(image_file, cv2.IMREAD_COLOR)
    progress.write(f"Working on {image_file}, which is {image.shape[1]}x{image.shape[0]} pixels and {humanbytes(os.path.getsize(image_file))} bytes")
  
    is_black_image = count_bright_pixels(image) < count_dim_pixels(image)
    progress.write(f"Image background is {'black' if is_black_image else 'white'}")
    #get the filenames for the black or white and transparent versions
    if is_black_image:
        color_filename = os.path.dirname(image_file) + "/black/" + "black-" + os.path.basename(image_file).replace(".jpg", ".png")
    else:
        color_filename = os.path.dirname(image_file) + "/white/" + "white-" + os.path.basename(image_file).replace(".jpg", ".png")
    transparent_filename = os.path.dirname(image_file) + "/transparent/" + "transparent-" + os.path.basename(image_file).replace(".jpg", ".png")
    upscaled_filename = os.path.dirname(image_file) + "/upscaled/upscaled-" + os.path.basename(image_file).replace(".jpg", ".png")


    #if the transparent and the color and the upscaled version exist, skip it
    if not args.force and os.path.exists(color_filename) and os.path.exists(transparent_filename) and os.path.exists(upscaled_filename):
        #if the files are smaler than 48 megabytes, skip it
        if os.path.getsize(color_filename) < args.target_size and os.path.getsize(transparent_filename) < args.target_size:
            progress.write(f"Skipping {image_file}, because it already exists")
            return
    if os.path.exists(upscaled_filename):
        up = cv2.imread(upscaled_filename, cv2.IMREAD_UNCHANGED)
        if up.shape[1] == args.target_resolution or up.shape[0] == args.target_resolution:
            progress.write("Found an existing upscaled image, using that")
            image = up
    if not (image.shape[1] == args.target_resolution or image.shape[0] == args.target_resolution):
        progress.write("Upscaling image to " + str(args.target_resolution) + " pixels on the longest side...")
        image = upscaler(image)
        progress.write(f"Resized to {image.shape[1]}x{image.shape[0]} pixels")
        compress(upscaled_filename, image, progress, args.target_size)

    #if resize is set, resize the image
    if args.resize > 0:
        progress.write("Resizing...")
        image = resize(image, args.resize)
        progress.write(f"Resized to {image.shape[1]}x{image.shape[0]} pixels")

    #if flip_horizontal is set, flip the image
    if args.flip_horizontal:
        progress.write("Flipping horizontally...")
        image = cv2.flip(image, 1)
        progress.write(f"Flipped")
    
    #if flip_vertical is set, flip the image
    if args.flip_vertical:
        progress.write("Flipping vertically...")
        image = cv2.flip(image, 0)
        progress.write(f"Flipped")

    #if rotate is set, rotate the image
    if args.rotate > 0:
        progress.write("Rotating...")
        image = rotate(image, args.rotate)
        progress.write(f"Rotated")

    #if crop is set, crop the image
    if args.crop > 0:
        h, w = image.shape[:2]
        progress.write("Cropping...")
        if h < args.crop and w < args.crop:
            image = black_border(image, args.crop)
        else:
            image = center_crop(image, args.crop)
        progress.write(f"Cropped to {image.shape[1]}x{image.shape[0]} pixels")

    #if tempeature is set, apply a temperature filter
    if args.temperature > 0:
        progress.write("Adjusting temperature...")
        image = adjust_temperature(image, args.temperature)
        progress.write(f"Temperature applied")
    
    #if hue is set, apply a hue filter
    if args.hue > 0:
        progress.write("Adjusting hue...")
        image = adjust_hue(image, args.hue)
        progress.write(f"Hue applied")
    
    #if saturation is set, apply a saturation filter
    if args.saturation > 0:
        progress.write("Adjusting saturation...")
        image = adjust_saturation(image, args.saturation)
        progress.write(f"Saturation applied")

    #if vignette is set, apply a vignette
    if args.vignette > 0:
        progress.write("Vignetting...")
        image = vignette(image, args.vignette, is_black_image)
        progress.write(f"Vignette applied")

    #if vignette_sides is set, apply a vignette to the sides
    if args.vignette_sides > 0:
        progress.write("Vignetting sides...")
        image = vignette_sides(image, args.vignette_sides, is_black_image)
        progress.write(f"Vignette applied")

    #if vignette_corners is set, apply a vignette to the corners
    if args.vignette_corners > 0:
        progress.write("Vignetting corners...")
        image = vignette_corners(image, args.vignette_corners, is_black_image)
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
        threshold = find_mean_color(image, is_black_image)
        progress.write(f"The suggested threshold value color is: {threshold}")
    
    #if mask is set, manually mask the edges
    if args.mask > 0:
        progress.write("Manually masking edges...")
        image = mask_edges(image, args.mask, args.blur_mask)
    
    progress.write("Computing transparency mask...")
    mask = compute_transparency_mask(image, threshold, is_black_image)
  

    #if blur is set, blur the mask
    if args.blur > 0:
        progress.write("Blurring mask...")
        mask = blur_mask(mask, args.blur, args.blur)

    if not args.skip_black:
        progress.write(f"Applying {'black' if is_black_image else 'white'} mask...")
        black = apply_mask(image, mask, is_black_image)
        progress.write(f"Saving {'black' if is_black_image else 'white'} image to {color_filename}...")
        # Save the transparent image
        compress(color_filename, black, progress, args.target_size)

    if not args.skip_transparent:
        progress.write("Applying transparency mask...")
        transparent = apply_transparency_mask(image, mask)
        progress.write(f"Saving transparent image to {transparent_filename}...")
        # Save the transparent image
        compress(transparent_filename, transparent, progress, args.target_size)

#a function to rename the files in a directory to the name of the directory plus an incrementing suffix
def rename_files(input_dir):

    new_name = input_dir.split(os.sep)[-1]

    #glob the files, rename them
    counter = 1
    for image_file in glob(os.path.join(input_dir, "*.*"), recursive=False):
        if os.path.basename(image_file).startswith(new_name):
            continue
        extension = os.path.splitext(image_file)[1]
        new_file = input_dir + os.sep + new_name + "-" + str(counter) + extension
        while os.path.exists(new_file):
            counter += 1
            new_file = input_dir + os.sep + new_name + "-" + str(counter) + extension
            print("Renaming " + image_file + " to " + new_file)
        shutil.move(image_file, new_name)


def process_directory(input_dir):
    if not os.path.exists(input_dir):
        return False

    #glob the files, check if their black and transparent counterparts exist. if any don't, return False
    for image_file in glob(os.path.join(input_dir, "*.*"), recursive=False):
        #get the filenames for the black and transparent versions
        black_filename = os.path.dirname(image_file) + "/black/" + "black-" + os.path.basename(image_file).replace(".jpg", ".png")
        transparent_filename = os.path.dirname(image_file) + "/transparent/" + "transparent-" + os.path.basename(image_file).replace(".jpg", ".png")
        white_filename = os.path.dirname(image_file) + "/white/" + "white-" + os.path.basename(image_file).replace(".jpg", ".png")

        if not os.path.exists(black_filename) and not os.path.exists(white_filename):
            return False
        if not os.path.exists(transparent_filename):
            return False
    #move the whole directory to the ready for use dir
    print(f"Moving {input_dir} to G:/Misery Apparel/Ready For Use")
    shutil.move(input_dir, "G:/Misery Apparel/Ready For Use")
    return True

class ContinueOuter(Exception):
    pass   

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
    # Add an optional vignette_sides argument with a default value of 0
    parser.add_argument("--vignette_sides", "-vs", help="the amount of side vignetting to apply", type=float, default=0)
    # Add an optional vignette_corners argument with a default value of 0
    parser.add_argument("--vignette_corners", "-vc", help="the amount of corner vignetting to apply", type=float, default=0)
    #Add an optional contrast argument with a default value of 0
    parser.add_argument("--contrast", "-c", help="the amount of contrast to apply", type=float, default=0)
    #Add an optional brightness argument with a default value of 0
    parser.add_argument("--brightness", "-b", help="the amount of brightness to apply", type=float, default=0)
    #Add an optional mask argument with a default value of 0
    parser.add_argument("--mask", "-m", help="the number of pixels to mask off the edges of the image", type=int, default=0)
    #Add an optional blur mask argument with a default value of 0
    parser.add_argument("--blur_mask", "-bm", help="the amount of blur to apply to the mask", type=int, default=0)
    #Add an optional blur argument with a default value of 0
    parser.add_argument("--blur", "-u", help="the amount of blur to apply to the transparency mask", type=int, default=0)
    #Add an optional sharpen argument with a default value of False
    parser.add_argument("--sharpen", "-s", help="To sharpen or not, default False", type=bool, default=False)
    #Add an optional crop argument with a default value of 0
    parser.add_argument("--crop", "-cr", help="The amount to clip off the edges of the image", type=int, default=0)
    #Add an optional resize argument with a default value of 0
    parser.add_argument("--resize", "-r", help="The amount to resize the image by", type=int, default=0)
    #Add an optional flip_horizontal argument with a default value of False
    parser.add_argument("--flip_horizontal", "-fh", help="To flip horizontally or not, default False", type=bool, default=False)
    #Add an optional flip vertical argument with a default value of False
    parser.add_argument("--flip_vertical", "-fv", help="To flip vertically or not, default False", type=bool, default=False)
    #Add an optional rotate argument with a default value of 0
    parser.add_argument("--rotate", "-ro", help="The amount to rotate the image by", type=int, default=0)
    #Add an optional temperature argument with a default value of 0
    parser.add_argument("--temperature", "-te", help="The amount to adjust the temperature by", type=int, default=0)
    #Add an optional tint argument with a default value of 0
    parser.add_argument("--tint", "-ti", help="The amount to adjust the tint by", type=int, default=0)
    #Add an optional hue argument with a default value of 0
    parser.add_argument("--hue", "-hu", help="The amount to adjust the hue by", type=int, default=0)
    #Add an optional saturation argument with a default value of 0
    parser.add_argument("--saturation", "-sa", help="The amount to adjust the saturation by", type=int, default=0)
    #Add an optional force argument with a default value of False
    parser.add_argument("--force", "-f", help="Force the program to overwrite existing files", type=bool, default=False)
    #Add an optional skip transparent argument with a default value of False
    parser.add_argument("--skip_transparent", "-st", help="Skip the transparent image", type=bool, default=False)
    #Add an optional skip black argument with a default value of False
    parser.add_argument("--skip_black", "-sb", help="Skip the black image", type=bool, default=False)
    #add an optional target_size argument with a default value of 49 * 1024 * 1024
    parser.add_argument("--target_size", "-ts", help="The target size of the image in bytes", type=int, default=49 * 1024 * 1024)
    #add an optional target_resolution argument with a default value of 8192
    parser.add_argument("--target_resolution", "-tr", help="The target resolution of the image in pixels", type=int, default=8192)

    # Parse the command line arguments
    args = parser.parse_args()
    use_path = args.input_file
    if use_path.endswith("/") or use_path.endswith("\\") or use_path.endswith("\""):
        use_path = use_path[:-1]
    
    upscaler = TargetUpscaler(args.target_resolution)
    while True:
        if path.isdir(use_path):
            directories = glob(use_path + os.sep + "*", recursive=False)
        else:
            directories = [args.input_file]
    
        print(f"Processing {len(directories)} directories...")
        for directory in directories:
            rename_files(directory)
            files = glob(directory + os.sep + "*.*", recursive=False)
            to_process = tqdm.tqdm(files)
            
            for image_file in to_process:
                to_process.write(f"Processing {image_file}...")
                print(f"Processing {image_file}...")
                process_image(image_file, to_process, upscaler, args)
            process_directory(directory)
