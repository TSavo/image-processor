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
from rembg import remove


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

def find_mean_color(img, is_black, progress, num_clusters=4, subsample=.25):
    # Subsample the image if specified
    if subsample < 1:
        img = cv2.resize(img, (0, 0), fx=subsample, fy=subsample, interpolation=cv2.INTER_LANCZOS4)
    # Convert the image to a single-channel, floating point representation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray= gray.reshape(-1,1)

    sse = []
    for k in range(1, 25):
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=256*12)
        kmeans.fit_predict(gray)
        sse.append(kmeans.inertia_)
        progress.write(f"{k} clusters has a squared error of {kmeans.inertia_}")
    sse = np.array(sse)
    derivative_1 = np.diff(sse)
    derivative_2 = np.diff(derivative_1)

    # find the elbow by locating the point where the second derivative changes sign
    num_clusters = min(5, np.argmin(derivative_2) + 1)
    progress.write(f"Found {num_clusters} clusters")

    kmeans = MiniBatchKMeans(n_clusters=num_clusters, max_iter=1000, batch_size=256*12, n_init=250)
    labels = kmeans.fit_predict(gray)
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

    flat_labels = labels.flatten()
    clustered_image = np.array(colors)[flat_labels].reshape(img.shape[0], img.shape[1], 3)
    cv2.imwrite("clustered.png", clustered_image)

    # Determine which is the second most dominant cluster
    sorted_clusters = np.argsort(kmeans.cluster_centers_.mean(axis=1))
    if is_black:
        dominant_cluster = sorted_clusters[0]
        second_dominant_cluster = sorted_clusters[1]
    else:
        dominant_cluster = sorted_clusters[-1]
        second_dominant_cluster = sorted_clusters[-2]

    # Find the mean color of the dominant cluster and the secondary dominant cluster
    pixel_indices = np.where(labels == dominant_cluster)
    progress.write(f"Found {len(pixel_indices[0])} pixels in dominant cluster which is {len(pixel_indices[0]) / len(labels) * 100:.2f}% of the image")
    dominant_pixels = gray[pixel_indices].astype(np.float)
    mean_color = dominant_pixels.max().astype(np.uint8)
    secondary_dominant_pixels = gray[np.where(labels == second_dominant_cluster)].astype(np.float)
    secondary_mean_color = secondary_dominant_pixels.max().astype(np.uint8)
    progress.write(f"Mean color of dominant cluster is {mean_color} and secondary dominant cluster is {secondary_mean_color}")
    return mean_color, secondary_mean_color

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

def compute_wipe_mask(image, threshold, is_black=True):
    # Convert the image to grayscale
    grayscaled = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    mask = np.ones(image.shape[:2], np.uint8) * 255
    if is_black:
        mask[np.where(grayscaled < threshold)] = 0
    else:
        mask[np.where(grayscaled >= threshold)] = 0
    return mask

def compute_gray_mask(image, threshold, tolerance=0.05, is_black=True):
    grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.ones(image.shape[:2], np.uint8) * 255
    condition = ((np.abs(image[..., 0] - grayscaled) < grayscaled * tolerance) &
                 (np.abs(image[..., 1] - grayscaled) < grayscaled * tolerance) &
                 (np.abs(image[..., 2] - grayscaled) < grayscaled * tolerance))
    condition &= grayscaled < threshold if is_black else grayscaled >= threshold
    #set the mask to grayscaled inverse value only where the condition is true
    mask[condition] = 0
    return mask

def optimize_blur_amount(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    width, height = rect[1]
    blur_amount = int((width + height) / 100)
    return blur_amount if blur_amount % 2 == 1 else blur_amount + 1


def blur_mask(mask, blur_amount=None, progress_bar=None):
    if blur_amount is None:
        blur_amount = optimize_blur_amount(mask)
    if progress_bar is not None:
        progress_bar.write("Blur size is {} pixels".format(blur_amount))
    # Apply a Gaussian blur to the binary mask
    blurred_mask = cv2.GaussianBlur(mask.astype(np.float32), (blur_amount, blur_amount), 0)
    
    # Convert the blurred mask back to binary format
    blurred_mask = np.round(blurred_mask)
    blurred_mask = blurred_mask.astype(np.uint8)
    
    return blurred_mask

def apply_transparency_mask(image, mask):
    alpha = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    alpha[:,:,3] = mask
    return alpha

def apply_mask(image, mask, is_black=True):
    # Apply the mask to the image
    masked_image = np.copy(image)
    #invert the mask if we want to apply it to the background
    mask = cv2.bitwise_not(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if is_black:
        #subtract the mask from the image
        masked_image = cv2.subtract(image, mask)
    else:
        masked_image = cv2.add(image, mask)
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
    image_contrast = np.clip((image - mean) * contrast + mean, 0, 255)

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

def calculate_reduction_percentage(original_size, compressed_size, target_compressed_size):
    reduction_needed = target_compressed_size - compressed_size
    reduction_percentage = reduction_needed / original_size
    return 1 + (reduction_percentage / 2)

def compress(image_file, image, progress, target_size=49 * 1024 * 1024, compression_level=4):
    # Encode the image as a png
    encoded = cv2.imencode('.png', image)[1]
    original_size = len(encoded.tobytes())
    progress.write(f"Original size: {humanbytes(original_size)} bytes")

    # Optimize the image with oxipng
    optimization_level = 4
    progress.write(f"Optimizing image with level {optimization_level} (this may take a while)...")
    optimized_data = oxipng.optimize_from_memory(encoded.tobytes(), level=optimization_level)
    optimized_size = len(optimized_data)
    progress.write(f"Optimized size: {humanbytes(optimized_size)} bytes")

    while optimized_size > target_size and max_iterations > 0:
        reduction_factor = calculate_reduction_percentage(original_size, optimized_size, target_size)
        progress.write(f"Reduction factor: {reduction_factor:.2f}")
        progress.write(f"Optimized image is too big, resizing by {100 - (reduction_factor * 100):.2f}%")
        image = cv2.resize(image, (0, 0), fx=reduction_factor, fy=reduction_factor, interpolation=cv2.INTER_AREA)
        progress.write(f"Resizing to {image.shape[1]}x{image.shape[0]} pixels")
        encoded = cv2.imencode('.png', image)[1]
        original_size = len(encoded.tobytes())
        progress.write(f"Original size: {humanbytes(original_size)} bytes")
        optimized_data = oxipng.optimize_from_memory(encoded.tobytes(), level=4)
        optimized_size = len(optimized_data)
        progress.write(f"Optimized size: {humanbytes(optimized_size)} bytes")
        max_iterations -= 1

    
    # Create the directory if necessary
    os.makedirs(os.path.dirname(image_file), exist_ok=True)

    # Save the image
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
    base_dir = os.path.dirname(image_file)
    base_name = os.path.basename(image_file).replace(".jpg", ".png")

    if is_black_image:
        color_dir = "black"
    else:
        color_dir = "white"

    color_filename = os.path.join(base_dir, color_dir, f"{color_dir}-{base_name}")
    transparent_filename = os.path.join(base_dir, "transparent", f"transparent-{base_name}")
    upscaled_filename = os.path.join(base_dir, "upscaled", f"upscaled-{base_name}")


    #if the transparent and the color and the upscaled version exist, skip it
    if not args.force and os.path.exists(color_filename) and os.path.exists(transparent_filename):
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
    #threshold = args.threshold
    #if args.threshold == -1:
        #progress.write("Computing background color threshold...")
        #threshold = find_mean_color(image, is_black_image)
        #progress.write(f"The suggested threshold value color is: {threshold}")
    
    #if mask is set, manually mask the edges
    if args.mask > 0:
        progress.write("Manually masking edges...")
        image = mask_edges(image, args.mask, args.blur_mask)
    
    progress.write("Computing background mask...")
    #mask = compute_transparency_mask(image, threshold, is_black_image)
 
    mask = remove(image, only_mask=True)
    cv2.imwrite("mask.png", mask)

    
    #mask = blur_mask(mask,4,4)
    #cv2.imwrite("mask_blured.png", mask)
    #black = apply_mask(image, mask, is_black_image)


    threshold, secondary_threshold = find_mean_color(image, is_black_image, progress, num_clusters=7)
    progress.write(f"The suggested threshold values are: {threshold} and {secondary_threshold}")

    wipe_mask = compute_wipe_mask(image, threshold, is_black_image)
    #mask = np.bitwise_and(mask, wipe_mask)
    mask = np.bitwise_or(mask, wipe_mask)


    cv2.imwrite("mask_wipe.png", mask)
    progress.write("Computing gray mask...")
    gray_mask = compute_gray_mask(image, secondary_threshold, 0.25, is_black_image)
    cv2.imwrite("mask_gray.png", gray_mask)
    progress.write("Computing final mask...")
    mask = np.bitwise_and(mask, gray_mask)
    progress.write("Blurring final mask...")
    mask = blur_mask(mask, None, progress)
    cv2.imwrite("mask_final.png", mask)
    #threshold the msdk to bring it back to black and white
    #mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)[1]
    #cv2.imwrite("mask_final_threshold.png", mask)
    #mask = blur_mask(mask, np.mean(mask.shape)/100, np.mean(mask.shape)/100)
    #cv2.imwrite("mask_final_threshold_blured.png", mask)

    #if blur is set, blur the mask
    if args.blur is not None:
        progress.write("Blurring mask...")
        mask = blur_mask(mask, args.blur, progress)

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
        #if the file contains the name in it, skip it
        if os.path.basename(image_file).index(new_name) != -1:
            continue
        extension = os.path.splitext(image_file)[1]
        new_file = input_dir + os.sep + new_name + "-" + str(counter) + extension
        while os.path.exists(new_file):
            counter += 1
            new_file = input_dir + os.sep + new_name + "-" + str(counter) + extension
            print("Renaming " + image_file + " to " + new_file)
        shutil.move(image_file, new_name)


def finish(dir, output_file):
    to_process.write(f"Finished {dir}...")
    basedir = os.path.join(output_file, os.path.basename(dir))

    if not os.path.isdir(basedir):
        os.mkdir(basedir)
    if os.path.exists(os.path.join(dir, "originals")):
        if not os.path.isdir(os.path.join(basedir, "originals")):
            os.mkdir(os.path.join(basedir, "originals"))
        merge_directories(os.path.join(dir, "originals"), os.path.join(basedir, "originals"))
    if os.path.exists(os.path.join(dir, "black")):
        if not os.path.isdir(os.path.join(basedir, "black")):
            os.mkdir(os.path.join(basedir, "black"))
        merge_directories(os.path.join(dir, "black"), os.path.join(basedir, "black"))
    if os.path.exists(os.path.join(dir, "white")):
        if not os.path.isdir(os.path.join(basedir, "white")):
            os.mkdir(os.path.join(basedir, "white"))
        merge_directories(os.path.join(dir, "white"), os.path.join(basedir, "white"))
    if os.path.exists(os.path.join(dir, "transparent")):
        if not os.path.isdir(os.path.join(basedir, "transparent")):
            os.mkdir(os.path.join(basedir, "transparent"))
        merge_directories(os.path.join(dir, "transparent"), os.path.join(basedir, "transparent"))
    if os.path.exists(os.path.join(dir, "upscaled")):
        if not os.path.isdir(os.path.join(basedir, "upscaled")):
            os.mkdir(os.path.join(basedir, "upscaled"))
        merge_directories(os.path.join(dir, "upscaled"), os.path.join(basedir, "upscaled"))
    try:
        shutil.rmtree(os.path.join(dir, "transparent"))
        shutil.rmtree(os.path.join(dir, "black"))
        shutil.rmtree(os.path.join(dir, "white"))
        shutil.rmtree(os.path.join(dir, "originals"))
        shutil.rmtree(os.path.join(dir, "upscaled"))
    except:
        pass

def merge_directories(input_dir, output_dir):
    #glob the files, check if their black and transparent counterparts exist. if any don't, return False
    for image_file in glob(os.path.join(input_dir, "*"), recursive=False):
        if os.path.isdir(image_file):
            if not os.path.exists(os.path.join(output_dir, os.path.basename(image_file))):
                os.mkdir(os.path.join(output_dir, os.path.basename(image_file)))
            merge_directories(image_file, os.path.join(output_dir, os.path.basename(image_file)))
        else:
            if not os.path.exists(os.path.join(output_dir)):
                os.mkdir(os.path.join(output_dir))
            shutil.move(image_file, os.path.join(output_dir, os.path.basename(image_file)))


def process_directory(input_dir):
    if not os.path.exists(input_dir):
        return False

    #glob the files, check if their black and transparent counterparts exist. if any don't, return False
    for image_file in glob(os.path.join(input_dir, "*.*"), recursive=False):
        #get the filenames for the black and transparent versions
        base_dir = os.path.dirname(image_file)
        base_name = os.path.basename(image_file).replace(".jpg", ".png")

        black_filename = os.path.join(base_dir, "black", f"black-{base_name}")
        transparent_filename = os.path.join(base_dir, "transparent", f"transparent-{base_name}")
        white_filename = os.path.join(base_dir, "white", f"white-{base_name}")

        if not os.path.exists(black_filename) and not os.path.exists(white_filename):
            return False
        if not os.path.exists(transparent_filename):
            return False
    #move the whole directory to the ready for use dir
    print(f"Moving {input_dir} to G:/Misery Apparel/Ready For Use")
    #os.rename(input_dir, "G:/Misery Apparel/Ready For Use/" + os.path.basename(input_dir))
    if len(os.listdir(input_dir)) > 0:
        return False
    
    shutil.rmtree(input_dir)
    return True


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
    parser.add_argument("--blur", "-u", help="the amount of blur to apply to the transparency mask", type=int, default=None)
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
    use_path = path.abspath(args.input_file)
    upscaler = TargetUpscaler(args.target_resolution)
    while True:
        if path.isdir(use_path):
            directories = glob(use_path + os.sep + "*", recursive=False)
        else:
            directories = [args.input_file]
    
        print(f"Processing {len(directories)} directories...")
        import random
        #build a list bof all the files in the directories
        todo = []
        for directory in directories:
            #rename_files(directory)
            files = glob(directory + os.sep + "*.*", recursive=False)
            for file in files:
                todo.append(os.path.join(directory, file))


        to_process = tqdm.tqdm(random.sample(todo, 5))
        
        for image_file in to_process:
            directory = os.path.dirname(image_file)
            try:
                process_image(image_file, to_process, upscaler, args)
                #move the original into the originals directory
                if not path.isdir(directory + os.sep + "originals"):
                    os.mkdir(directory + os.sep + "originals")
                os.rename(image_file, os.path.join(directory, "originals", os.path.basename(image_file)))
                finish(directory, args.output_file)
                process_directory(directory)
            except Exception as e:
                print(e)
                print(f"Error processing {image_file}")
                continue