import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import oxipng


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

def optimize_blur_amount(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    width, height = rect[1]
    blur_amount = int((width + height) / 100)
    return blur_amount if blur_amount % 2 == 1 else blur_amount + 1   

def calculate_reduction_percentage(original_size, compressed_size, target_compressed_size):
    reduction_needed = target_compressed_size - compressed_size
    reduction_percentage = reduction_needed / original_size
    return max(50, 1 + (reduction_percentage / 2))

def compute_optimal_clusters(mask):
    # Subsample the image if specified

    sse = []
    for k in range(1, 25):
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=256*12)
        kmeans.fit_predict(mask)
        sse.append(kmeans.inertia_)
    sse = np.array(sse)
    derivative_1 = np.diff(sse)
    derivative_2 = np.diff(derivative_1)

    # find the elbow by locating the point where the second derivative changes sign
    num_clusters = min(4, np.argmin(derivative_2) + 1)
    return num_clusters

class PrintProgress:
    def write(self, message):
        print(message)

class ImageManipulator:
    def __init__(self, image, progress = PrintProgress()):
        self.image = image
        self.progress = progress

    def resize(self, new_length):
        h, w = self.image.shape[:2]
        if h > w:
            # resize the image by specifying the new height
            self.image = cv2.resize(self.image, (int(w * new_length / h), new_length), interpolation=cv2.INTER_NEAREST)
        else:
            # resize the image by specifying the new width
            self.image = cv2.resize(self.image, (new_length, int(h * new_length / w)), interpolation=cv2.INTER_NEAREST)
        return self
    
    def mask_edges(self, num_pixels, is_black, blur=0):
        height, width, _ = self.image.shape
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
            self.image = np.maximum(cv2.subtract(self.image.astype(np.int16), mask.astype(np.int16)), (0,0,0)).astype(np.uint8)
        else:
            self.image = np.minimum(cv2.add(self.image.astype(np.int16), mask.astype(np.int16)), (255,255,255)).astype(np.uint8)
        return self

    def count_dim_pixels(self, threshold=50):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return np.sum(image < threshold)

    def count_bright_pixels(self, threshold=230):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return np.sum(image > threshold)
    def count_pixels(self):
        return self.image.shape[0] * self.image.shape[1]
    def brightness(self, brightness):
        self.image = cv2.convertScaleAbs(self.image, alpha=brightness)
        return self
    def contrast(self, contrast):
        self.image = cv2.convertScaleAbs(self.image, beta=contrast)
        return self
    def saturation(self, saturation):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.image[:, :, 1] = cv2.convertScaleAbs(self.image[:, :, 1], alpha=saturation)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_HSV2BGR)
        return self
    def hue(self, hue):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.image[:, :, 0] = cv2.convertScaleAbs(self.image[:, :, 0], alpha=hue)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_HSV2BGR)
        return self
    def gamma(self, gamma):
        self.image = np.array(255 * (self.image / 255) ** gamma, dtype='uint8')
        return self
    def sharpen(self):
        self.image = cv2.filter2D(self.image, -1, np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]))
        return self
    def blur(self, blur):
        self.image = cv2.blur(self.image, (blur, blur))
        return self
    def noise(self, noise):
        self.image = cv2.blur(self.image, (noise, noise))
        return self
    def temperature(self, temperature):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.image[:, :, 2] = cv2.convertScaleAbs(self.image[:, :, 2], alpha=temperature)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_HSV2BGR)
        return self
    def tint(self, tint):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.image[:, :, 1] = cv2.convertScaleAbs(self.image[:, :, 1], alpha=tint)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_HSV2BGR)
        return self
    def exposure(self, exposure):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.image[:, :, 0] = cv2.convertScaleAbs(self.image[:, :, 0], alpha=exposure)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_HSV2BGR)
        return self
    def rotate(self, angle):
        self.image = cv2.rotate(self.image, angle)
        return self
    def flip(self, flip):
        self.image = cv2.flip(self.image, flip)
        return self
    def crop(self, x, y, width, height):
        self.image = self.image[y:y+height, x:x+width]
        return self
    def resize(self, width, height):
        self.image = cv2.resize(self.image, (width, height), interpolation=cv2.INTER_LANCZOS4)
        return self
    def resize_to(self, longest_side):
        if self.image.shape[0] > self.image.shape[1]:
            self.image = cv2.resize(self.image, (int(longest_side * self.image.shape[1] / self.image.shape[0]), longest_side), interpolation=cv2.INTER_LANCZOS4)
        elif self.image.shape[0] < self.image.shape[1]:
            self.image = cv2.resize(self.image, (longest_side, int(longest_side * self.image.shape[0] / self.image.shape[1])), interpolation=cv2.INTER_LANCZOS4)
        else:
            self.image = cv2.resize(self.image, (longest_side, longest_side), interpolation=cv2.INTER_LANCZOS4)
        return self
    def grain(self, grain):
        # Add grain to the image
        self.image = self.image + grain * np.random.randn(*self.image.shape)
        return self
    def vignette(self, level = 1, is_black=True): 
        height, width = self.image.shape[:2]
        x_resultant_kernel = cv2.getGaussianKernel(width, width/level)
        y_resultant_kernel = cv2.getGaussianKernel(height, height/level)
        kernel = y_resultant_kernel * x_resultant_kernel.T
        mask = kernel / kernel.max()
        if is_black:
            self.image = self.image * mask[:, :, np.newaxis]
        else:
            self.image = self.image * (1 - mask[:, :, np.newaxis])
        return self
    def vignette_sides(self, dim_factor=0.5, is_black=True):
        height, width = self.image.shape[:2]

        # Create the vignette mask using absolute horizontal distance
        x = np.linspace(-1, 1, width)
        vignette_mask = 1 - np.abs(np.tile(x, (height, 1)))

        # Apply the mask to the image
        image_vignette = np.copy(self.image)
        if is_black:
            vignette_mask = vignette_mask * dim_factor + (1 - dim_factor)
            image_vignette = np.maximum(image_vignette * vignette_mask[:,:,np.newaxis], 0)
        else:
            vignette_mask = vignette_mask * (1 - dim_factor)
            image_vignette = np.minimum(image_vignette * vignette_mask[:,:,np.newaxis], 255)
        self.image = image_vignette
        return self

    def vignette_corners(self, dim_factor=0.5, is_black=True):
        height, width = self.image.shape[:2]

        # Create the vignette mask using absolute horizontal and vertical distance
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        xx, yy = np.meshgrid(x, y)
        vignette_mask = 1 - np.sqrt(xx**2 + yy**2)

        # Apply the mask to the image
        image_vignette = np.copy(self.image)
        if is_black:
            vignette_mask = vignette_mask * dim_factor + (1 - dim_factor)
            image_vignette = np.maximum(image_vignette * vignette_mask[:,:,np.newaxis], 0)
        else:
            vignette_mask = vignette_mask * (1 - dim_factor)
            image_vignette = np.minimum(image_vignette * vignette_mask[:,:,np.newaxis], 255)
        self.image = image_vignette
        return self

    def mask(self, mask, is_black=True):
        # Apply the mask to the image
        masked_image = np.copy(self.image)
        #invert the mask if we want to apply it to the background
        mask = cv2.bitwise_not(mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if is_black:
            #subtract the mask from the image
            masked_image = cv2.subtract(self.image, mask)
        else:
            masked_image = cv2.add(self.image, mask)
        self.image = masked_image
        return self

    def get_gray_mask(self, threshold, tolerance=0.05, is_black=True):
        grayscaled = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        mask = np.ones(self.image.shape[:2], np.uint8) * 255
        condition = ((np.abs(self.image[..., 0] - grayscaled) < grayscaled * tolerance) &
                    (np.abs(self.image[..., 1] - grayscaled) < grayscaled * tolerance) &
                    (np.abs(self.image[..., 2] - grayscaled) < grayscaled * tolerance))
        condition &= grayscaled < threshold if is_black else grayscaled >= threshold
        #set the mask to grayscaled inverse value only where the condition is true
        mask[condition] = 0
        return mask

    def mask_transparent(self, mask):
        alpha = cv2.cvtColor(self.image, cv2.COLOR_BGR2BGRA)
        alpha[:,:,3] = mask
        self.image = alpha
        return self
    
    def get_threshold_mask(self, threshold, is_black=True):
        # Convert the image to grayscale
        grayscaled = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        mask = np.ones(self.image.shape[:2], np.uint8) * 255
        if is_black:
            mask[np.where(grayscaled < threshold)] = 0
        else:
            mask[np.where(grayscaled >= threshold)] = 0
        return mask
    
    def brightness_contrast(self, brightness, contrast):
        # Calculate the actual brightness and contrast values
        alpha = contrast * 2 + 1
        beta = brightness * 255

        # Scale and shift the pixel values of the image
        adjusted_image = cv2.convertScaleAbs(self.image, alpha=alpha, beta=beta)

        # Clamp the pixel values to the valid range
        adjusted_image = np.clip(adjusted_image, 0, 255)

        # Return the adjusted image
        self.image = adjusted_image
        return self   



    def find_mean_color(self, is_black, num_clusters=-1, subsample=.25):
        # Subsample the image if specified
        if subsample < 1:
            image = cv2.resize(self.image, (0, 0), fx=subsample, fy=subsample, interpolation=cv2.INTER_LANCZOS4)
        else:
            image = self.image
        # Convert the image to a single-channel, floating point representation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray= gray.reshape(-1,1)

        if num_clusters < 2:
            num_clusters = self.compute_num_clusters(gray, is_black)
        self.progress.write(f"Found {num_clusters} clusters")

        kmeans = MiniBatchKMeans(n_clusters=num_clusters, max_iter=1000, batch_size=256*12, n_init=250)
        labels = kmeans.fit_predict(gray)
        clustered_image = np.zeros((self.image.shape[0], self.image.shape[1], 3), dtype=np.uint8)
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
        clustered_image = np.array(colors)[flat_labels].reshape(self.image.shape[0], self.image.shape[1], 3)
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
        self.progress.write(f"Found {len(pixel_indices[0])} pixels in dominant cluster which is {len(pixel_indices[0]) / len(labels) * 100:.2f}% of the image")
        dominant_pixels = gray[pixel_indices].astype(np.float)
        mean_color = dominant_pixels.max().astype(np.uint8)
        secondary_dominant_pixels = gray[np.where(labels == second_dominant_cluster)].astype(np.float)
        secondary_mean_color = secondary_dominant_pixels.max().astype(np.uint8)
        return mean_color, secondary_mean_color



    def center_crop(self, size):
        h, w = self.image.shape[:2]
        y = h//2-(size//2)
        x = w//2-(size//2)
        self.image = self.image[y:y+size, x:x+size]
        return self
        
    def compress(self, target_path=None, target_size=49 * 1024 * 1024, max_iterations=7):
        encoded = cv2.imencode('.png', self.image)[1]
        original_size = len(encoded.tobytes())
        self.progress.write(f"Original size: {humanbytes(original_size)} bytes")
        optimization_level = 2 if original_size <= target_size else 4
        self.progress.write(f"Optimizing image with level {optimization_level} (this may take a while)...")
        optimized_data = oxipng.optimize_from_memory(encoded.tobytes(), level=optimization_level)
        optimized_size = len(optimized_data)
        self.progress.write(f"Optimized size: {humanbytes(optimized_size)} bytes")
        while optimized_size > target_size and max_iterations > 0:
            reduction_factor = calculate_reduction_percentage(original_size, optimized_size, target_size)
            self.progress.write(f"Reduction factor: {reduction_factor:.2f}")
            self.progress.write(f"Optimized image is too big, resizing by {100 - (reduction_factor * 100):.2f}%")
            self.image = cv2.resize(self.image, (0, 0), fx=reduction_factor, fy=reduction_factor, interpolation=cv2.INTER_AREA)
            self.progress.write(f"Resizing to {self.image.shape[1]}x{self.image.shape[0]} pixels")
            encoded = cv2.imencode('.png', self.image)[1]
            original_size = len(encoded.tobytes())
            self.progress.write(f"Original size: {humanbytes(original_size)} bytes")
            optimized_data = oxipng.optimize_from_memory(encoded.tobytes(), level=4)
            optimized_size = len(optimized_data)
            self.progress.write(f"Optimized size: {humanbytes(optimized_size)} bytes")
            max_iterations -= 1
        self.progress.write(f"Final size: {humanbytes(optimized_size)} bytes")
        if target_path:
            with open(target_path, 'wb') as f:
                f.write(optimized_data)
        return optimized_data


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

