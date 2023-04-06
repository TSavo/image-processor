import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import oxipng
from functools import partial
import inspect
import rembg


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

def unwrap_image(func):
    def unwrap(*args, **kwargs):
        if not isinstance(args[0], tuple):
            return func(*args, **kwargs)

        image, metadata = args[0]
        args = args[1:]

        mapped_kwargs = {item.name: metadata.get(item.name, kwargs[item.name]) 
                        for item in inspect.signature(func).parameters.values() 
                        if item.kind == inspect.Parameter.KEYWORD_ONLY and item.name in kwargs}

        if kwargs.get("image_key"):
            image = metadata[kwargs["image_key"]]

        result = func(image, *args, **mapped_kwargs)

        if isinstance(result, tuple):
            result, new_metadata = result
            metadata.update(new_metadata)

        if kwargs.get("save_key"):
            metadata[kwargs["save_key"]] = result

        return image, metadata if isinstance(result, tuple) else result, metadata
    return unwrap
class ImageEditor:

    @unwrap_image
    def load_image_from_file(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        return image, {"filename": image_path, "file_size": os.path.getsize(image_path), "original_dimensions": image.shape}
    
    @unwrap_image
    def load_image_from_bytes(self, image_bytes):
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        return image, {"file_size": len(image_bytes)}
    


    @unwrap_image
    def encode(self, image, image_format="png", quality=100):
        if image_format == "png":
            return cv2.imecode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, quality])
        elif image_format == "jpg":
            return cv2.imecode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif image_format == "webp":
            return cv2.imecode(".webp", image, [cv2.IMWRITE_WEBP_QUALITY, quality])
        elif image_format == "tiff":
            return cv2.imecode(".tiff", image, [cv2.IMWRITE_TIFF_COMPRESSION, quality])
        elif image_format == "jpeg2000":
            return cv2.imecode(".jp2", image, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, quality])
        else:
            return cv2.imencode(f".{image_format}", image)

    def write(self, image, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            f.write(image.tobytes())

    @unwrap_image
    def save_as(self, image, filename):
        cv2.imwrite(filename, image)
        return image, {"file_size": os.path.getsize(filename)}

    def update_metadata(self, t, update_func):
        image, metadata = t
        kws = {item.name: metadata.get(item.name, None) for item in inspect.signature(update_func).parameters.values() if item.kind == inspect.Parameter.KEYWORD_ONLY}
        for k, v in kws.items():
            metadata[k] = update_func(**{k: v})
        return image, metadata


    @unwrap_image
    def resize(self, image, new_length, interpolation=cv2.INTER_NEAREST):
        h, w = image.shape[:2]
        if h > w:
            # resize the image by specifying the new height
            image = cv2.resize(image, (int(w * new_length / h), new_length), interpolation=interpolation)
        else:
            # resize the image by specifying the new width
            image = cv2.resize(image, (new_length, int(h * new_length / w)), interpolation=interpolation)
        return image
    @unwrap_image
    def set_blur(self, image, blur_amount):
        return image, {"blur": blur_amount}
    @unwrap_image
    def set_blur_multiplier(self, image, blur_multiplier):
        return image, {"blur_multiplier": blur_multiplier}
    @unwrap_image
    def mask_edges(self, image, num_pixels=10, bias=0, blur=0, blur_multiplier=1):
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
            mask = cv2.GaussianBlur(mask, (blur,blur), blur * blur_multiplier)
        #invert the mask
        mask = cv2.bitwise_not(mask)
        # Convert the mask to 3 channels
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # Apply the mask to the image
        if bias == 0:
            image = np.maximum(cv2.subtract(image.astype(np.int16), mask.astype(np.int16)), (0,0,0)).astype(np.uint8)
        else:
            image = np.minimum(cv2.add(image.astype(np.int16), mask.astype(np.int16)), (255,255,255)).astype(np.uint8)
        return image
    @unwrap_image
    def count_dim_pixels(self, image, threshold=50):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.sum(image < threshold)
    @unwrap_image
    def count_bright_pixels(self, image, threshold=230):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.sum(image > threshold)
    @unwrap_image
    def count_pixels(self, image):
        return image.shape[0] * image.shape[1]
    @unwrap_image
    def brightness(self, image, brightness):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = np.clip(v + brightness, 0, 255)
        final_hsv = cv2.merge((h, s, v))
        image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return image
    @unwrap_image
    def contrast(self, image, contrast):
        alpha = 1.0 + contrast / 127.0
        beta = -contrast
        image = cv2.addWeighted(image, alpha, image, 0, beta)
        return image
    @unwrap_image
    def saturation(self, image, saturation):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(s + saturation, 0, 255)
        final_hsv = cv2.merge((h, s, v))
        image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return image
    @unwrap_image
    def hue(self, image, hue):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        h = np.clip(h + hue, 0, 255)
        final_hsv = cv2.merge((h, s, v))
        image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return image
    @unwrap_image
    def sharpen(self, image, sharpen):
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        image = cv2.filter2D(image, -1, kernel)
        return image
    @unwrap_image
    def blur(self, image, blur):
        image = cv2.blur(image, (blur, blur))
        return image
    @unwrap_image
    def gaussian_blur(self, image, gaussian_blur):
        image = cv2.GaussianBlur(image, (gaussian_blur, gaussian_blur), 0)
        return image
    @unwrap_image
    def median_blur(self, image, median_blur):
        image = cv2.medianBlur(image, median_blur)
        return image
    @unwrap_image
    def bilateral_blur(self, image, bilateral_blur):
        image = cv2.bilateralFilter(image, bilateral_blur, bilateral_blur * 2, bilateral_blur / 2)
        return image
    @unwrap_image
    def rotate(self, image, angle):
        image = cv2.rotate(image, angle)
        return image
    @unwrap_image
    def flip(self, image, flip):
        image = cv2.flip(image, flip)
        return image
    @unwrap_image
    def crop(self, image, x, y, w, h):
        image = image[y:y+h, x:x+w]
        return image
    @unwrap_image
    def resize_to(self, image, width, height):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
        return image
    @unwrap_image
    def salt_and_pepper(self, image, amount):
        image = image.copy()
        num_salt = np.ceil(amount * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        image[coords] = 1
        num_pepper = np.ceil(amount * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        image[coords] = 0
        return image
    @unwrap_image
    def temperature(self, image, temperature):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image[:, :, 2] = cv2.convertScaleAbs(image[:, :, 2], alpha=temperature)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image
    @unwrap_image
    def tint(self, image, tint):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image[:, :, 1] = cv2.convertScaleAbs(image[:, :, 1], alpha=tint)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image
    @unwrap_image
    def gamma(self, image, gamma):
        image = np.power(image / float(np.max(image)), gamma)
        return image
    @unwrap_image
    def equalize(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
        return image
    @unwrap_image
    def grayscale(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    @unwrap_image
    def sepia(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.transform(image, np.matrix([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]]))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
    @unwrap_image
    def invert(self, image):
        image = cv2.bitwise_not(image)
        return image
    @unwrap_image
    def solarize(self, image, threshold):
        image = np.where(image < threshold, image, 255 - image)
        return image
    @unwrap_image
    def solarize_add(self, image, threshold, addition=0):
        image = np.where(image < threshold, image + addition, image)
        return image
    @unwrap_image
    def posterize(self, image, bits):
        shift = 8 - bits
        image = np.right_shift(np.left_shift(image, shift), shift)
        return image
    @unwrap_image
    def tint(self, image, tint):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image[:, :, 1] = cv2.convertScaleAbs(image[:, :, 1], alpha=tint)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image
    @unwrap_image
    def grain(self, image, grain):
        image = image + grain * np.random.randn(*image.shape)
        return image
    @unwrap_image
    def vignette(self, image, level = 1, bias=0):
        height, width = image.shape[:2]
        x_resultant_kernel = cv2.getGaussianKernel(width, width/level)
        y_resultant_kernel = cv2.getGaussianKernel(height, height/level)
        kernel = y_resultant_kernel * x_resultant_kernel
        kernel = kernel / np.max(kernel)
        if bias == 0:
            kernel = 1 - kernel
        image = cv2.filter2D(image, -1, kernel)
        return image
    @unwrap_image
    def vignette_sides(self, image, dim_factor=0.5, bias=0):
        height, width = image.shape[:2]
        x = np.linspace(-1, 1, width)
        vignette_mask = 1 - np.abs(np.tile(x, (height, 1)))
        image_vignette = np.copy(image)
        if bias == 0:
            vignette_mask = vignette_mask * dim_factor + (1 - dim_factor)
        image_vignette = image_vignette * vignette_mask
        return image_vignette
    @unwrap_image
    def vignette_corners(self, image, dim_factor=0.5, bias=0):
        height, width = image.shape[:2]
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        x_mask, y_mask = np.meshgrid(x, y)
        mask = np.sqrt(x_mask**2 + y_mask**2)
        mask = np.clip(mask, 0, 1) * dim_factor + (1 - dim_factor)
        if bias == 0:
            mask = 1 - mask
        image = image * mask
        return image
    @unwrap_image
    def vignette_circle(self, image, dim_factor=0.5, bias=0):
        height, width, channels = image.shape
        center = (int(width/2), int(height/2))
        radius = int(np.sqrt((height/2)**2 + (width/2)**2))
        # Create a meshgrid of the image dimensions
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        # Calculate the distance from each pixel to the center of the circle
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        # Normalize the distance so that it is between 0 and 1
        norm_dist = dist / radius
        # Calculate the dimness as a function of the distance from the center
        dimness = 1 - norm_dist
        # Create a circular mask based on the dimness
        mask = (dimness * 255).clip(0, 255).astype(np.uint8)
        mask = mask * dim_factor + (1 - dim_factor)
        if bias == 0:
            mask = 255 - mask
        image = image * mask
        return image
    @unwrap_image
    def vignette_at(self, image, x, y, radius, dim_factor=0.5, bias=0):
        height, width = image.shape[:2]
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        x_mask, y_mask = np.meshgrid(x, y)
        mask = np.sqrt((x_mask - x)**2 + (y_mask - y)**2)
        mask = np.clip(mask, 0, radius)
        mask = mask * dim_factor + (1 - dim_factor)
        if bias == 0:
            mask = 1 - mask
        image = image * mask
        return image
    @unwrap_image
    def vignette_at_multiple(self, image, points, radius, dim_factor=0.5, bias=0):
        for point in points:
            image = self.vignette_at(image, point[0], point[1], radius, dim_factor, bias)
        return image
    @unwrap_image
    def vignette_at_multiple_random(self, image, num_random, radius, dim_factor=0.5, bias=0):
        return self.vignette_at_multiple(image, self.random_points(image, num_random), radius, dim_factor, bias)
    @unwrap_image
    def random_points(self, image, num_random=5):
        height, width = image.shape[:2]
        points = []
        for _ in range(num_random):
            x = np.random.uniform(0, height)
            y = np.random.uniform(0, width)
            points.append((x, y))
        return points
    @unwrap_image
    def grayscale(self, image):
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    @unwrap_image
    def dim_pixels(self, image, lower_threshold=50):
        image, metadata = self.grayscale(image)
        metadata['dim_pixels'] = np.sum(image < lower_threshold)
        return image, metadata
    @unwrap_image
    def bright_pixels(self, image, upper_threshold=255-50):
        image, metadata = self.grayscale(image)
        metadata['bright_pixels'] = np.sum(image > upper_threshold)
        return image, metadata
    @unwrap_image
    def bias(self, image, lower_threshold=50, upper_threshold=255-50):
        image = self.dim_pixels(image, lower_threshold=lower_threshold)
        image = self.bright_pixels(image, upper_threshold=upper_threshold)
        image, metadata = image
        metadata['bias'] = 0 if metadata['dim_pixels'] > metadata['bright_pixels'] else 255
        return image, metadata
    @unwrap_image
    def mask(self, image, mask=None, bias=None):
        if mask is None:
            raise ValueError('No mask found in metadata')
        if bias is None:
            image, metadata = self.bias(image)
            bias = metadata['bias']
        # Apply the mask to the image
        masked_image = np.copy(image)
        #invert the mask if we want to apply it to the background
        if mask.shape[2] == 1:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        transparency = None
        #if the image has transparency, we need to remove it
        if image.shape[2] == 4:
            transparency = image[:, :, 3]
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if bias == 0:
            #subtract the mask from the image
            masked_image = cv2.subtract(image, mask)
        else:
            masked_image = cv2.add(image, mask)
        #if the image had transparency, we need to add it back
        if transparency is not None:
            masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2BGRA)
            masked_image[:, :, 3] = transparency
        return masked_image, metadata
    @unwrap_image
    def mask_transparent(self, image, mask=None):
        if mask is None:
            raise ValueError('No mask found in metadata')
        if image.shape[2] != 4:
            image = self.convert_to_transparent(image)
        image[:, :, 3] = mask
        return image
    @unwrap_image
    def convert_to_transparent(self, image):
        alpha = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        return alpha
    @unwrap_image
    def and_together(self, image1, image2):
        if isinstance(image2, tuple):
            image2 = image2[0]
        return cv2.bitwise_and(image1, image2)
    @unwrap_image
    def or_together(self, image1, image2):
        if isinstance(image2, tuple):
            image2 = image2[0]
        return cv2.bitwise_or(image1, image2)
    @unwrap_image
    def xor_together(self, image1, image2):
        if isinstance(image2, tuple):
            image2 = image2[0]
        return cv2.bitwise_xor(image1, image2)
    @unwrap_image
    def not_together(self, image1, image2):
        if isinstance(image2, tuple):
            image2 = image2[0]
        return cv2.bitwise_not(image1, image2)
    @unwrap_image
    def add(self, image1, image2):
        if isinstance(image2, tuple):
            image2 = image2[0]
        return cv2.add(image1, image2)
    @unwrap_image
    def subtract(self, image1, image2):
        if isinstance(image2, tuple):
            image2 = image2[0]
        return cv2.subtract(image1, image2)
    @unwrap_image
    def multiply(self, image1, image2):
        if isinstance(image2, tuple):
            image2 = image2[0]
        return cv2.multiply(image1, image2)
    @unwrap_image
    def divide(self, image1, image2):
        if isinstance(image2, tuple):
            image2 = image2[0]
        return cv2.divide(image1, image2)
    @unwrap_image
    def get_threshold_mask(self, image):
        image, metadata = image
        threshold = metadata['threshold']
        if threshold is None:
            raise ValueError('No threshold found in metadata')
        bias = metadata['bias']
        if bias is None:
            bias = 0
        # Convert the image to grayscale
        grayscaled = self.grayscale(image)
        mask = np.ones(image.shape[:2], np.uint8) * 255
        if bias == 0:
            mask[np.where(grayscaled < threshold)] = 0
        else:
            mask[np.where(grayscaled >= threshold)] = 0
        return mask
    #find the background color using a histogram to find the black or white spike
    @unwrap_image
    def find_background_color(self, image, bias=0):
        # Convert the image to grayscale
        grayscaled = self.grayscale(image)
        # Find the histogram of the image
        hist = cv2.calcHist([grayscaled], [0], None, [256], [0, 256])
        # Find the lower and upper peaks of the histogram
        lower_peak = np.argmin(hist)
        upper_peak = np.argmax(hist)
        # Find the background color
        if bias == 0:
            return lower_peak
        else:
            return upper_peak

    @unwrap_image
    def subsample(self, image, subsample=.25):
        return image, {'subsample': subsample}
    @unwrap_image
    def mask_background_canny(self, image):
        # use cv2 to find the contours
        edges = cv2.Canny(image, 100, 200)
        # find the largest contour
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        # create a mask of the largest contour
        mask = np.zeros(image.shape, np.uint8)
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        return image, {'canny_mask': mask}
    @unwrap_image
    def mask_background_rembg(self, image, name='rembg_mask', **kwargs):
        mask = rembg.remove(image, only_mask=True, **kwargs)
        return image, {name: mask}
    @unwrap_image
    def threshold_mask(self, image, bias=0, threshold=0, name='threshold_mask'):
        # Convert the image to grayscale
        grayscaled = self.grayscale(image)
        mask = np.ones(image.shape[:2], np.uint8) * 255
        if bias == 0:
            mask[np.where(grayscaled < threshold)] = 0
        else:
            mask[np.where(grayscaled >= threshold)] = 0
        return image, {name: mask}

    @unwrap_image
    def dominant_colors(self, image, bias=0, num_clusters=-1):
        # Convert the image to a single-channel, floating point representation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray= gray.reshape(-1,1)

        if num_clusters < 2:
            num_clusters = self.compute_num_clusters(gray, bias)

        kmeans = MiniBatchKMeans(n_clusters=num_clusters, max_iter=10000, batch_size=256*12, n_init=250)
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
        clustered_image = np.array(colors)[flat_labels].reshape(image.shape[0], image.shape[1], 3)


        # Determine which is the second most dominant cluster
        sorted_clusters = np.argsort(kmeans.cluster_centers_.mean(axis=1))
        if bias == 0:
            dominant_cluster = sorted_clusters[0]
            second_dominant_cluster = sorted_clusters[1]
        else:
            dominant_cluster = sorted_clusters[-1]
            second_dominant_cluster = sorted_clusters[-2]
        # Find the mean color of the dominant cluster and the secondary dominant cluster
        dominant_pixels = image[np.where(labels == dominant_cluster)]
        secondary_dominant_pixels = image[np.where(labels == second_dominant_cluster)]
        if bias == 0:
            mean_color = gray[np.where(labels == dominant_cluster)].max(axis=0).astype(np.uint8)
            secondary_mean_color = gray[np.where(labels == second_dominant_cluster)].max(axis=0).astype(np.uint8)
        else:
            mean_color = gray[np.where(labels == dominant_cluster)].min(axis=0).astype(np.uint8)
            secondary_mean_color = gray[np.where(labels == second_dominant_cluster)].min(axis=0).astype(np.uint8)
        return image, {'dominant_color':mean_color, 'dominant_pixels':dominant_pixels, 'secondary_dominant_color':secondary_mean_color, 'secondary_dominant_pixels':secondary_dominant_pixels, 'num_clusters':num_clusters, 'clustered_image':clustered_image}
    

class ImageManipulator:
    def __init__(self, image, editor = ImageEditor()):
        self.image = image
        self.imageEditor = editor

    def __getattr__(self, name):
        if hasattr(self.imageEditor, name):
            return partial(getattr(self.imageEditor, name), self.image)
        else:
            raise AttributeError("ImageEditor has no attribute: " + name)

    def find_mean_color(self, bias, num_clusters=-1, subsample=.25):
        # Subsample the image if specified
        if subsample < 1:
            image = cv2.resize(self.image, (0, 0), fx=subsample, fy=subsample, interpolation=cv2.INTER_LANCZOS4)
        else:
            image = self.image
        # Convert the image to a single-channel, floating point representation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray= gray.reshape(-1,1)

        if num_clusters < 2:
            num_clusters = self.compute_num_clusters(gray, bias)
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
        if bias == 0:
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

