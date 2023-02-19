from image_manipulator import ImageManipulator
import cv2
class ImagePipeline():
    def __init__(self):
        self.actions = []
    def crop(self, x, y, width, height):
        self.actions.append(("crop", (x, y, width, height)))
        return self
    def center_crop(self, width, height):
        self.actions.append(("center_crop", (width, height)))
        return self
    def resize(self, width, height):
        self.actions.append(("resize", (width, height)))
        return self
    def rotate(self, angle):
        self.actions.append(("rotate", (angle,)))
        return self
    def flip(self, axis):
        self.actions.append(("flip", (axis,)))
        return self
    def blur(self, kernel_size_x, kernel_size_y=None):
        if kernel_size_y is None:
            kernel_size_y = kernel_size_x
        self.actions.append(("blur", (kernel_size_x,kervel_size_y,)))
        return self
    def brightness(self, value):
        self.actions.append(("brightness", (value,)))
        return self
    def contrast(self, value):
        self.actions.append(("contrast", (value,)))
        return self
    def saturation(self, value):
        self.actions.append(("saturation", (value,)))
        return self
    def hue(self, value):
        self.actions.append(("hue", (value,)))
        return self
    def gamma(self, value):
        self.actions.append(("gamma", (value,)))
        return self
    def sharpen(self, value):
        self.actions.append(("sharpen", (value,)))
        return self
    def emboss(self, value):
        self.actions.append(("emboss", (value,)))
        return self
    def equalize(self):
        self.actions.append(("equalize", ()))
        return self
    def grayscale(self):
        self.actions.append(("grayscale", ()))
        return self
    def invert(self):
        self.actions.append(("invert", ()))
        return self
    def salt_pepper_noise(self, value):
        self.actions.append(("salt_pepper_noise", (value,)))
        return self
    def gaussian_noise(self, value):
        self.actions.append(("gaussian_noise", (value,)))
        return self
    def poisson_noise(self, value):
        self.actions.append(("poisson_noise", (value,)))
        return self
    def speckle_noise(self, value):
        self.actions.append(("speckle_noise", (value,)))
        return self
    def sp_noise(self, value):
        self.actions.append(("sp_noise", (value,)))
        return self
    def random_noise(self, value):
        self.actions.append(("random_noise", (value,)))
        return self
    def motion_blur(self, value):
        self.actions.append(("motion_blur", (value,)))
        return self
    def zoom_blur(self, value):
        self.actions.append(("zoom_blur", (value,)))
        return self
    def gaussian_blur(self, value):
        self.actions.append(("gaussian_blur", (value,)))
        return self
    def median_blur(self, value):
        self.actions.append(("median_blur", (value,)))
        return self
    def bilateral_blur(self, value):
        self.actions.append(("bilateral_blur", (value,)))
        return self
    def add(self, value):
        self.actions.append(("add", (value,)))
        return self
    def subtract(self, value):
        self.actions.append(("subtract", (value,)))
        return self
    def multiply(self, value):
        self.actions.append(("multiply", (value,)))
        return self
    def divide(self, value):
        self.actions.append(("divide", (value,)))
        return self
    def bitwise_and(self, value):
        self.actions.append(("bitwise_and", (value,)))
        return self
    def bitwise_or(self, value):
        self.actions.append(("bitwise_or", (value,)))
        return self
    def bitwise_xor(self, value):
        self.actions.append(("bitwise_xor", (value,)))
        return self
    def bitwise_not(self, value):
        self.actions.append(("bitwise_not", (value,)))
        return self
    def max_dimension(self, width, height):
        self.actions.append(("max_dimension", (width, height)))
        return self
    def max_size(self, value):
        self.actions.append(("max_size", (value,)))
        return self
    def mask_border(self, thickness, color):
        self.actions.append(("border", (thickness, color)))
        return self
    def copy(self):
        image_pipeline = ImagePipeline()
        image_pipeline.actions = self.actions.copy()
        return image_pipeline

    def blur_mask(self, kernel_size_x, kernel_size_y=None):
        if kernel_size_y is None:
            kernel_size_y = kernel_size_x
        self.actions.append(("blur_mask", (kernel_size_x,kernel_size_y,)))
        return self
    
    def process(self, image):
        if image.isinstance("numpy.ndarray"):
            manipulator = ImageManipulator(image)
        elif image.isinstance("str"):
            manipulator = ImageManipulator(cv2.imread(image))
        #else if image.isinstance("PIL.Image") convert to np array
        elif image.isinstance("PIL.Image"):
            manipulator = ImageManipulator(np.array(image))
        elif image.isinstance("ImageManipulator"):
            manipulator = image
        elif image.isinstance("ImagePipeline"):
            manipulator = ImageManipulator(image.process())
        #if there is a get_image, use it
        elif hasattr(image, "get_image") and callable(image.get_image):
            manipulator = ImageManipulator(image.get_image())
        else:
            raise Exception("Invalid image type")
        for action in self.actions:
            if action[0] == "copy":
                return manipulator.image.copy()
            if hasattr(manipulator, action[0]):
                manipulator.__getattribute__(action[0])(*action[1])
        return manipulator.image

    def __call__(self, image):
        return self.process(image)
    def batch_process(self, images):
        for image in images:
            yield self(image)  