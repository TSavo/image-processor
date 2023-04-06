from image_manipulator import ImageManipulator
import cv2
from pipeline import Pipeline

class ImagePipeline(Pipeline):
    def __init__(self, *items):
        super().__init__(*items)
        self.manipulator = ImageManipulator()

    def fork(self, num_forks=2):
        def fork_action(image):
            return tuple([image for _ in range(num_forks)])
        self.next_steps.append(fork_action)
        return self

    def __getattr__(self, name):
        if hasattr(self.manipulator, name):
            def wrapper(*args, **kwargs):
                def action(image):
                    return getattr(self.manipulator, name)(image, *args, **kwargs)
                self.next_steps.append(action)
                return self
            return wrapper
        else:
            raise AttributeError("ImageManipulator has no attribute " + name)
