from utils.utils import nms_on_predictions
from art.attacks import EvasionAttack
import numpy as np

class WhiteBoxAttack():
    """
    Base class for white-box attacks. Provides methods for generating adversarial patchs and applying them to images.
    """

    def __init__(self, estimator, images, orig_predictions, target_class=None):
        """
        Initialize the WhiteBoxAttack class.

        Args:
            estimator (object): The object detection model estimator.
            images (np.ndarray): Array of images to apply the attack on. Shape: (BATCH, C, H, W).
            orig_predictions (list): Original predictions in the format:
                [
                    ([class_name1, ...], [[(y1, x1), (y2, x2)], ...], [score1, ...]),
                    ...
                ].
            target_class (str, optional): The target class for the attack. Defaults to None.
        """
        self.estimator = estimator
        self.attack: EvasionAttack = None
        self.attack_name = ""
        self.target_class = target_class
        self.orig_predictions = orig_predictions
        self.images = images

        self.adversarial_example = []

    def generate(self, images, target_shape, target_location, targets=None, orig_predictions=(), norm=np.inf, eps=0.3, max_iter=1000):
        """
        Abstract method to generate adversarial examples. Should be implemented by subclasses.

        Args:
            images (np.ndarray): Array of images to apply the attack on. Shape: (BATCH, C, H, W).
            target_shape (tuple): Shape of the adversarial patch. Format: (C, H, W).
            target_location (list): Location of the patch in the image. Format: [y, x].
            targets (list, optional): List of target dictionaries for the attack. Defaults to None.
            orig_predictions (tuple, optional): Original predictions. Defaults to ().
            norm (float, optional): Norm to use for the attack. Defaults to np.inf.
            eps (float, optional): Maximum perturbation allowed for the attack. Defaults to 0.3.
            max_iter (int, optional): Maximum number of iterations for the attack. Defaults to 1000.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be called by subclasses.")

    def apply_attack_to_image(self, image=[], train_on=1, image_name="", threshold=0.5, plot_and_predictions=True):
        """
        Apply the adversarial attack to a single image or a list of images and save it with the predictions on it.
        If a list is given, the attack will be applied to each image in the list and the image name will be auto-incremented.

        Args:
            image (np.ndarray): The image to apply the attack on. Shape: (C, H, W) or (BATCH, C, H, W).
            train_on (int, optional): Number of images used to train the patch. Defaults to 1.
            image_name (str, optional): Name of the image for saving results. Defaults to "".
            threshold (float, optional): Confidence threshold for predictions. Defaults to 0.5.
            plot_and_predictions (bool, optional): Whether to plot the image and generate predictions. Defaults to True.

        Returns:
            tuple: 
                - predictions (list): Predictions after applying the attack.
                - adversarial_example (np.ndarray): The adversarial example(s) generated.

        Raises:
            ValueError: If no adversarial examples are generated.
        """
        if len(self.adversarial_example) == 0:
            raise ValueError("This method should be called by subclasses.")

        if image_name == "":
            image_name = self.attack_name

        image_to_display = 3  # Only used in the notebooks

        dets = self.estimator.estimator.predict(self.adversarial_example)
        # Apply NMS on the predictions
        dets = nms_on_predictions(dets, iou_threshold=0.5)

        predictions = []
        for i in range(len(dets)):
            prediction = self.estimator.extract_predictions(dets[i], threshold)
            self.estimator.plot_image(
                img=self.adversarial_example[i].transpose(1, 2, 0).copy(),
                boxes=prediction[1],
                pred_cls=prediction[0],
                title=f"{(image_name + str(i)) if len(dets) > 1 else image_name}",
                folder=f"../data/{self.attack_name}/",
                show=(i < image_to_display)
            )
            predictions.append(prediction)

        if len(predictions) == 3 and len(image) == 1:
            predictions = np.array([predictions])
            self.adversarial_example = [self.adversarial_example]

        return predictions, self.adversarial_example

    def generate_fake_target(self, target_location, target_shape):
        """
        Generate a fake target for the attack. 

        It creates a target dictionary with a single bounding box that matches the adversarial patch's location and shape.
        The target class used is the one specified during initialization.
        This method cannot be used in untargeted mode.

        Args:
            target_location (list): Location of the patch in the image. Format: [y, x].
            target_shape (tuple): Shape of the adversarial patch. Format: (C, H, W).

        Returns:
            dict: Fake target dictionary containing:
                - 'boxes': np.ndarray of shape (1, 4) with box coordinates [y1, x1, y2, x2].
                - 'labels': np.ndarray of shape (1,) with the class label.
                - 'scores': np.ndarray of shape (1,) with the confidence score.

        Raises:
            ValueError: If the target class is not specified.
        """
        if self.target_class is '':
            raise ValueError("Target class must be specified to generate a fake target.")

        idx = self.estimator.name_class_to_id(self.target_class)
        box_location = [target_location[0], target_location[1], target_location[0] + target_shape[1], target_location[1] + target_shape[2]]
        target_dict = {
            "boxes": np.array([box_location], dtype=np.float32),
            "labels": np.array([idx], dtype=np.int64),
            "scores": np.array([0.8], dtype=np.float32),
        }
        return target_dict

    def insert_transformed_patch(self, x: np.ndarray, patch: np.ndarray, image_coords: np.ndarray):
        """
        Recopy of the insert_transformed_patch method from the AdversarialPatchPyTorch attack.


        Insert patch to image based on given or selected coordinates.

        Args:
            x (np.ndarray): A single image of shape HWC to insert the patch.
            patch (np.ndarray): The patch to be transformed and inserted.
            image_coords (np.ndarray): The coordinates of the 4 corners of the transformed, inserted patch of shape
                [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] in pixel units going in clockwise direction, starting with upper
                left corner.

        Returns:
            np.ndarray: The input image `x` with the patch inserted.
        """
        import cv2

        scaling = False

        if np.max(x) <= 1.0:
            scaling = True
            x = (x * 255).astype(np.uint8)
            patch = (patch * 255).astype(np.uint8)

        rows = patch.shape[0]
        cols = patch.shape[1]

        if image_coords.shape[0] == 4:
            patch_coords = np.array([[0, 0], [cols - 1, 0], [cols - 1, rows - 1], [0, rows - 1]])
        else:
            patch_coords = np.array(
                [
                    [0, 0],
                    [cols - 1, 0],
                    [cols - 1, (rows - 1) // 2],
                    [cols - 1, rows - 1],
                    [0, rows - 1],
                    [0, (rows - 1) // 2],
                ]
            )

        # calculate homography
        height, _ = cv2.findHomography(patch_coords, image_coords)

        # warp patch to destination coordinates
        x_out = cv2.warpPerspective(patch, height, (x.shape[1], x.shape[0]), cv2.INTER_CUBIC)  # type: ignore

        # mask to aid with insertion
        mask = np.ones(patch.shape)
        mask_out = cv2.warpPerspective(mask, height, (x.shape[1], x.shape[0]), cv2.INTER_CUBIC)  # type: ignore

        # save image before adding shadows
        x_neg_patch = np.copy(x)
        x_neg_patch[mask_out != 0] = 0  # negative of the patch space

        if x_neg_patch.shape[2] == 1:
            x_out = np.expand_dims(x_out, axis=2)

        x_out = x_neg_patch.astype("float32") + x_out.astype("float32")

        if scaling:
            x_out = x_out / 255

        return x_out