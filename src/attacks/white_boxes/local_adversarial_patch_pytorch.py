from attacks.white_boxes.white_box_attack import WhiteBoxAttack
from utils.utils import check_image_format

from art.attacks.evasion import AdversarialPatchPyTorch

import numpy as np


class LocalAdversarialPatchPytorch(WhiteBoxAttack):
    """
    Implementation of the Local Adversarial Patch attack using PyTorch. This attack generates adversarial patches
    and applies them to specific locations in images to fool object detection models.
    """

    def __init__(self, estimator, images, orig_predictions, target_class=None):
        """
        Initialize the LocalAdversarialPatchPytorch class.

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
        super().__init__(estimator, images, orig_predictions, target_class)
        self.attack_name = "AdversarialPatchPytorch"
        self.patch = None
        self.patch_mask = None
        self.target_location = ()
        self.target_shape = ()

    def generate(self, images, target_shape, target_location, targets=None, orig_predictions=(), norm=np.inf, eps=0.3, max_iter=1000):
        """
        Generate an adversarial patch using the AdversarialPatchPyTorch attack for a batch of images.

        Args:
            images (np.ndarray): The input images to attack. Shape: (BATCH, C, H, W).
            target_shape (tuple): Shape of the patch to be generated. Format: (C, H, W).
            target_location (tuple): Location where the patch should be applied. Format: (y, x).
            targets (list, optional): List of target dictionaries for the attack. Defaults to None.
            orig_predictions (tuple, optional): Original predictions for the images from the current model.
                                                Format: ([class_name1, ...], [[(y1, x1), (y2, x2)], ...], [score1, ...]).
            norm (float, optional): Useless parameter for this attack, but kept for compatibility. Defaults to np.inf.
            eps (float, optional): Useless parameter for this attack, but kept for compatibility. Defaults to 0.3.
            max_iter (int): Maximum number of iterations for the attack.

        Returns:
            tuple: A tuple containing:
                - patch (np.ndarray): The generated adversarial patch.
                - attack (AdversarialPatchPyTorch): The attack object.
        """
        img_size = min(images[0].shape[1:3])
        scale = min(target_shape[1:]) / img_size
        rotation_max = 0.0
        scale_min = (scale - 0.1) if scale > 0.1 else 0.0
        scale_max = (scale + 0.1) if scale < 0.9 else 1.0
        distortion_scale_max = 0.0
        learning_rate = 1.99
        batch_size = 4 if not isinstance(self.estimator.estimator, PyTorchFasterRCNN) else 1
        patch_type = "square"
        optimizer = "pgd"

        detector = self.estimator.estimator

        # Create the AdversarialPatchPyTorch attack object with specified parameters
        self.attack = AdversarialPatchPyTorch(
            estimator=detector,
            rotation_max=rotation_max,
            scale_min=scale_min,
            scale_max=scale_max,
            optimizer=optimizer,
            distortion_scale_max=distortion_scale_max,
            learning_rate=learning_rate,
            max_iter=max_iter,
            batch_size=batch_size,
            patch_shape=target_shape,
            patch_type=patch_type,
            verbose=True,
            targeted=(self.target_class != ''),
            summary_writer = False
        )

        if self.target_class != '':
            # Generate the adversarial patch
            if targets is None:
                targets = [self.generate_fake_target(target_location, target_shape) for _ in range(len(images))]
            self.patch, self.patch_mask = self.attack.generate(x=images, y=targets)
        else:
            # Format the original predictions for the attack
            predictions_formatted = self.estimator.prediction_format_to_art_format(orig_predictions)
            self.patch, self.patch_mask = self.attack.generate(x=images, y=predictions_formatted)

        self.target_location = target_location
        self.target_shape = target_shape

        return self.patch, self.attack

    def apply_attack_to_image(self, image=[], train_on=1, image_name="", threshold=0.5, plot_and_predictions=True):
        """
        Apply the previously generated adversarial patch to the input image(s).

        Args:
            image (np.ndarray): The input image(s) to which the patch will be applied. Shape: (BATCH, C, H, W) or (C, H, W).
            train_on (int, optional): Number of images used to train the patch. Defaults to 1.
            image_name (str, optional): The name of the image for saving purposes. Defaults to "".
            threshold (float, optional): Threshold for filtering predictions. Defaults to 0.5.
            plot_and_predictions (bool, optional): Whether to plot the image and generate predictions. Defaults to True.

        Returns:
            tuple: A tuple containing:
                - predictions (list): List of predictions after applying the attack.
                - adversarial_example (np.ndarray): The adversarial example(s) generated by the attack.

        Raises:
            ValueError: If the attack, patch, patch location, or patch shape has not been generated.
        """
        if not self.attack:
            raise ValueError("The attack has not been generated. Call generate() first.")
        if self.patch is None:
            raise ValueError("The patch has not been generated. Call generate() first.")
        if len(self.target_location) == 0:
            raise ValueError("The patch location has not been given. Call generate() first.")
        if len(self.target_shape) == 0:
            raise ValueError("The patch shape has not been given. Call generate() first.")

        if len(image) == 0:
            image = self.images

        # Check if there is only one image to patch and expand dimensions if necessary
        image = check_image_format(image)

        coords = np.array([
            [self.target_location[1], self.target_location[0]],
            [self.target_location[1] + self.target_shape[2], self.target_location[0]],
            [self.target_location[1] + self.target_shape[2], self.target_location[0] + self.target_shape[1]],
            [self.target_location[1], self.target_location[0] + self.target_shape[1]],
        ], dtype=np.float32)

        self.adversarial_example = []
        for img in image:
            patched = self.insert_transformed_patch(
                x=img.transpose(1, 2, 0),
                patch=self.patch.transpose(1, 2, 0),  # Convert from (C, H, W) to (H, W, C)
                image_coords=coords
            ).transpose(2, 0, 1)

            self.adversarial_example.append(patched)
        self.adversarial_example = np.array(self.adversarial_example)
        
        if plot_and_predictions:
            return super().apply_attack_to_image(
                image=image,
                train_on=train_on,
                image_name=image_name,
                threshold=threshold,
                plot_and_predictions=plot_and_predictions
            )
        return [], self.adversarial_example
