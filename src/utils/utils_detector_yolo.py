import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from yolov5.utils.loss import ComputeLoss

from art.estimators.estimator import BaseEstimator
from art.estimators.object_detection.pytorch_yolo import PyTorchYolo

from utils.utils import nms_on_predictions

COCO_INSTANCE_CATEGORY_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

class UtilsDetectorYolo():
    """
    Utility class for working with YOLO object detection models.
    """

    def __init__(self, N, threshold=0.7, victim_class=""):
        """
        Initialize the UtilsDetectorYolo class.

        Args:
            N (int): The number of images to load for testing.
            threshold (float, optional): Confidence threshold for filtering predictions. Defaults to 0.7.
            victim_class (str, optional): The name of the victim class to filter images. Defaults to everything.
        """
        self.estimator : BaseEstimator
        self.images = np.array([])

        if victim_class != "" and victim_class not in COCO_INSTANCE_CATEGORY_NAMES:
            raise ValueError(f"Victim class '{victim_class}' is not in the known labels.")

        print("Loading the model")
        self.load_model()
        print("Loading the dataset")
        self.images = self.load_dataset(N, threshold=threshold, victim_class=victim_class)

        print(f"Number of images: {len(self.images)}")

    def load_model(self):
        """
        Load the YOLOv5 object detection model and configure it for use with the Adversarial Robustness Toolbox (ART).

        This method initializes the YOLOv5 model, wraps it in a custom PyTorch module to compute losses during training,
        and sets it up as an ART estimator for adversarial attacks.

        Steps:
        1. Load the YOLOv5 model from a pre-trained checkpoint file (`yolov5s.pt`).
        2. Define a custom PyTorch module (`Yolo`) to handle both training and inference modes.
           - In training mode, the module computes the total loss and its components (`loss_box`, `loss_obj`, `loss_cls`).
           - In inference mode, the module outputs raw predictions.
        3. Wrap the YOLOv5 model in the custom `Yolo` class and set it to evaluation mode.
        4. Create an ART `PyTorchYolo` estimator using the wrapped model, specifying input shape, clip values, and attack losses.

        Attributes:
            self.model (torch.nn.Module): The YOLOv5 model wrapped in the custom `Yolo` class.
            self.estimator (PyTorchYolo): The ART estimator for the YOLOv5 model, used for adversarial attacks.

        Raises:
            FileNotFoundError: If the YOLOv5 checkpoint file (`yolov5s.pt`) is not found.

        Notes:
            - The model is loaded onto the GPU (`cuda:0`) for faster computation.
            - The `ComputeLoss` class from YOLOv5 is used to calculate the loss components during training.
            - The associated labels for this model are stored in the `COCO_INSTANCE_CATEGORY_NAMES` list defined in this file.        
        """
        matplotlib.use("Agg")

        class Yolo(torch.nn.Module):
            """
            Custom PyTorch module for YOLOv5 to handle both training and inference modes.

            Attributes:
                model (torch.nn.Module): The YOLOv5 model.
                compute_loss (ComputeLoss): Utility to compute loss components during training.
            """

            def __init__(self, model: torch.nn.Module):
                """
                Initialize the custom YOLOv5 module.

                Args:
                    model (torch.nn.Module): The YOLOv5 model to wrap.
                """
                super().__init__()
                self.model = model
                self.model.hyp = {
                    'box': 0.05,
                    'obj': 1.0,
                    'cls': 0.5,
                    'anchor_t': 4.0,
                    'cls_pw': 1.0,
                    'obj_pw': 1.0,
                    'fl_gamma': 0.0
                }
                self.compute_loss = ComputeLoss(self.model)

            def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
                """
                Forward pass for the YOLOv5 model.

                Args:
                    x (torch.Tensor): Input tensor of shape (B, C, H, W), where B is the batch size, 
                                      C is the number of channels, H is the height, and W is the width.
                    targets (torch.Tensor, optional): Ground truth targets for training. Defaults to None.

                Returns:
                    dict: If in training mode, returns a dictionary with loss components:
                          - "loss_total" (torch.Tensor): Total loss.
                          - "loss_box" (torch.Tensor): Bounding box regression loss.
                          - "loss_obj" (torch.Tensor): Objectness loss.
                          - "loss_cls" (torch.Tensor): Classification loss.
                    torch.Tensor: If in inference mode, returns raw predictions of shape (B, N, 85), 
                                  where N is the number of predictions per image.
                """
                if self.training:
                    outputs = self.model(x)
                    loss, loss_items = self.compute_loss(outputs, targets)
                    return {
                        "loss_total": loss,
                        "loss_box": loss_items[0],
                        "loss_obj": loss_items[1],
                        "loss_cls": loss_items[2]
                    }
                else:
                    return self.model(x)[0]

        self.model = torch.load("../data/yolov5s.pt")["model"].float().to('cuda:0')
        self.model = Yolo(self.model)
        self.model.eval()


        self.estimator = PyTorchYolo(model=self.model,
                            input_shape=(3, 640, 640),
                            clip_values=(0, 255), 
                            attack_losses=("loss_total", "loss_cls",
                                            "loss_box",
                                            "loss_obj"))


    def load_dataset(self, N, threshold=0.7, victim_class="", input_size=(640, 640)):
        """
        Load dataset from the desired folder.

        Args:
            N (int): The number of images to load.
            threshold (float): Confidence threshold for filtering predictions.
            victim_class (str): The name of the victim class to filter images. If empty, no filtering is applied.
            input_size (tuple[int, int]): The size to which images should be resized (height, width).

        Returns:
            tuple[np.ndarray, list]: 
                - A NumPy array of images with shape `(N, C, H, W)` where `C` is the number of channels, 
                `H` is the height, and `W` is the width.
        """    
        transform = transforms.Compose([
            transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])

        # Custom images
        images = []

        folder_path = '../data/custom_images/'
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            if not os.path.isfile(img_path):
                continue
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img)
            images.append(img_tensor)

        not_filtered_images = images[:]

        # Convert to numpy array
        not_filtered_images = torch.stack(not_filtered_images).numpy() * 255

        
        if victim_class != "":
            # Filter images based on the predictions and the victim class
            filtered_images = []
            final_count = 0

            batch_s = 30

            for i in range(0, len(not_filtered_images), batch_s):
                predictions = self.estimator.predict(not_filtered_images[i:i+batch_s])
                dets = nms_on_predictions(predictions)

                for i, img in enumerate(not_filtered_images[i:i+batch_s]):
                    preds = self.extract_predictions(dets[i], threshold)
                    if victim_class in preds[0]:
                        filtered_images.append(img)
                        final_count += 1
                    if final_count >= N:
                        break
                if final_count >= N:
                    break
            print(final_count, "images with victim class found")
            return np.array(filtered_images)
        else:
            return not_filtered_images[:N]

    def id_to_name_class(self, id):
        """
        Convert the COCO ID to the class name.
        """
        if id < len(COCO_INSTANCE_CATEGORY_NAMES):
            return COCO_INSTANCE_CATEGORY_NAMES[id]
        else:
            return "Unknown"

    def name_class_to_id(self, name):
        """
        Convert the class name to the COCO ID.
        """
        if name in COCO_INSTANCE_CATEGORY_NAMES:
            return COCO_INSTANCE_CATEGORY_NAMES.index(name)
        else:
            return -1

    def extract_predictions(self, predictions_, conf_thresh):
        """
        This function processes the model output to filter out predictions based on a confidence threshold.
        Args:
            predictions_ (dict): The model output containing 'labels', 'boxes', and 'scores'.
                                The value of each key should be a list or numpy array. To avoid issues with single predictions,
                                we ensure that they are always treated as lists.
            conf_thresh (float): Confidence threshold for filtering predictions.
        Returns:
            tuple: A tuple containing:
                - predictions_class (list): List of predicted class names.
                - predictions_boxes (list): List of predicted bounding boxes.
                - predictions_scores (list): List of predicted scores.
        """
        if isinstance(predictions_["labels"], (int, np.integer)) or isinstance(predictions_["labels"], str):
            predictions_["labels"] = np.array([predictions_["labels"]])
            predictions_["boxes"] = np.array([predictions_["boxes"]])
            predictions_["scores"] = np.array([predictions_["scores"]])

        # Get the predicted class
        predictions_class = [self.id_to_name_class(i) for i in list(predictions_["labels"])]
        if len(predictions_class) < 1:
            return [], [], []
        # Get the predicted bounding boxes
        predictions_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(predictions_["boxes"])]

        # Get the predicted prediction score
        predictions_score = list(predictions_["scores"])

        # Get a list of index with score greater than threshold
        threshold = conf_thresh
        predictions_t = [predictions_score.index(x) for x in predictions_score if x > threshold]
        if len(predictions_t) > 0:
            predictions_t = predictions_t  # [-1] #indices where score over threshold
        else:
            # no predictions esxceeding threshold
            return [], [], []
        # predictions in score order
        predictions_boxes = [predictions_boxes[i] for i in predictions_t]
        predictions_class = [predictions_class[i] for i in predictions_t]
        predictions_scores = [predictions_score[i] for i in predictions_t]
        return predictions_class, predictions_boxes, predictions_scores


    def plot_image(self, img, pred_cls, title, boxes=[], folder="../data/", show=False):
        """
        Plot an image with bounding boxes and predicted classes.
        
        Args:
            img (np.ndarray): The image to plot.
            pred_cls (list): List of predicted class names.
            title (str): Title for the plot.
            boxes (list): List of bounding boxes, each box is a tuple of two points ((x1, y1), (x2, y2)).
            folder (str): Folder to save the image.
            show (bool): Whether to display the image using plt.show(). Defaults to False.
        """
        plt.style.use("ggplot")
        text_size = 1
        text_th = 3
        rect_th = 1

        img = img.copy()

        for i in range(len(boxes)):
            cv2.rectangle(
                img,
                (int(boxes[i][0][0]), int(boxes[i][0][1])),
                (int(boxes[i][1][0]), int(boxes[i][1][1])),
                color=(255, 0, 0),
                thickness=rect_th,
            )
            # Write the prediction class
            cv2.putText(
                img,
                pred_cls[i],
                (int(boxes[i][0][0]), int(boxes[i][0][1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_size,
                (255, 0, 0),
                thickness=text_th,
            )
        os.makedirs(folder, exist_ok=True)

        plt.figure()
        plt.axis("off")
        if boxes is not None:
            plt.title(title)
        else:
            plt.title(pred_cls)

        plt.imshow(img.astype(np.uint8), interpolation="nearest")
        plt.savefig(folder + title + ".png")
        
        if show:
            plt.show()
        plt.close()

    def prediction_format_to_art_format(self, orig_predictions):
        """
        Convert predictions from the original format to the ART format.

        At the end of the method, the list in every dictionary will have the same size.
        To do that, the method will repeat all the elements in the list until it reaches the maximum size of the list.
        For an image with no predictions (due to a too big threshold), it will create a dummy prediction with a box of size 0 and a class label of -1.
        
        Args:
            orig_predictions (list): Original predictions in the format:
                [
                    ([class_name1, ...], [[(y1, x1), (y2, x2)], ...], [score1, ...]),
                    ...
                ].

        Returns:
            list: Predictions in ART format with keys:
                - 'boxes': np.ndarray of shape (N, 4) with box coordinates [y1, x1, y2, x2].
                - 'labels': np.ndarray of shape (N,) with class labels.
                - 'scores': np.ndarray of shape (N,) with confidence scores.
        """
        # If the same size is needed in the predictions, replace the len(pred[1]) by the max_size in the loops.
        max_size = 0
        for pred in orig_predictions:
            if len(pred[1]) > max_size:
                max_size = len(pred[1])

        predictions = []
        for pred in orig_predictions:
            if len(pred[1]) == 0:
                predictions.append({
                    'boxes': np.array([[0, 0, 0, 0]] * max_size),
                    'labels': np.array([-1] * max_size),
                    'scores': np.array([0.0] * max_size)
                })
                continue
            predictions.append({
                'boxes': np.array([[pred[1][i % len(pred[1])][0][0], pred[1][i % len(pred[1])][0][1], pred[1][i % len(pred[1])][1][0], pred[1][i % len(pred[1])][1][1]] for i in range(max_size)]),
                'labels': np.array([self.name_class_to_id(pred[0][i % len(pred[0])]) for i in range(max_size)]),
                'scores': np.array([pred[2][i % len(pred[1])] for i in range(max_size)])
            })
        return predictions