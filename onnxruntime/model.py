import cv2
import numpy as np
from cv2.typing import MatLike
from typing import List

from OutputScalar import generateOffsetsAndScalars
from Match import Match, Point, mergeListOfMatches

import onnxruntime as ort


class Model:
    INPUT_SIZE = 416
    BOUNDING_BOX_CONFIDENCE_THRESHOLD = 0.85

    color_to_word = ["Blue", "Red", "Neutral", "Purple"]
    tag_to_word = ["Sentry", "1", "2", "3", "4", "5", "Outpost", "Base"]

    def __init__(self, model_path: str = "model/model.onnx") -> None:
        providers = self.getConfiguredProviders()

        self.model = ort.InferenceSession(model_path, providers=providers)

        self.outputOffsetAndScale = generateOffsetsAndScalars()

    # This sets what to run the model on, priority is tensorrt if avaialble, then CUDA, then CPU (default)
    def getConfiguredProviders(self):
        return [
            (
                "TensorrtExecutionProvider",
                {
                    # Select GPU to execute, doesn't matter in our case
                    "device_id": 0,
                    # Set GPU memory usage limit, this is in bytes (2**30 = 1GB)
                    "trt_max_workspace_size": (2**30) * 4,
                    # Enable INT8 precision for faster inference (3ms -> 1.75ms)
                    "trt_int8_enable": True,
                    # Cache created engine so it doesn't have to be recreated every time
                    "trt_engine_cache_enable": True,
                    # Directory to store the cached engine
                    "trt_engine_cache_path": "./trt_engines",
                    # Setup a CUDA graph, this maybe help optimize our model since it has so many layers
                    "trt_cuda_graph_enable": True,
                    # Max out optimization level for funsies
                    "trt_builder_optimization_level": 5,
                },
            ),
            ("CUDAExecutionProvider"),
        ]

    def processInput(self, img: MatLike) -> List[Match]:
        img, scalar_h, scalar_w, x_cutoff = self.formatInput(img)

        output = self.model.run(None, {"images": img})
        output = np.array(output)[0][0]

        boxes = self.getBoxesFromOutput(output)

        # Scale the boxes back to the original image size
        for box in boxes:
            for i in range(4):
                box.points[i].x = box.points[i].x * scalar_w + x_cutoff
                box.points[i].y = box.points[i].y * scalar_h

        boxes = mergeListOfMatches(boxes)

        return boxes

    # Resize and format to nhwc format needed by the model
    def formatInput(self, img: MatLike):
        # Chop off the sides to make it square
        x_cutoff = 0
        if img.shape[1] != img.shape[0]:
            x_cutoff = img.shape[1] - img.shape[0]
            x_cutoff = x_cutoff // 2
            img = img[:, x_cutoff : x_cutoff + img.shape[0]]

        # Resize the image to the input size of the model
        scalar_h = img.shape[0] / self.INPUT_SIZE
        scalar_w = img.shape[1] / self.INPUT_SIZE

        # Image shape is resized to (416, 416, 3)
        img = cv2.resize(img, (self.INPUT_SIZE, self.INPUT_SIZE))

        # Model input expects (1, 3, 416, 416)
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)

        # Convert to float32
        img = img.astype(np.float32)

        return img, scalar_h, scalar_w, x_cutoff

    def getBoxesFromOutput(self, values) -> List[Match]:
        boxes = []

        NUM_COLORS = 8
        NUM_TAGS = 8

        indices = np.where(values[:, 8] > self.BOUNDING_BOX_CONFIDENCE_THRESHOLD)
        values = values[indices]

        outputOffsetAndScale = self.outputOffsetAndScale[indices]

        for element, offsetAndScale in zip(values, outputOffsetAndScale):
            x_offset, y_offset, scalar = (
                offsetAndScale.x_offset,
                offsetAndScale.y_offset,
                offsetAndScale.scalar,
            )

            x_1 = (element[0] + x_offset) * scalar
            y_1 = (element[1] + y_offset) * scalar
            x_2 = (element[2] + x_offset) * scalar
            y_2 = (element[3] + y_offset) * scalar
            x_3 = (element[4] + x_offset) * scalar
            y_3 = (element[5] + y_offset) * scalar
            x_4 = (element[6] + x_offset) * scalar
            y_4 = (element[7] + y_offset) * scalar

            confidence = element[8]

            color = np.argmax(element[9 : 9 + NUM_COLORS])
            tag = np.argmax(element[9 + NUM_COLORS : 9 + NUM_COLORS + NUM_TAGS])

            bottomLeft = Point(x_1, y_1)
            topLeft = Point(x_2, y_2)
            topRight = Point(x_3, y_3)
            bottomRight = Point(x_4, y_4)

            box = Match(
                [bottomLeft, topLeft, topRight, bottomRight],
                self.color_to_word[int(color / 2)],
                self.tag_to_word[tag],
                confidence,
            )
            boxes.append(box)

        # Sort the boxes by confidence
        boxes.sort(reverse=True)

        return boxes


def main():

    input_file = "test_image.jpg"

    model = Model()
    img = cv2.imread(input_file)

    # Take center crop of 540x540
    x_cutoff = 960 - 540
    x_cutoff = x_cutoff // 2
    img = img[:540, x_cutoff : x_cutoff + 540]

    boxes = model.processInput(img)
    print("Found ", len(boxes), " boxes: \n")
    for box in boxes:
        print(box)

    def timing():
        from time import time_ns as time
        from tqdm import tqdm

        timings = []
        ITERATIONS = 1000
        for _ in tqdm(range(ITERATIONS)):
            start = time()
            model.processInput(img)
            end = time()
            timings.append(end - start)
        timings = np.array(timings)
        # Print avg time in ms and FPS
        print(f"Average time: {np.mean(timings) / 1e6} ms")
        print(f"Average FPS: {1e9 / np.mean(timings)}")

    timing()


if __name__ == "__main__":
    main()
