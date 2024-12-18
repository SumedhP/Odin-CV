import cv2
import numpy as np
from cv2.typing import MatLike
from typing import List

from GridStride import generateGridsAndStride
from Match import Match, Point, mergeListOfMatches

from rknn.api import RKNN

class Model:
    INPUT_SIZE = 416
    BOUNDING_BOX_CONFIDENCE_THRESHOLD = 0.85

    color_to_word = ["Blue", "Red", "Neutral", "Purple"]
    tag_to_word = ["Sentry", "1", "2", "3", "4", "5", "Outpost", "Base"]

    def __init__(self, model_path: str = "models/model.rknn", core_mask=RKNN.NPU_CORE_ALL) -> None:
        self.model = RKNN(verbose=True)
        self.model.load_rknn(model_path)
        self.model.init_runtime(target="rk3588", core_mask=core_mask)

        self.grid_strides = generateGridsAndStride()
        self.grid_strides = np.array(self.grid_strides)

    def processInput(self, img: MatLike) -> List[Match]:
        img, scalar_h, scalar_w, x_cutoff = self.formatInput(img)
        
        output = self.model.inference(inputs=[img], data_format="nhwc", inputs_pass_through=[0])

        output = np.array(output)[0][0]

        boxes = self.getBoxesFromOutput(output)
        
        # print("Found ", len(boxes), " boxes: \n")

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

        img = cv2.resize(img, (self.INPUT_SIZE, self.INPUT_SIZE))
        
        assert img.shape == (416, 416, 3)
        
        # Add in the n value (batch size of 1) to the image
        img = np.expand_dims(img, axis=0)

        return img, scalar_h, scalar_w, x_cutoff

    def getBoxesFromOutput(self, values) -> List[Match]:
        boxes = []

        NUM_COLORS = 8
        NUM_TAGS = 8

        indices = np.where(values[:, 8] > self.BOUNDING_BOX_CONFIDENCE_THRESHOLD)
        values = values[indices]

        curr_grid_strides = self.grid_strides[indices]

        for element, grid_stride in zip(values, curr_grid_strides):
            grid0, grid1, stride = (
                grid_stride.grid0,
                grid_stride.grid1,
                grid_stride.stride,
            )

            x_1 = (element[0] + grid0) * stride
            y_1 = (element[1] + grid1) * stride
            x_2 = (element[2] + grid0) * stride
            y_2 = (element[3] + grid1) * stride
            x_3 = (element[4] + grid0) * stride
            y_3 = (element[5] + grid1) * stride
            x_4 = (element[6] + grid0) * stride
            y_4 = (element[7] + grid1) * stride

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


def putTextOnImage(img: MatLike, boxes: List[Match]) -> MatLike:

    for i in range(len(boxes)):
        box = boxes[i]
        for j in range(4):
            cv2.line(
                img,
                (int(box.points[j].x), int(box.points[j].y)),
                (int(box.points[(j + 1) % 4].x), int(box.points[(j + 1) % 4].y)),
                (0, 255, 0),
                2,
            )
        cv2.putText(
            img,
            f"{box.color} {box.tag}",
            (int(box.points[0].x), int(box.points[0].y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return img


def main():

    input_file = "../test_image.jpg"

    model = Model()
    img = cv2.imread(input_file)
    
    # Take center crop of 540x540
    x_cutoff = 960 - 540
    x_cutoff = x_cutoff // 2
    img = img[:540, x_cutoff:x_cutoff+540]
    

    boxes = model.processInput(img)
    print("Found ", len(boxes), " boxes: \n")
    for box in boxes:
        print(box)

    img = putTextOnImage(img, boxes)

    output_file = "../labelled_image.jpg"
    cv2.imwrite(output_file, img)
    
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
