import cv2
import numpy as np
from cv2.typing import MatLike
from typing import List
import onnxruntime as ort
import rknn.api
import rknn.utils

from GridStride import generateGridsAndStride
from Match import Match, Point, mergeListOfMatches

from time import time_ns

import rknn

from rknn.api import RKNN


print("Starting model")


model_path = "model.onnx"
RKNN_PATH = "model_quantizied.rknn"

rknn = RKNN(verbose=True)

# rknn.config(target_platform='rk3588')

# ret = rknn.load_onnx(model_path)
# if ret != 0:
#     print('Load model failed!')
#     exit(ret)

# ret = rknn.build(do_quantization=True, dataset='../2_17.txt')
# # ret = rknn.build(do_quantization=False)

# if ret != 0:
#         print('Build model failed!')
#         exit(ret)

# rknn.export_rknn(RKNN_PATH)
# if ret != 0:
#         print('Export rknn model failed!')
#         exit(ret)

# print("We've exported the model")
    
rknn.load_rknn(RKNN_PATH)    

ret = rknn.init_runtime(target='rk3588', core_mask=RKNN.NPU_CORE_AUTO)

if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)

print("Made model")


BBOX_CONFIDENCE_THRESHOLD = 0.85
grid_strides = generateGridsAndStride()
grid_strides = np.array(grid_strides)


from line_profiler import profile

@profile
def getBoxesFromOutput(output) -> List[Match]:
    boxes = []
    values = output[0]
    values = np.array(values)

    NUM_COLORS = 8
    NUM_TAGS = 8

    confience_values = values[:, 8]
    indices = np.where(confience_values > BBOX_CONFIDENCE_THRESHOLD)
    values = values[indices]
    
    # print("Max confidence: ", np.max(confience_values))
    curr_grid_strides = grid_strides[indices]

    for i in range(len(values)):
        element = values[i]

        grid0 = curr_grid_strides[i].grid0
        grid1 = curr_grid_strides[i].grid1
        stride = curr_grid_strides[i].stride

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
            color_to_word[int(color / 2)],
            tag_to_word[tag],
            confidence,
        )
        boxes.append(box)

    # Sort the boxes by confidence
    boxes.sort(reverse=True)

    return boxes

@profile
def makeImageAsInput(img: MatLike) -> np.ndarray:
    # Assert that the image is 416x416
    assert img.shape == (416, 416, 3)
    
    # img = img.astype(np.float32)
    # img = img.astype(np.int8)
    img = np.expand_dims(img, axis=0)
    
    return img

@profile
def getBoxesForImg(img: MatLike) -> List[Match]:
    img = makeImageAsInput(img)
    
    output = rknn.inference(inputs=[img], data_format='nhwc', inputs_pass_through=[0])
    
    output = np.array(output)

    output = output[0]
    
    boxes = getBoxesFromOutput(output)
    return boxes

color_to_word = [
    "Blue",
    "Red",
    "Neutral",
    "Purple",
]
tag_to_word = ["Sentry", "1", "2", "3", "4", "5", "Outpost", "Base"]


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

TARGET_WH = 416
def compressImageAndScaleOutput(img: MatLike) -> List[Match]:
    input_w = img.shape[1]
    input_h = img.shape[0]

    scalar_w = input_w / TARGET_WH
    scalar_h = input_h / TARGET_WH
    img = cv2.resize(img, (TARGET_WH, TARGET_WH))

    boxes = getBoxesForImg(img)
    
    for box in boxes:
        for i in range(4):
            box.points[i].x = box.points[i].x * scalar_w
            box.points[i].y = box.points[i].y * scalar_h
    return boxes


def timing(img: MatLike):
    # Give 1 start for processing
    getBoxesForImg(img)

    ITERATIONS = 500
    times = []
    from tqdm import tqdm
    
    for _ in tqdm(range(ITERATIONS)):
        start = time_ns()
        boxes = getBoxesForImg(img)
        merged = mergeListOfMatches(boxes)
        end = time_ns()
        times.append(end - start)
    avg_time = np.mean(times)
    print(f"Time taken: {(avg_time) / 1e6} ms")
    
    import matplotlib.pyplot as plt
    times = np.array(times) / 1e6
    plt.hist(times, bins=100)
    plt.savefig("histogram.png")


def main():
    
    input_file = "../test_image.jpg"

    img = cv2.imread(input_file)
    
    img = img[:416, :416]
    
    timing(img)

    start_time = time_ns()
    boxes = getBoxesForImg(img)
    end_time = time_ns()
    print("Found ", len(boxes), " boxes: \n")
    for box in boxes:
        print(box)

    merged = mergeListOfMatches(boxes)
    print("After merging: \n")
    for box in merged:
        print(box)

    img = putTextOnImage(img, merged)

    output_file = "../labelled_image.jpg"
    cv2.imwrite(output_file, img)
    

if __name__ == "__main__":
    main()
