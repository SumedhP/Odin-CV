import numpy as np
from dataclasses import dataclass
from typing import List

"""
Each output is shifted and scaled to line back up with the image.
This class handles the coversion of that output to the image space.
"""


@dataclass
class OffsetAndScale:
    x_offset: int
    y_offset: int
    scalar: int

    def __str__(self) -> str:
        return f"X Offset: {self.x_offset}, Y Offset: {self.y_offset}, Scalar: {self.scalar}"


"""
Code is based on the following:
https://github.com/HUSTLYRM/HUST_HeroAim_2024/blob/main/src/armor_detector/src/Inference.cpp#L61
"""


def generateOffsetsAndScalars() -> np.ndarray:
    INPUT_W = 416
    INPUT_H = 416
    SCALE = [8, 16, 32]
    output = []
    for scalar in SCALE:
        grid_h = INPUT_H // scalar
        grid_w = INPUT_W // scalar
        for y in range(grid_h):
            for x in range(grid_w):
                output.append(OffsetAndScale(x, y, scalar))
    output = np.array(output)
    return output


if __name__ == "__main__":
    grids = generateOffsetsAndScalars()
    print("Shape of grids:", grids.shape)
    print("First 5 grids:", grids[:5])
