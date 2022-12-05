from typing import Tuple, TypeVar

import numpy as np
import numpy.typing as npt

Img = npt.NDArray[np.uint8]
Mat = npt.NDArray[np.generic]
Point = Tuple[int, int]  # Not really!
Scalar = npt.ArrayLike  # TODO: 4D
Size = Tuple[int, int]  # Not really!
T = TypeVar("T", bound=np.floating)

class VideoCapture:
    def __init__(self, index: int) -> None: ...
    def read(self) -> Tuple[bool, Img]: ...
    def getBackendName(self) -> str: ...

class Node:
    def mat(self) -> Mat: ...

class FileStorage:
    def __init__(self, filename: str, flags: int, encoding: str = ...) -> None: ...
    def getNode(self, nodename: str) -> Node: ...

def bitwise_and(src1: Img, src2: Img, mask: Img = ...) -> Img: ...
def circle(
    src: Img,
    center: Point,
    radius: int,
    color: Scalar,
    thickness: int = ...,
    lineType: int = ...,
    shift: int = ...,
) -> Img: ...
def cvtColor(src: Img, code: int, dstCn: int = ...) -> Img: ...
def dilate(
    src: Img,
    kernel: Mat,
    anchor: Point = ...,
    iterations: int = ...,
    borderType: int = ...,
    borderValue: Scalar = ...,
) -> Img: ...
def erode(
    src: Img,
    kernel: Mat,
    anchor: Point = ...,
    iterations: int = ...,
    borderType: int = ...,
    borderValue: Scalar = ...,
) -> Img: ...
def imshow(winname: str, mat: Img): ...
def GaussianBlur(
    src: Img, ksize: Size, sigmaX: float, sigmaY: float = ..., borderType: int = ...
) -> Img: ...
def inRange(src: Img, lowerb: Mat, upperb: Mat) -> Img: ...
def minMaxLoc(src: Img) -> Tuple[float, float, Point, Point]: ...
def putText(
    img: Img,
    text: str,
    org: Point,
    fontFace: int,
    fontScale: float,
    color: Scalar,
    thickness: int = ...,
    lineType: int = ...,
    bottomLeftOrigin: bool = ...,
): ...
def Rodrigues(src: npt.NDArray[T]) -> Tuple[npt.NDArray[T], npt.NDArray[T]]: ...
def waitKey(delay: int = ...) -> int: ...

class BackgroundSubtractor:
    def apply(self, image: Img, learningRate: float = ...) -> Img: ...

class BackgroundSubtractorKNN(BackgroundSubtractor): ...

def createBackgroundSubtractorKNN(
    history: int = ..., varThreshold: float = ..., detectShadows: bool = ...
) -> BackgroundSubtractorKNN: ...

# Enums
COLOR_BGR2HSV: int
COLOR_BGR2GRAY: int

FILE_STORAGE_READ: int

FONT_HERSHEY_SIMPLEX: int

TERM_CRITERIA_EPS: int
TERM_CRITERIA_MAX_ITER: int
