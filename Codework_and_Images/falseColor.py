import cv2
import numpy

def Color(image):
    w, h = image.shape
    size = (w, h, 3)
    # iColor = cv2.CreateImage(size, 8, 3)
    iColor = numpy.zeros(size, dtype=float)
    for i in range(w):
        for j in range(h):
            r = GetR(image[i, j])
            g = GetG(image[i, j])
            b = GetB(image[i, j])
            iColor[i, j] = (r, g, b)

    iColor = iColor.astype(numpy.uint8)

    return iColor


def GetR(gray):
    if gray < 127:
        return 0
    elif gray > 191:
        return 255
    else:
        return (gray - 127) * 4 - 1


def GetG(gray):
    if gray < 64:
        return 4 * gray
    elif gray > 191:
        return 256 - (gray - 191) * 4
    else:
        return 255


def GetB(gray):
    if gray < 64:
        return 255
    elif gray > 127:
        return 0
    else:
        return 256 - (gray - 63) * 4


def FColor(image, array):
    w, h = image.shape
    size = (w, h, 3)
    iColor = numpy.zeros(size, dtype=float)
    for i in range(w):
        for j in range(h):
            iColor[i, j] = array[int(image[i, j] / 16)]

    iColor = iColor.astype(numpy.uint8)

    return iColor


FCArray = [(0, 51, 0), (0, 51, 102), (51, 51, 102), (51, 102, 51),
           (51, 51, 153), (102, 51, 102), (153, 153, 0), (51, 102, 153),
           (153, 102, 51), (153, 204, 102), (204, 153, 102), (102, 204, 102),
           (153, 204, 153), (204, 204, 102), (204, 255, 204), (255, 255, 204)]

img_name = "images/sjtu.png"  # 图像的路径
image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
iColor = Color(image)
iColor_image = cv2.cvtColor(iColor, cv2.COLOR_RGB2BGR)
iFColor = FColor(image, FCArray)
iFColor_image = cv2.cvtColor(iFColor, cv2.COLOR_RGB2BGR)
# cv2.imshow('image', image)
# cv2.imshow('iColor', iColor)
# cv2.imshow('iFColor', iFColor)
# cv2.waitKey(0)
cv2.imwrite("result_color/ori.jpg",image)
cv2.imwrite("result_color/Color1.jpg",iColor)
cv2.imwrite("result_color/Color2.jpg",iFColor)

