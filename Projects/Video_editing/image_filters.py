import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
from google.colab.patches import cv2_imshow
def quantimage(image,k=10):
    i = np.float32(image).reshape(-1,3)
    condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,20,1.0)
    ret,label,center = cv2.kmeans(i, k , None, condition,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    final_img = center[label.flatten()]
    final_img = final_img.reshape(image.shape)
    return final_img
def cartoonify(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    gray = cv2.medianBlur(gray, 7) 
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 10)
    color = cv2.bilateralFilter(image, 12, 250, 250) 
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return quantimage(cartoon,10)

def img_filter(image,filter):
    if filter=="greyscale":
        return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    elif filter=="bright":
        return cv2.convertScaleAbs(image,beta=np.random.randint(-100,100)) # beta = +ve for brighter and -ve for darker image
    elif filter=='sharpen':
        kernel=np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
        return cv2.filter2D(image,-1,kernel)
    elif filter=='sepia':
        img_sepia = np.array(image, dtype=np.float64)
        img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131],
                                    [0.349, 0.686, 0.168],
                                    [0.393, 0.769, 0.189]]))
        img_sepia[np.where(img_sepia > 255)] = 255
        return np.array(img_sepia, dtype=np.uint8)
    elif filter=='pencil_sketch':
        sk_gray, sk_color = cv2.pencilSketch(image, sigma_s=60, sigma_r=0.07, shade_factor=0.1) #sigma_s=60, sigma_r=0.07, shade_factor=0.1
        return np.sk_gray, sk_color
    elif filter=='HDR':
        return cv2.detailEnhance(image, sigma_s=12, sigma_r=0.15)
    elif filter=='invert':
        return cv2.bitwise_not(image)
    elif filter=='sketch':
        k_size=25
        grey_img=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        invert_img=cv2.bitwise_not(grey_img)
        blur_img=cv2.GaussianBlur(invert_img, (k_size,k_size),0)
        invblur_img=cv2.bitwise_not(blur_img)
        return cv2.divide(grey_img,invblur_img, scale=256.0)
    elif filter=='cartoonify':
        return cartoonify(image)
filters=["greyscale","sharpen","sepia","pencil_sketch","HDR","sketch","cartoonify"]
