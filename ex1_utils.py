"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from email.mime import image
from typing import List
import cv2
import numpy as np
import  matplotlib.pyplot as plt 
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2
MAX_INT_8 = 255
YIQ_MAT = np.array([[0.299, 0.587, 0.114],
                    [0.59590059, -0.27455667, -0.32134392],
                    [0.21153661, -0.52273617, 0.31119955]])

def myID() -> np.int32:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 203265186


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    img = cv2.imread(filename)

    if representation == LOAD_GRAY_SCALE: 
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif  representation == LOAD_RGB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img.astype(float)/MAX_INT_8
 

def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """

    img = imReadAndConvert(filename, representation)*MAX_INT_8
    img = img.astype(int)

    if representation == LOAD_GRAY_SCALE:
        plt.imshow(img, cmap = 'gray')
    elif representation == LOAD_RGB:
        plt.imshow(img)
    plt.show()

def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    return imgRGB.dot(YIQ_MAT.T)

def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    rgb = imgYIQ.dot(np.linalg.inv(YIQ_MAT).T)

    # Values clipping
    return np.clip(rgb,0,255) 


def hsitogramEqualize(imgOrig: np.ndarray) ->tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    gray_img = imgOrig.copy()
    
    #check if imgOrig is not gray
    if len(imgOrig.shape) > 2: 
        YIQ_img = transformRGB2YIQ(imgOrig)
        gray_img = YIQ_img[:,:,0]

    #get image cumsum LUT
    gray_img *= MAX_INT_8
    gray_img_hist = np.histogram(gray_img, np.arange(MAX_INT_8 + 2))[0]
    gray_img_cumsum = np.cumsum(gray_img_hist)
    new_cumsum = np.ceil(MAX_INT_8*gray_img_cumsum/gray_img.size)
    equalized_img = np.zeros_like(gray_img)
    
    #equalize image according to LUT
    for old_color, new_color in enumerate(new_cumsum):
        equalized_img[gray_img.astype(int) == old_color] = new_color

    #get equalized histogrm and normlize back to [0,1] range
    equalized_img_hist = np.histogram(equalized_img, np.arange(MAX_INT_8 + 2))[0]
    equalized_img /= MAX_INT_8
    
    #for RGB image transforme back to RGB from YIQ
    if len(imgOrig.shape) > 2: 
        YIQ_img[:,:,0] = equalized_img
        equalized_img = transformYIQ2RGB(YIQ_img)
    
    return (equalized_img, gray_img_hist, equalized_img_hist)


def quantizeImage(imOrig: np.ndarray, nQuant:int, nIter:int) -> tuple[list[np.ndarray], list[float]]:
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
  
    gray_img = imOrig.copy()
    #get gray image
    if len(gray_img.shape) > 2: 
        YIQ_img = transformRGB2YIQ(gray_img)
        gray_img = YIQ_img[:,:,0]
        orig_img_modified = imOrig
    else: 
        orig_img_modified = (gray_img * MAX_INT_8)
    gray_img = (gray_img*MAX_INT_8)

    q_img_list,MSE_error_list = [],[]
    borders,q_values = init_borders_and_qvals_arrs(gray_img,nQuant)
    for iter in range(nIter):
        #preforms single update of k_means
        borders,q_values = get_kmeans(gray_img, nQuant, borders, q_values)
        #quantize image by recived borders and q_values
        tmp = get_quntizied_img(gray_img, borders, q_values)
        
        if len(imOrig.shape) > 2:
            tmp = get_yiq_image(YIQ_img, tmp)

        MSE_error_list.append(np.square(orig_img_modified - tmp).mean())
        q_img_list.append(tmp)
    
    return q_img_list, MSE_error_list

def get_yiq_image(yiq:np.ndarray, gray:np.ndarray):
    tmp_yiq = yiq.copy()
    tmp_yiq[:, :, 0] = gray / MAX_INT_8
    gray = transformYIQ2RGB(tmp_yiq)
    return gray

def init_borders_and_qvals_arrs(img:np.ndarray,nQuant: int)->tuple[np.ndarray,np.ndarray]:
    '''
        The function initiate borders and quants values array for later
        use in k_means algo. 
    '''

    borders = np.round(np.linspace(img.min(), img.max(),nQuant+1))
    np.append(borders,[img.max()])
    q_values = get_consequetive_mean(borders)

    return borders.astype(int), q_values.astype(int)

def get_consequetive_mean(arr:np.ndarray)->np.ndarray:
    '''
        The function returns array of mean valuea of every 
        two consequative elements in arr
    '''
    mean = (arr[:-1] + arr[1:])/2

    return mean.astype(int)

def get_kmeans(img:np.ndarray,nQuant: int, borders:np.ndarray, q_values:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    '''
        The function preforms single iteration of k_means finding of image.
        borders: array containing the initiated borders array. 
        q_values : array containign the initiated quants values array
        nQuant : num of q_values requierd.

        the function returns updated borders and q_values arrays. 
    '''
    img_histogram = np.histogram(img, np.arange(0,257))[0]
    for i in range(nQuant):
        q_values[i] = get_wighted_mean_val(img_histogram,borders[i],borders[i+1])
                    
    borders = np.zeros_like(borders)
    borders[1:-1] = get_consequetive_mean(q_values)
    borders[-1] =  img.max() + 1

    return borders, q_values
            
def get_wighted_mean_val(mat: np.ndarray, start_border:int, end_border:int)->int:
    '''
        The function preforms whighted mean for mat (which should be img histogram)
    '''
    slice_by_borders = np.s_[start_border:end_border] 
    sub_hist = mat[slice_by_borders]
    summed_wighted_sub_hist = sub_hist.dot(np.arange(start_border,end_border))

    return int(np.round(summed_wighted_sub_hist/sub_hist.sum()))

def get_quntizied_img(img:np.ndarray,borders:list, q_values:list)->np.ndarray: 
    '''
        the function quantizes image
    '''
    tmp_img = np.zeros_like(img)
    for i in range(len(borders)-1): 
        mask = (img >= borders[i]) * (img < borders[i+1])
        tmp_img[mask] = q_values[i]
    return tmp_img 
