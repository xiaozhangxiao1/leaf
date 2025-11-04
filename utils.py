import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_images(images, titles, figsize=(15, 10)):
    """显示多张图片"""
    plt.figure(figsize=figsize)
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 3, i + 1)
        if len(image.shape) == 2:  # 灰度图
            plt.imshow(image, cmap='gray')
        else:  # 彩色图
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def save_result(image, filename, folder='results'):
    """保存结果图片"""
    import os
    if not os.path.exists(folder):
        os.makedirs(folder)
    cv2.imwrite(f'{folder}/{filename}', image)
    print(f'已保存: {folder}/{filename}')