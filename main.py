import cv2
import numpy as np
import os
from utils import show_images, save_result


def get_image_files(folder_path):
    """自动获取文件夹中的所有图片文件"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []

    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)

    print(f"在 {folder_path} 中找到 {len(image_files)} 张图片: {image_files}")
    return image_files


def extract_leaf_background(image_path):
    """第一步：从白色背景中提取叶子"""
    print(f"正在处理: {image_path}")

    # 1. 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取图片 {image_path}，文件可能不存在或格式不支持")
        return None, None, None

    print(f"图片尺寸: {img.shape}")

    # 2. 转换到HSV颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 3. 定义白色背景的范围
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 55, 255])

    # 4. 创建背景掩码
    mask_bg = cv2.inRange(hsv, lower_white, upper_white)

    # 5. 反转掩码得到叶子区域
    mask_leaf = cv2.bitwise_not(mask_bg)

    # 6. 形态学操作优化掩码
    kernel = np.ones((5, 5), np.uint8)
    mask_leaf = cv2.morphologyEx(mask_leaf, cv2.MORPH_CLOSE, kernel)  # 填充小洞
    mask_leaf = cv2.morphologyEx(mask_leaf, cv2.MORPH_OPEN, kernel)  # 去除噪点

    # 7. 应用掩码提取叶子
    result = cv2.bitwise_and(img, img, mask=mask_leaf)

    # 8. 创建白色背景的版本
    white_bg = np.ones_like(img) * 255
    white_bg = cv2.bitwise_and(white_bg, white_bg, mask=mask_leaf)
    result_white_bg = cv2.bitwise_or(result, white_bg)

    return result_white_bg, mask_leaf, img


def extract_veins(leaf_image, mask, original_img):
    """提取叶脉 - 使用Sobel算子"""
    print("开始提取叶脉...")

    # 方法1: 在原图上直接提取叶脉（效果更好）
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # 高斯模糊减少噪声
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Sobel边缘检测 - 分别计算x和y方向梯度
    sobelx = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)  # x方向
    sobely = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=3)  # y方向

    # 计算梯度幅值
    sobel_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    # 归一化到0-255
    sobel_magnitude_normalized = np.uint8(255 * sobel_magnitude / np.max(sobel_magnitude))

    # 只在叶子区域内
    sobel_magnitude_normalized = cv2.bitwise_and(sobel_magnitude_normalized, sobel_magnitude_normalized, mask=mask)

    # 方法2: 自适应阈值提取叶脉
    veins_adaptive = cv2.adaptiveThreshold(
        sobel_magnitude_normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # 方法3: 普通阈值
    _, veins_binary = cv2.threshold(sobel_magnitude_normalized, 30, 255, cv2.THRESH_BINARY)

    # 清理噪声
    kernel = np.ones((2, 2), np.uint8)
    veins_cleaned = cv2.morphologyEx(veins_binary, cv2.MORPH_OPEN, kernel)

    return sobel_magnitude_normalized, veins_binary, veins_cleaned


def create_veins_overlay(original_img, veins_mask, color=(0, 255, 0)):
    """在原图上叠加叶脉"""
    overlay = original_img.copy()
    veins_rgb = cv2.cvtColor(veins_mask, cv2.COLOR_GRAY2BGR)

    # 创建彩色掩码
    color_mask = np.zeros_like(overlay)
    color_mask[veins_mask > 0] = color

    # 叠加叶脉
    overlay = cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0)

    return overlay


def tune_vein_extraction(original_img, mask,
                         blur_size=3, sobel_ksize=3,
                         threshold_value=30, morph_size=2):
"""可调节参数的叶脉提取函数"""

    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # 1. 高斯模糊
    gray_blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    # 2. Sobel检测
    sobelx = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    sobely = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    sobel_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    sobel_magnitude_normalized = np.uint8(255 * sobel_magnitude / np.max(sobel_magnitude))
    sobel_magnitude_normalized = cv2.bitwise_and(sobel_magnitude_normalized, sobel_magnitude_normalized, mask=mask)

    # 3. 阈值处理
    _, veins_binary = cv2.threshold(sobel_magnitude_normalized, threshold_value, 255, cv2.THRESH_BINARY)

    # 4. 形态学清理
    kernel = np.ones((morph_size, morph_size), np.uint8)
    veins_cleaned = cv2.morphologyEx(veins_binary, cv2.MORPH_OPEN, kernel)

    return sobel_magnitude_normalized, veins_binary, veins_cleaned

def main():
    print("开始叶子提取项目...")

    # 自动获取所有图片文件
    image_files = get_image_files('images')

    if not image_files:
        print("在 images 文件夹中没有找到任何图片！")
        return

    all_results = []
    all_veins_results = []
    all_overlays = []

    for i, image_file in enumerate(image_files):
        print(f"\n--- 处理第 {i + 1} 张图片: {image_file} ---")

        # 提取叶子主体
        extracted_leaf, mask, original_img = extract_leaf_background(f'images/{image_file}')

        if extracted_leaf is not None:
            # 保存叶子提取结果
            save_result(extracted_leaf, f'extracted_leaf_{i + 1}.png')
            save_result(mask, f'leaf_mask_{i + 1}.png')

            # 提取叶脉
            sobel_mag, veins_binary, veins_cleaned = extract_veins(extracted_leaf, mask, original_img)

            # 保存叶脉结果
            save_result(sobel_mag, f'sobel_magnitude_{i + 1}.png')
            save_result(veins_binary, f'veins_binary_{i + 1}.png')
            save_result(veins_cleaned, f'veins_cleaned_{i + 1}.png')

            # 创建叶脉叠加图
            veins_overlay = create_veins_overlay(original_img, veins_cleaned)
            save_result(veins_overlay, f'veins_overlay_{i + 1}.png')

            all_results.append(extracted_leaf)
            all_veins_results.append(veins_cleaned)
            all_overlays.append(veins_overlay)

            print(f"第 {i + 1} 张图片处理完成 - 叶脉已提取")
        else:
            print(f"第 {i + 1} 张图片处理失败")

    print(f"\n处理完成！共成功处理 {len(all_results)} 张图片")

    # 显示所有结果
    if all_results:
        print("\n显示处理结果...")

        # 显示提取的叶子
        show_images(all_results, [f'Leaf {i + 1}' for i in range(len(all_results))])

        # 显示叶脉
        show_images(all_veins_results, [f'Veins {i + 1}' for i in range(len(all_veins_results))])

        # 显示叠加图
        show_images(all_overlays, [f'Overlay {i + 1}' for i in range(len(all_overlays))])

        print("\n所有结果已保存到 results 文件夹！")
        print("生成的文件包括：")
        print("- extracted_leaf_X.png: 提取的叶子")
        print("- leaf_mask_X.png: 叶子掩码")
        print("- sobel_magnitude_X.png: Sobel梯度图")
        print("- veins_binary_X.png: 二值化叶脉")
        print("- veins_cleaned_X.png: 清理后的叶脉")
        print("- veins_overlay_X.png: 叶脉叠加在原图上的效果")


if __name__ == "__main__":
    main()