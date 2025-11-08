import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from utils import save_result, show_images


ResultTuple = Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]


@dataclass
class BackgroundParams:
    lower_white: Sequence[int] = (0, 0, 200)
    upper_white: Sequence[int] = (180, 55, 255)
    close_kernel: int = 5
    open_kernel: int = 5

    def clamp(self) -> "BackgroundParams":
        self.lower_white = tuple(max(0, min(255, int(v))) for v in self.lower_white)
        self.upper_white = tuple(max(0, min(255, int(v))) for v in self.upper_white)
        self.close_kernel = max(0, int(self.close_kernel))
        self.open_kernel = max(0, int(self.open_kernel))
        return self


@dataclass
class VeinParams:
    name: str = "default"
    blur_size: int = 3
    sobel_ksize: int = 3
    threshold_value: int = 30
    morph_size: int = 2
    adaptive_block_size: Optional[int] = None
    adaptive_c: int = 2
    overlay_color: Sequence[int] = (0, 255, 0)
    result_folder: Optional[str] = None
    auto_threshold: bool = True
    target_white_ratio: float = 0.03
    ratio_tolerance: float = 0.01
    auto_threshold_min: int = 5
    auto_threshold_max: int = 120
    auto_threshold_steps: int = 8
    closing_size: Optional[int] = None

    def clamp(self, index: int = 0) -> "VeinParams":
        self.blur_size = _ensure_odd(self.blur_size, minimum=3)
        self.sobel_ksize = _ensure_odd(self.sobel_ksize, minimum=3)
        self.morph_size = max(1, int(self.morph_size))
        self.threshold_value = max(0, min(255, int(self.threshold_value)))
        if self.adaptive_block_size is not None:
            self.adaptive_block_size = _ensure_odd(self.adaptive_block_size, minimum=3)
        self.adaptive_c = int(self.adaptive_c)
        self.overlay_color = tuple(int(x) for x in self.overlay_color)
        self.auto_threshold = bool(self.auto_threshold)
        self.target_white_ratio = float(max(0.0, min(1.0, self.target_white_ratio)))
        self.ratio_tolerance = float(max(0.0, min(0.5, self.ratio_tolerance)))
        self.auto_threshold_min = max(0, min(255, int(self.auto_threshold_min)))
        self.auto_threshold_max = max(self.auto_threshold_min, min(255, int(self.auto_threshold_max)))
        self.auto_threshold_steps = max(1, int(self.auto_threshold_steps))
        if self.closing_size is not None:
            self.closing_size = max(1, int(self.closing_size))
        if not self.name:
            self.name = f"variant_{index + 1}"
        return self

    @property
    def output_tag(self) -> str:
        return _slugify(self.name)


def _ensure_odd(value: int, minimum: int = 1) -> int:
    value = max(minimum, int(value))
    if value % 2 == 0:
        value += 1
    return value


def _slugify(value: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_")
    return slug or "variant"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leaf vein extraction with tunable parameters")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip displaying preview windows (useful in headless environments)",
    )
    return parser.parse_args()


def load_config(config_path: Optional[str]) -> Tuple[BackgroundParams, List[VeinParams]]:
    if not config_path:
        return BackgroundParams().clamp(), [VeinParams().clamp()]

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    background_data = data.get("background", {})
    background_params = BackgroundParams(**background_data).clamp()

    vein_base = data.get("vein", {})
    vein_variants_data: Optional[Sequence[Dict]] = data.get("vein_variants")

    if vein_variants_data:
        vein_params_list: List[VeinParams] = []
        for idx, variant in enumerate(vein_variants_data):
            merged = {**vein_base, **variant}
            vein_params_list.append(VeinParams(**merged).clamp(index=idx))
        return background_params, vein_params_list

    return background_params, [VeinParams(**vein_base).clamp()]


def get_image_files(folder_path: str) -> List[str]:
    """自动获取文件夹中的所有图片文件"""
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    image_files = []

    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)

    print(f"在 {folder_path} 中找到 {len(image_files)} 张图片: {image_files}")
    return sorted(image_files)


def extract_leaf_background(image_path: str, params: BackgroundParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """第一步：从白色背景中提取叶子"""
    print(f"正在处理: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"错误：无法读取图片 {image_path}，文件可能不存在或格式不支持")

    print(f"图片尺寸: {img.shape}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_white = np.array(params.lower_white, dtype=np.uint8)
    upper_white = np.array(params.upper_white, dtype=np.uint8)
    mask_bg = cv2.inRange(hsv, lower_white, upper_white)
    mask_leaf = cv2.bitwise_not(mask_bg)

    if params.close_kernel > 0:
        kernel_close = np.ones((params.close_kernel, params.close_kernel), np.uint8)
        mask_leaf = cv2.morphologyEx(mask_leaf, cv2.MORPH_CLOSE, kernel_close)

    if params.open_kernel > 0:
        kernel_open = np.ones((params.open_kernel, params.open_kernel), np.uint8)
        mask_leaf = cv2.morphologyEx(mask_leaf, cv2.MORPH_OPEN, kernel_open)

    result = cv2.bitwise_and(img, img, mask=mask_leaf)

    white_bg = np.ones_like(img) * 255
    white_bg = cv2.bitwise_and(white_bg, white_bg, mask=mask_leaf)
    result_white_bg = cv2.bitwise_or(result, white_bg)

    return result_white_bg, mask_leaf, img


def extract_veins(original_img: np.ndarray, mask: np.ndarray, params: VeinParams) -> ResultTuple:
    print(f"开始提取叶脉 ({params.name})...")

    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (params.blur_size, params.blur_size), 0)

    sobelx = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=params.sobel_ksize)
    sobely = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=params.sobel_ksize)
    sobel_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    max_val = np.max(sobel_magnitude)
    if max_val == 0:
        sobel_magnitude_normalized = np.zeros_like(gray, dtype=np.uint8)
    else:
        sobel_magnitude_normalized = np.uint8(255 * sobel_magnitude / max_val)

    sobel_magnitude_normalized = cv2.bitwise_and(sobel_magnitude_normalized, sobel_magnitude_normalized, mask=mask)

    source_for_threshold = sobel_magnitude_normalized
    adaptive_used = False
    if params.adaptive_block_size:
        adaptive_used = True
        source_for_threshold = cv2.adaptiveThreshold(
            sobel_magnitude_normalized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            params.adaptive_block_size,
            params.adaptive_c,
        )

    threshold_value = params.threshold_value
    auto_ratio = 0.0
    if params.auto_threshold:
        threshold_value, auto_ratio = _auto_select_threshold(
            source_for_threshold,
            mask,
            params,
            fallback=threshold_value,
        )

    _, veins_binary = cv2.threshold(
        source_for_threshold,
        threshold_value,
        255,
        cv2.THRESH_BINARY,
    )

    if params.closing_size:
        kernel_close = np.ones((params.closing_size, params.closing_size), np.uint8)
        veins_binary = cv2.morphologyEx(veins_binary, cv2.MORPH_CLOSE, kernel_close)

    kernel_open = np.ones((params.morph_size, params.morph_size), np.uint8)
    veins_cleaned = cv2.morphologyEx(veins_binary, cv2.MORPH_OPEN, kernel_open)

    debug_info: Dict[str, object] = {
        "threshold": float(threshold_value),
        "auto_ratio": float(auto_ratio),
        "auto_enabled": bool(params.auto_threshold),
        "adaptive_used": bool(adaptive_used),
    }

    return sobel_magnitude_normalized, veins_binary, veins_cleaned, debug_info


def _auto_select_threshold(
    source: np.ndarray,
    mask: np.ndarray,
    params: VeinParams,
    fallback: int,
) -> Tuple[int, float]:
    masked_values = source[mask > 0]
    if masked_values.size == 0:
        return fallback, 0.0

    candidate_low = params.auto_threshold_min
    candidate_high = params.auto_threshold_max
    target = params.target_white_ratio
    tolerance = params.ratio_tolerance

    best_threshold = int(np.clip(fallback, 0, 255))
    best_ratio = _compute_ratio(masked_values, best_threshold)
    best_score = abs(best_ratio - target)

    for _ in range(params.auto_threshold_steps):
        if candidate_low > candidate_high:
            break
        candidate = int(round((candidate_low + candidate_high) / 2))
        candidate = int(np.clip(candidate, 0, 255))
        ratio = _compute_ratio(masked_values, candidate)
        score = abs(ratio - target)

        if score < best_score:
            best_threshold = candidate
            best_ratio = ratio
            best_score = score

        if score <= tolerance:
            best_threshold = candidate
            best_ratio = ratio
            break

        if ratio > target and candidate < 255:
            candidate_low = candidate + 1
        else:
            candidate_high = candidate - 1

    return best_threshold, best_ratio


def _compute_ratio(values: np.ndarray, threshold: int) -> float:
    if values.size == 0:
        return 0.0
    ratio = float(np.count_nonzero(values >= threshold)) / float(values.size)
    return ratio


def create_veins_overlay(original_img: np.ndarray, veins_mask: np.ndarray, color: Sequence[int] = (0, 255, 0)) -> np.ndarray:
    """在原图上叠加叶脉"""
    overlay = original_img.copy()
    color_mask = np.zeros_like(overlay)
    color_mask[veins_mask > 0] = color
    overlay = cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0)
    return overlay


def ensure_output_dirs(root: str, vein_params_list: Sequence[VeinParams]) -> Dict[str, str]:
    os.makedirs(root, exist_ok=True)
    directories = {
        "leaves": os.path.join(root, "leaves"),
        "masks": os.path.join(root, "masks"),
    }
    for folder in directories.values():
        os.makedirs(folder, exist_ok=True)

    for params in vein_params_list:
        folder = params.result_folder or os.path.join(root, params.output_tag)
        os.makedirs(folder, exist_ok=True)
        directories[params.output_tag] = folder
        params.result_folder = folder

    return directories


def process_images(
    image_files: Sequence[str],
    background_params: BackgroundParams,
    vein_params_list: Sequence[VeinParams],
    output_root: str,
    show_preview: bool,
) -> None:
    all_extracted: List[np.ndarray] = []
    variant_results: Dict[str, List[np.ndarray]] = {p.output_tag: [] for p in vein_params_list}
    variant_overlays: Dict[str, List[np.ndarray]] = {p.output_tag: [] for p in vein_params_list}
    run_logs: List[Dict[str, object]] = []

    directories = ensure_output_dirs(output_root, vein_params_list)

    for idx, image_file in enumerate(image_files):
        image_path = os.path.join("images", image_file)
        print(f"\n--- 处理第 {idx + 1} 张图片: {image_file} ---")

        try:
            extracted_leaf, mask, original_img = extract_leaf_background(image_path, background_params)
        except FileNotFoundError as exc:
            print(exc)
            continue

        base_name = os.path.splitext(os.path.basename(image_file))[0]
        save_result(extracted_leaf, f"{base_name}_leaf.png", folder=directories["leaves"])
        save_result(mask, f"{base_name}_mask.png", folder=directories["masks"])

        all_extracted.append(extracted_leaf)
        for params in vein_params_list:
            sobel_mag, veins_binary, veins_cleaned, debug_info = extract_veins(original_img, mask, params)

            tag = params.output_tag
            save_result(sobel_mag, f"{base_name}_sobel.png", folder=params.result_folder)
            save_result(veins_binary, f"{base_name}_binary.png", folder=params.result_folder)
            save_result(veins_cleaned, f"{base_name}_cleaned.png", folder=params.result_folder)

            overlay = create_veins_overlay(original_img, veins_cleaned, color=params.overlay_color)
            save_result(overlay, f"{base_name}_overlay.png", folder=params.result_folder)

            variant_results[tag].append(veins_cleaned)
            variant_overlays[tag].append(overlay)
            enriched_info = {
                **debug_info,
                "image": image_file,
                "variant": params.name,
                "output_folder": params.result_folder,
            }
            run_logs.append(enriched_info)

            print(
                "    阈值: {threshold:.1f} | 叶脉像素占比: {auto_ratio:.3f}".format(
                    **enriched_info
                )
            )

        print(f"第 {idx + 1} 张图片处理完成 - 叶脉已提取")

    print(f"\n处理完成！共成功处理 {len(all_extracted)} 张图片")

    if run_logs:
        summary_path = os.path.join(output_root, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as fp:
            json.dump(run_logs, fp, ensure_ascii=False, indent=2)
        print(f"处理参数摘要已保存到: {summary_path}")

    if not show_preview or not all_extracted:
        return

    print("\n显示处理结果...")
    show_images(all_extracted, [f"Leaf {i + 1}" for i in range(len(all_extracted))])
    for tag, images in variant_results.items():
        if images:
            show_images(images, [f"Veins {tag} {i + 1}" for i in range(len(images))])
    for tag, overlays in variant_overlays.items():
        if overlays:
            show_images(overlays, [f"Overlay {tag} {i + 1}" for i in range(len(overlays))])

    print("\n所有结果已保存到输出目录！")


def main() -> None:
    print("开始叶子提取项目...")
    args = parse_args()

    background_params, vein_params_list = load_config(args.config)
    image_files = get_image_files("images")

    if not image_files:
        print("在 images 文件夹中没有找到任何图片！")
        return

    process_images(
        image_files=image_files,
        background_params=background_params,
        vein_params_list=vein_params_list,
        output_root=args.output,
        show_preview=not args.no_preview,
    )


if __name__ == "__main__":
    main()
