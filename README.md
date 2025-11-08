# Leaf extraction

This project extracts leaves from white backgrounds and highlights their vein structures.

## Usage

1. Place your source images inside the `images/` directory.
2. (Optional) Adjust processing parameters by creating a JSON configuration file. An example is provided at `config.example.json`.
3. Run the script:

```bash
python main.py --config my_config.json --output results --no-preview
```

Omit `--config` to use the default parameters. Remove `--no-preview` if you would like matplotlib preview windows to appear (requires a display).

### Configuration reference

```json
{
  "background": {
    "lower_white": [0, 0, 200],
    "upper_white": [180, 55, 255],
    "close_kernel": 5,
    "open_kernel": 5
  },
  "vein": {
    "blur_size": 3,
    "sobel_ksize": 3,
    "threshold_value": 30,
    "morph_size": 2,
    "adaptive_block_size": null,
    "adaptive_c": 2,
    "overlay_color": [0, 255, 0],
    "auto_threshold": true,
    "target_white_ratio": 0.03,
    "ratio_tolerance": 0.01,
    "auto_threshold_min": 5,
    "auto_threshold_max": 120,
    "closing_size": null
  },
  "vein_variants": [
    {
      "name": "fine_details",
      "blur_size": 3,
      "sobel_ksize": 3,
      "threshold_value": 25,
      "morph_size": 1,
      "overlay_color": [255, 0, 0],
      "target_white_ratio": 0.05
    },
    {
      "name": "strong_noise_reduction",
      "blur_size": 7,
      "sobel_ksize": 5,
      "threshold_value": 45,
      "morph_size": 3,
      "overlay_color": [0, 128, 255],
      "closing_size": 3,
      "target_white_ratio": 0.015
    }
  ]
}
```

- `background` controls how the leaf is separated from a white background.
- `vein` defines the default vein-extraction parameters.
- `vein_variants` allows you to experiment with multiple parameter sets in one run. Each variant gets its own output subfolder inside the results directory.
- Enabling `auto_threshold` lets the script adaptively search for a threshold so that the highlighted vein pixels occupy roughly `target_white_ratio` of the masked leaf. `ratio_tolerance`, `auto_threshold_min`, and `auto_threshold_max` control the search range and convergence condition. Set `auto_threshold` to `false` to use a fixed `threshold_value` instead.
- `closing_size` can optionally fill small gaps in the vein map before the final opening step when the mask contains fragmented ridges.

The script saves the extracted leaves, masks, Sobel magnitude maps, binary vein maps, cleaned vein maps, color overlays, and a `summary.json` describing the chosen threshold for every image and variant.

### Dependencies

Install the required packages before running the script:

```bash
pip install opencv-python matplotlib
```
