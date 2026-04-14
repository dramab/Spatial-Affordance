"""
深度图可视化工具

用法:
    python visualize_depth.py <深度图路径> [--colormap <颜色映射>] [--output <输出路径>]

示例:
    python visualize_depth.py data/hope/hope_video/scene_0000/0000_depth.png
    python visualize_depth.py data/hope/hope_video/scene_0000/0000_depth.png --colormap jet --output depth_vis.png
"""
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_depth(depth_path: str, colormap: str = "turbo", output_path: str = None):
    """
    可视化深度图

    Args:
        depth_path: 深度图文件路径 (16-bit PNG)
        colormap: 颜色映射名称 (turbo, jet, plasma, viridis, rainbow)
        output_path: 输出文件路径，默认为 None 时显示图像
    """
    # 读取深度图
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

    if depth is None:
        raise ValueError(f"无法读取深度图: {depth_path}")

    print(f"深度图尺寸: {depth.shape}")
    print(f"数据类型: {depth.dtype}")
    print(f"最小值: {depth.min()}")
    print(f"最大值: {depth.max()}")

    # 获取有效深度值（排除0）
    valid_depth = depth[depth > 0]
    if len(valid_depth) > 0:
        print(f"非零最小值: {valid_depth.min()}")
        print(f"非零最大值: {valid_depth.max()}")
        print(f"非零均值: {valid_depth.mean():.2f}")

    # 归一化深度值用于可视化
    # 使用0值作为无效深度，其他值归一化到0-255
    depth_vis = depth.astype(np.float32)

    # 只对有效深度进行归一化
    if len(valid_depth) > 0:
        min_val = valid_depth.min()
        max_val = valid_depth.max()
        depth_vis[depth > 0] = (depth_vis[depth > 0] - min_val) / (max_val - min_val) * 255

    depth_vis = depth_vis.astype(np.uint8)

    # 应用颜色映射
    cmap_dict = {
        "turbo": cv2.COLORMAP_TURBO,
        "jet": cv2.COLORMAP_JET,
        "plasma": cv2.COLORMAP_PLASMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "rainbow": cv2.COLORMAP_RAINBOW,
        "hot": cv2.COLORMAP_HOT,
        "cool": cv2.COLORMAP_COOL,
    }

    cmap = cmap_dict.get(colormap.lower(), cv2.COLORMAP_TURBO)
    depth_colored = cv2.applyColorMap(depth_vis, cmap)

    # 将无效深度（0值）显示为黑色
    depth_colored[depth == 0] = [0, 0, 0]

    # 创建对比图：原始深度图（灰度）vs 彩色深度图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原始灰度图
    axes[0].imshow(depth_vis, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Grayscale Depth')
    axes[0].axis('off')

    # 彩色深度图
    axes[1].imshow(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Colormap: {colormap}')
    axes[1].axis('off')

    # 直方图
    axes[2].hist(valid_depth, bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[2].set_title('Depth Histogram (valid values)')
    axes[2].set_xlabel('Depth value (mm)')
    axes[2].set_ylabel('Frequency')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存到: {output_path}")

        # 同时保存单独的彩色深度图
        color_only_path = str(Path(output_path).with_suffix('')) + f"_{colormap}.png"
        cv2.imwrite(color_only_path, depth_colored)
        print(f"彩色深度图已保存到: {color_only_path}")
    else:
        plt.show()

    plt.close()

    return depth_colored


def main():
    parser = argparse.ArgumentParser(description='深度图可视化工具')
    parser.add_argument('depth_path', help='深度图文件路径 (16-bit PNG)')
    parser.add_argument('--colormap', '-c', default='turbo',
                        choices=['turbo', 'jet', 'plasma', 'viridis', 'rainbow', 'hot', 'cool'],
                        help='颜色映射名称 (默认: turbo)')
    parser.add_argument('--output', '-o', help='输出文件路径')

    args = parser.parse_args()

    # 如果没有指定输出路径，使用默认路径
    if args.output is None:
        input_path = Path(args.depth_path)
        args.output = str(input_path.with_suffix('')) + '_vis.png'

    visualize_depth(args.depth_path, args.colormap, args.output)


if __name__ == '__main__':
    main()
