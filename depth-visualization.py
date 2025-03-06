!pip install ipywidgets
from google.colab import output
output.enable_custom_widget_manager()

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

def process_and_visualize(pfm_path, png_path, target_value=0.5, tolerance=0.05):
    """
    PFMファイルとPNGファイルを読み込んで、深度マップの可視化を行う。

    Parameters:
    - pfm_path (str): PFMファイルのパス
    - png_path (str): PNGファイルのパス
    - target_value (float): ハイライトする深度値
    - tolerance (float): 許容範囲
    """
    # PFMファイルの読み込み
    def load_pfm(file):
        with open(file, 'rb') as f:
            header = f.readline().decode('latin-1').rstrip()
            color = True if header == 'PF' else False
            dims = f.readline().decode('latin-1').strip()
            width, height = map(int, dims.split())
            scale = float(f.readline().decode('latin-1').strip())
            endian = '<' if scale < 0 else '>'
            scale = abs(scale)
            data = np.fromfile(f, endian + 'f')
            shape = (height, width, 3) if color else (height, width)
            return np.reshape(data, shape), scale

    # PFMとPNGの読み込み
    depth_map, scale = load_pfm(pfm_path)
    depth_map = np.flipud(depth_map)
    png_image = Image.open(png_path)
    png_array = np.array(png_image)
    if png_array.shape[2] == 4:
      png_array = png_array[:, :, :3]

    #print("Depth Map Shape:", depth_map.shape)
    #print("PNG Shape:", png_array.shape)
    #print("Depth Map Min:", depth_map.min())
    #print("Depth Map Max:", depth_map.max())
    #print("Depth Map Mean:", depth_map.mean())
    #plt.hist(depth_map.ravel(), bins=50)
    #plt.title("Depth Map Value Distribution")
    #plt.xlabel("Depth Value")
    #plt.ylabel("Frequency")
    #plt.show()

    # 可視化関数
    def visualize_depth(target_value=0.5, tolerance=0.005):
        # 指定の深度値に近い部分をマスク
        mask = np.abs(depth_map - target_value) < tolerance
        print(f"Target Value: {target_value}, Tolerance: {tolerance}")
        print(f"Mask Coverage: {np.sum(mask) / mask.size * 100:.2f}%")
        # PNG画像に緑色を重ねる
        result_image = png_array.copy()
        result_image[mask, 0] = 0   # 赤チャンネルを0に
        result_image[mask, 1] = 255 # 緑チャンネルを最大に
        result_image[mask, 2] = 0   # 青チャンネルを0に

        # 結果を表示
        plt.figure(figsize=(15, 10))
        plt.imshow(result_image)
        plt.title(f"Target Depth: {target_value}±{tolerance}")
        plt.axis('off')
        plt.show()

        #plt.imshow(depth_map, cmap='viridis')
        #plasma
        #plt.colorbar()
        #plt.title("Depth Map")
        #plt.show()

    # スライダー付きのインターフェースを作成
    interact(
        visualize_depth,
        target_value=FloatSlider(min=0.0, max=1.0, step=0.01, value=0.5, description="Target Value"),
        tolerance=FloatSlider(min=0.001, max=0.03, step=0.001, value=0.005, description="Tolerance")
    )
