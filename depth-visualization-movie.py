import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter  # 変更箇所

def create_depth_video(pfm_path, png_path, tolerance, output_file="depth_video.mp4"):
    """
    PFMファイルとPNGファイルを読み込んで、target_valueを0~1まで変化させた動画を生成する。

    Parameters:
    - pfm_path (str): PFMファイルのパス
    - png_path (str): PNGファイルのパス
    - tolerance (float): 許容範囲
    - output_file (str): 出力動画ファイル名（MP4形式）
    """
    # PFMファイルの読み込み関数
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

    # フレームごとの画像を作成する関数
    def update_frame(target_value):
        mask = np.abs(depth_map - target_value) < tolerance
        result_image = png_array.copy()
        result_image[mask, 0] = 0   # 赤チャンネルを0に
        result_image[mask, 1] = 255 # 緑チャンネルを最大に
        result_image[mask, 2] = 0   # 青チャンネルを0に
        return result_image

    # アニメーションを作成
    fig, ax = plt.subplots(figsize=(24, 12))
    img = ax.imshow(png_array)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 余白を最小化
    def animate(frame):
        target_value = frame / 100  # target_valueを0~1に変化
        result_image = update_frame(target_value)
        img.set_data(result_image)
        ax.set_title(f"Target Depth: {target_value:.2f}±{tolerance}", fontsize=16)
        return [img]

    # アニメーション設定
    anim = FuncAnimation(fig, animate, frames=101, interval=100, blit=True)

    # 動画として保存（MP4形式）
    writer = FFMpegWriter(fps=7, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(output_file, writer=writer,dpi=300)
    plt.close()

    print(f"動画が保存されました: {output_file}")

# 実行例
#create_depth_video("result.pfm", "result.png", tolerance=0.005, output_file="depth_video.mp4")
