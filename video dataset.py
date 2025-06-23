import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# 设置视频文件夹路径
video_folder = "/Users/yuqijin/PycharmProjects/AMLProject/project/project/videos/input"  # 修改为实际的视频文件夹路径
output_folder = "/Users/yuqijin/PycharmProjects/AMLProject/project/project/videos/output"  # 处理后数据的保存路径
os.makedirs(output_folder, exist_ok=True)

# 获取所有视频文件
video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path)

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算抽取的帧索引（均匀分布）
    frame_indices = np.linspace(0, frame_count - 1, 120, dtype=int)  # 取 120 帧
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            resized_frame = cv2.resize(frame, (150, 150))  # 缩放到 150x150
            frames.append(resized_frame)

    cap.release()

    if frames:
        video_frames = np.array(frames) / 255.0  # 归一化

        # 保存为 NumPy 数组
        output_path = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}.npy")
        np.save(output_path, video_frames)
        print(f"Processed and saved: {output_path}")

print("Batch processing completed!")



import numpy as np
import matplotlib.pyplot as plt

# 读取 .npy 文件
video_frames = np.load("/Users/yuqijin/PycharmProjects/AMLProject/project/project/videos/output/double_pendulum_1.npy")  # 替换成你实际的 .npy 文件路径

# 检查数据形状
print(video_frames.shape)  # (num_frames, height, width, channels)

# 选择第一帧并显示
print(video_frames)
plt.imshow((video_frames[0][:, :, ::-1]*255).astype(np.uint8))  # 转换回 uint8 类型，并将BGR->RGB
plt.axis("off")  # 关闭坐标轴
plt.show()