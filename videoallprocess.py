import cv2
import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from core.archs.ir.ETDS.arch import ETDSForInference
from core.utils import imwrite, tensor2img
# from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip


# 图像切分成帧
def video_to_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # open video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
        
    cap.release()
    print(f"Total frames: {frame_count}")

# 将帧合成视频
def frames_to_video(input_folder, output_video_path, fps=30.00):
    frame_files = sorted(os.listdir(input_folder))  # 按顺序读取帧
    first_frame_path = os.path.join(input_folder, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    height, width, _ = first_frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for frame_file in frame_files:
        frame_path = os.path.join(input_folder, frame_file)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)
    
    video_writer.release()
    print(f"Video saved to {output_video_path}")

# 加载超分辨率模型
def load_model(model_load_path_ETDS, device='cpu'):
    """
    加载预训练的超分模型
    :param model_load_path_ETDS: 模型权重路径
    :param device: 使用的设备（'cuda' 或 'cpu'）
    :return: 加载好的模型
    """
    # 初始化模型（替换为你的模型定义）
    model = ETDSForInference(num_in_ch=3, num_out_ch=3, upscale=2, num_block=1, num_feat=32, num_residual_feat=3)
    model.load_state_dict(torch.load(model_load_path_ETDS, map_location=device))
    model.to(device)  # 确保模型在指定的设备上
    model.eval()  # 设置为评估模式

    print(f"模型加载成功：{model_load_path_ETDS}")
    return model

# 对单帧进行推理
def inference(model, img_path, output_path, device='cpu'):
    """
    输入图像到模型并输出超分图像
    :param model: 加载好的超分模型
    :param img_path: 输入低分辨率图像路径
    :param output_path: 保存超分辨率图像路径
    :param device: 使用的设备（'cuda' 或 'cpu'）
    """
    # 读取图像并转换为 Tensor
    img = Image.open(img_path).convert('RGB')  # 确保是 RGB 格式
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)  # [1, C, H, W]

    # 推理
    with torch.no_grad():
        output = model(img_tensor)  # 输入图像到模型进行超分辨率处理

    # 转换为图像格式并保存
    sr_img = tensor2img([output.squeeze().cpu()])  # 将输出移回 CPU
    imwrite(sr_img, output_path)
    print(f"超分辨率图像已保存：{output_path}")

# 处理视频的超分辨率流程
def process_video(video_path, output_video_path, model, device='cpu'):
    # 1. 视频切分成帧
    frames_folder_original = "frames_original"  # 原始图像存放文件夹
    frames_folder_super_res = "frames_super_res"  # 超分图像存放文件夹
    
    # 创建文件夹
    os.makedirs(frames_folder_original, exist_ok=True)
    os.makedirs(frames_folder_super_res, exist_ok=True)

    video_to_frames(video_path, frames_folder_original)

    # 2. 对每一帧进行超分辨率处理
    frame_files = sorted(os.listdir(frames_folder_original))
    for idx, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frames_folder_original, frame_file)
        output_frame_path = os.path.join(frames_folder_super_res, f"super_res_{frame_file}")
        
        # 执行推理
        inference(model, frame_path, output_frame_path, device=device)

    # 3. 将处理后的帧合成超分视频
    frames_to_video(frames_folder_super_res, output_video_path)
    print(f"超分辨率视频已保存：{output_video_path}")

    # # 4. 提取音频并合成视频
    # # 使用 moviepy 提取音频并合成超分视频
    # video_clip = VideoFileClip(video_path)
    # audio_clip = video_clip.audio
    # audio_clip.write_audiofile("audio.mp3")  # 保存音频

    # # 合成音频和视频
    # video_with_audio = VideoFileClip(output_video_path)
    # video_with_audio = video_with_audio.set_audio(AudioFileClip("audio.mp3"))
    # video_with_audio.write_videofile(output_video_path, codec='libx264', audio_codec='aac')
    # print(f"超分辨率视频与音频合成完毕：{output_video_path}")

if __name__ == '__main__':
    # 参数设置
    model_load_path_ETDS = '/workspace/ETDS/etdsmodel/etds_deletethree.pth'
    input_video_path = '/workspace/ETDS/14种旅行人像拍摄技巧_256x144.mp4'  # 替换为你的输入视频路径
    output_video_path = '/workspace/ETDS/14种旅行人像拍摄技巧_512x288.mp4'  # 替换为输出视频路径
    device = 'cpu'  # 使用 CPU 进行推理

    # 加载模型
    model = load_model(model_load_path_ETDS, device=device)

    # 进行视频处理
    process_video(input_video_path, output_video_path, model, device=device)
