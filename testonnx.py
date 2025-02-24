import torch
from torch import nn
# from core.models.ir.ir_model import IRModel
from core.archs.ir.ETDS.arch import ETDSForInference
from thop import profile
from torchsummary import summary

model_load_path_ETDS = '/workspace/ETDS/etdsmodel/etds_deletethree.pth'
model_save_path_ETDS = '/workspace/ETDS/etdsmodel/etds_deletethree_deleteclip02122110.onnx'

#ETDS_M4C32_X2
#加载模型
# model = torch.load(model_load_path_ETDS)
# opt = {
#     'train': {
#         'fixed_residual_model_iters': 10,  # 冻结残差参数的训练轮次
#         'interpolation_loss_weight': 0.5    # 插值损失的权重
#     },
#     'num_gpu': 1,  # 这里设置为 1，表示使用一个 GPU，或者设置为 0 表示使用 CPU
#     'is_train': False,
#     'dist': False,
#     'network_g':{
#         'type': 'ETDS',
#         'num_in_ch': 3,
#         'num_out_ch': 3,
#         'upscale': 2,
#         'num_block': 4,
#         'num_feat': 32,
#         'num_residual_feat': 3,
#     },
# }

model = ETDSForInference(num_in_ch=3, num_out_ch=3, upscale=2, num_block=1, num_feat=32, num_residual_feat=3)

model.load_state_dict(torch.load(model_load_path_ETDS, map_location=torch.device('cpu')))

# 检查是否有可用的 GPU，如果有就使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)  # 将模型移动到 GPU（如果有的话）

# 打印模型结构
print(model)

# 打印模型的详细信息（包括每层参数量）
summary(model, input_size=(3, 240, 240))  # 输入尺寸根据你的模型输入形状调整

# model.eval()
# 加载模型文件
# model = torch.load('/workspace/ETDS/experiments/ETDS_M4C32_x2/models/network_g_latest.pth')  # 'model.pth' 是你的模型文件路径

# 打印模型结构
print(model)
# 准备输入实例，根据模型的实际输入形状调整张量形式
dummy_input = torch.randn(1,3,240,240).to(device)#(batch_size, channels, height, weight)

# 计算FLOPs和参数量
flops, params = profile(model, inputs=(dummy_input,))

# 打印FLOPs和参数量
print(f"FLOPs: {flops}, Parameters: {params}")

# torch.onnx.export(model, (dummy_input), model_save_path_ETDS, verbose=True)
# print("模型转换成功!")
