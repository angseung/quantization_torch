import copy
import time
import torch
from models.resnet import *
from utils.quantization_utils import cal_mse

model = resnet152(quantize=True, is_qat=False)
model_fp = copy.deepcopy(model)
input = torch.randn(1, 3, 224, 224)
model(input)
torch.ao.quantization.convert(model, inplace=True)
start = time.time()
dummy_output = model(input)
end = time.time() - start
print(end)
dummy_output_fp = model_fp(input)
nmse = cal_mse(dummy_output, dummy_output_fp, norm=True)
