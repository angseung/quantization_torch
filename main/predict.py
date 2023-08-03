import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import copy
import time

import torch

from models.squeezenet import *
from utils.quantization_utils import cal_mse

model = squeezenet1_0(quantize=True, is_qat=False)
model_fp = copy.deepcopy(model)
input = torch.randn(1, 3, 224, 224)
model(input)
torch.ao.quantization.convert(model, inplace=True)
start = time.time()
dummy_output = model(input)
end = time.time() - start
print(end)
start = time.time()
dummy_output_fp = model_fp(input)
end = time.time() - start
print(end)
nmse = cal_mse(dummy_output, dummy_output_fp, norm=True)
