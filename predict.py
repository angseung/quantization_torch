from utils.quantization_utils import QuantizableModel
import copy
import time
from models.densenet import *
from models.resnet import *
import torch

# model = densenet121(DenseNet121_Weights.DEFAULT).eval()
# model = densenet161(DenseNet161_Weights.DEFAULT).eval()
# model = densenet169(DenseNet169_Weights.DEFAULT).eval()
model = resnet152().eval()
model_fp = copy.deepcopy(model)
input = torch.randn(1, 3, 224, 224)
model.fuse_model()
model = QuantizableModel(model).prepare()
model(input)
torch.ao.quantization.convert(model, inplace=True)
start = time.time()
dummy_output = model(input)
end = time.time() - start
print(end)
dummy_output_fp = model_fp(input)
