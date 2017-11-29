from fast_neural_style import network
from options import Options

opt = Options().parse()
print(opt)
model = network.SimpleModel(opt)
print(model)
