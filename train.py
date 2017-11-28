from fast_neural_style import network
from options import Options

opt = Options().parse()
model = network.SimpleModel(opt)
print(model)
