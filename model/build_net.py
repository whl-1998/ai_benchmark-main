
from torch import nn
import torchvision.models as models
import model.models as customized_models
from torchsummary import summary
from args import args

# Models
default_model_names = sorted(name for name in models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
                                 if not name.startswith("__")
                                 and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names


def make_model(args):
    # 加载预训练模型
    model = models.alexnet()
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(
        nn.Linear(9216, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, args.num_classes)
    )
    
    return model

