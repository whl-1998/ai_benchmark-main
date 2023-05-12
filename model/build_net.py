
from torch import nn
import torchvision.models as models
from args import args

# Models
default_model_names = sorted(name for name in models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(models.__dict__[name]))

model_names = default_model_names


def make_model():
    # 加载预训练模型
    model = models.__dict__[args.arch]()

    in_features = None
    for c in model.classifier:
        if hasattr(c, 'in_features'):
            in_features = c.in_features
            break

    for param in model.parameters():
        param.requires_grad = False

    # todo
    assert in_features is not None, 'in_features error!'

    sequential = []
    print('please input block type: \n'
          '1. Linear\n'
          '2. Relu\n'
          '3. Dropout\n'
          '4. quit!')
    block_type = int(input())
    layer_num = 0
    while block_type != 4:
        if layer_num == 0 and block_type == 1:
            print('if this is the first layer of your classifier, \n'
                  'in_features is {} already set in model, please give the linear layer output features: ')
            sequential.append(nn.Linear(in_features, int(input())))
            layer_num += 1
        if block_type == 1:
            print('please give the linear layer in and output features: ')
            in_feat = int(input())
            out_feat = int(input())
            sequential.append(nn.Linear(in_feat, out_feat))
            layer_num += 1
        if block_type == 2:
            sequential.append(nn.ReLU())
            layer_num += 1
        if block_type == 4:
            print('please give the dropout layer probability: ')
            p = int(input())
            sequential.append(nn.Dropout(p))
            layer_num += 1

        block_type = int(input())

    model.classifier = nn.Sequential(*sequential)

    # model.classifier = nn.Sequential(
    #     nn.Linear(in_features, 256),
    #     nn.ReLU(),
    #     nn.Dropout(0.4),
    #     nn.Linear(256, args.num_classes)
    # )
    
    return model

