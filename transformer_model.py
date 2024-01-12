import torch
import timm
import torch.nn as nn
import copy    
from ml_decoder import MLDecoder
from positional_encodings.torch_encodings import PositionalEncoding2D, Summer


class transformer_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('convnext_small.fb_in22k_ft_in1k', num_classes=19, pretrained=True)#'convnext_base.fb_in22k_ft_in1k_384'
        self.model.head = nn.Identity()
        self.pos_encoding = Summer(PositionalEncoding2D(768))#Summer(PositionalEncoding2D(1024))# 
        self.head = MLDecoder(num_classes=19, initial_num_features=768)#MLDecoder(num_classes=19, initial_num_features=1024)# 

    def forward(self, x):
        x = self.model(x)
        x = self.pos_encoding(x)
        x = self.head(x)
        return x

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# test = transformer_model().to(device)
# # print(test)

# tens = torch.randn(1, 3, 256, 256).to(device)

# out = test(tens)

# print(out.shape)