import torch
from Models.models import R34, NSA_UNet_R34, UNet_R34
from ptflops import get_model_complexity_info

nsa = NSA_UNet_R34(num_classes=4, pretrained=False)
unet = UNet_R34(num_classes=4, pretrained=False)
classifier = R34(num_classes=2, pretrained=False)
networks = [nsa, unet, classifier]

for name, net, size in zip(["NSA", "UNet", "Classifier"], networks, [768, 768, 256]):
    flops, params = get_model_complexity_info(
        net, (3, size, size), as_strings=False, print_per_layer_stat=False
    )
    print("Computing parameters for: {}".format(name))
    print("{:}  {:.5E}".format("Computational complexity: ", flops))
    print("{:}  {:.5E}".format("Number of parameters: ", params))
