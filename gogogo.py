from CustomFastVit import CustomFastVit
import torch
import open_clip2
import copy
import torch.nn as nn
from PIL import Image
import logging
# logging.basicConfig(level=logging.INFO)


class Apple2Featurizer(nn.Module):

    def __init__(self, model_type="MobileCLIP2-S0", pretrained="dfndr2b"):
        super().__init__()
        self.model, self.preprocess = open_clip2.create_model_from_pretrained(
            model_type, pretrained=pretrained)

        self.model.eval()
        self.model = self.reparameterize_model(self.model)
        self.patch_size = 32

    def forward(self, img, output_cls_token: bool = True):

        return self.model.visual(img,output_cls_token)

    def reparameterize_model(self, model: torch.nn.Module) -> nn.Module:
        """Method returns a model where a multi-branched structure
            used in training is re-parameterized into a single branch
            for inference.

        Args:
            model: MobileOne model in train mode.

        Returns:
            MobileOne model in inference mode.
        """
        # Avoid editing original graph
        model = copy.deepcopy(model)
        for module in model.modules():
            if hasattr(module, "reparameterize"):
                module.reparameterize()
        return model


if __name__ == "__main__":
    import torchvision.transforms as T
    from PIL import Image

    model_name = "MobileCLIP2-S0"
    model_kwargs = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    image = Image.open("bird_full.jpg")



    model = Apple2Featurizer(model_name).to(device)
    precess_img = model.preprocess(image).unsqueeze(0).to(device).half()
    results = model(precess_img)
    print(results[0].shape)
    for i in results[1]:
        print(i.shape)
