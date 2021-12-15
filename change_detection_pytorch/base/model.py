import torch
from . import initialization as init


class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def base_forward(self, x1, x2):
        """Sequentially pass `x1` `x2` trough model`s encoder, decoder and heads"""
        if self.siam_encoder:
            features = self.encoder(x1), self.encoder(x2)
        else:
            features = self.encoder(x1), self.encoder_non_siam(x2)

        decoder_output = self.decoder(*features)

        # TODO: features = self.fusion_policy(features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            raise AttributeError("`classification_head` is not supported now.")
            # labels = self.classification_head(features[-1])
            # return masks, labels

        return masks

    def forward(self, x1, x2):
        """Sequentially pass `x1` `x2` trough model`s encoder, decoder and heads"""
        return self.base_forward(x1, x2)

    def predict(self, x1, x2):
        """Inference method. Switch model to `eval` mode, call `.forward(x1, x2)` with `torch.no_grad()`

        Args:
            x1, x2: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x1, x2)

        return x
