import torch


class Decoder(torch.nn.Module):
    # TODO: support learnable fusion modules
    def __init__(self):
        super().__init__()
        self.FUSION_DIC = {"2to1_fusion": ["sum", "diff", "abs_diff"],
                           "2to2_fusion": ["concat"]}

    def fusion(self, x1, x2, fusion_form="concat"):
        """Specify the form of feature fusion"""
        if fusion_form == "concat":
            x = torch.cat([x1, x2], dim=1)
        elif fusion_form == "sum":
            x = x1 + x2
        elif fusion_form == "diff":
            x = x2 - x1
        elif fusion_form == "abs_diff":
            x = torch.abs(x1 - x2)
        else:
            raise ValueError('the fusion form "{}" is not defined'.format(fusion_form))

        return x

    def aggregation_layer(self, fea1, fea2, fusion_form="concat", ignore_original_img=True):
        """aggregate features from siamese or non-siamese branches"""

        start_idx = 1 if ignore_original_img else 0
        aggregate_fea = [self.fusion(fea1[idx], fea2[idx], fusion_form)
                         for idx in range(start_idx, len(fea1))]

        return aggregate_fea
