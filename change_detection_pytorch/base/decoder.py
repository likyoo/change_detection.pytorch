import torch


class Decoder(torch.nn.Module):

    def fusion(self, x1, x2, fusion_form="concat"):
        """Specify the form of feature fusion"""
        if fusion_form == "concat":
            x = torch.cat([x1, x2], dim=1)
        elif fusion_form == "sum":
            x = x1 + x2
        elif fusion_form == "diff":
            x = torch.abs(x1 - x2)
        else:
            raise ValueError('the fusion form "{}" is not defined'.format(fusion_form))

        return x

    def aggregation_layer(self, fea1, fea2, fusion_from="concat", ignore_original_img=True):
        """aggregate features from siamese or non-siamese branches"""

        start_idx = 1 if ignore_original_img else 0
        aggregate_fea = [self.fusion(fea1[idx], fea2[idx], fusion_from)
                         for idx in range(start_idx, len(fea1))]

        return aggregate_fea
