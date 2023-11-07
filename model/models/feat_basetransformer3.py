from .feat_basetransformer2 import FEATBaseTransformer2
import torch
import numpy as np
import torch.nn.functional as F


# BaseTransformer3 --> folds the prorotypes into batch dimension
# so attention between 1 prototype and its topk base instances

class FEATBaseTransformer3(FEATBaseTransformer2):
    def _forward(self, instance_embs, support_idx, query_idx, ids=None):
        raise NotImplementedError