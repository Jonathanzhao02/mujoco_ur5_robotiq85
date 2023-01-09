import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from models.film_resnet import resnet18
import clip
import contextlib


class Controller(nn.Module):
    def __init__(self, num_traces=3, num_weight_points=11, embedding_size=128):
        super(Controller, self).__init__()

        self.layer1_1 = nn.Linear(embedding_size, 16 * 16 * 8)
        self.layer2 = nn.Linear(16 * 16 * 8, 16 * 16 * 4)
        self.layer3 = nn.Linear(16 * 16 * 4, 16 * 16 * 1)
        self.layer4 = nn.Linear(16 * 16 * 1, num_traces * num_weight_points)

        self.layer1_bn = nn.BatchNorm1d(16 * 16 * 4)
        self.layer2_bn = nn.BatchNorm1d(16 * 16 * 4)
        self.layer3_bn = nn.BatchNorm1d(16 * 16 * 4)

    def forward(self, goal_embedding):
        x = F.selu(self.layer1_1(goal_embedding))
        x = F.selu(self.layer2(x))
        x = F.selu(self.layer3(x))
        x = self.layer4(x)
        return x


class Interp1d(torch.autograd.Function):
    def __call__(self, x, y, xnew, out=None):
        return self.forward(x, y, xnew, out)

    def forward(ctx, x, y, xnew, out=None):
        """
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.
        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.
        """
        # making the vectors at least 2D
        is_flat = {}
        require_grad = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
            assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
                                        'at most 2-D.'
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1
            require_grad[name] = vec.requires_grad
            device = list(set(device + [str(vec.device)]))
        assert len(device) == 1, 'All parameters must be on the same device.'
        device = device[0]

        # Checking for the dimensions
        assert (v['x'].shape[1] == v['y'].shape[1]
                and (
                     v['x'].shape[0] == v['y'].shape[0]
                     or v['x'].shape[0] == 1
                     or v['y'].shape[0] == 1
                    )
                ), ("x and y must have the same number of columns, and either "
                    "the same number of row or one of them having only one "
                    "row.")

        reshaped_xnew = False
        if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
           and (v['xnew'].shape[0] > 1)):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v['xnew'].shape
            v['xnew'] = v['xnew'].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v['x'].shape[0], v['xnew'].shape[0])
        shape_ynew = (D, v['xnew'].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0]*shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v['xnew'].shape[0] == 1:
            v['xnew'] = v['xnew'].expand(v['x'].shape[0], -1)

        torch.searchsorted(v['x'].contiguous(),
                           v['xnew'].contiguous(), out=ind)

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        saved_inputs = []
        for name in ['x', 'y', 'xnew']:
            if require_grad[name]:
                enable_grad = True
                saved_inputs += [v[name]]
            else:
                saved_inputs += [None, ]
        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat['slopes'] = is_flat['x']
        # now we have found the indices of the neighbors, we start building the
        # output. Hence, we start also activating gradient tracking
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v['slopes'] = (
                    (v['y'][:, 1:]-v['y'][:, :-1])
                    /
                    (eps + (v['x'][:, 1:]-v['x'][:, :-1]))
                )

            # now build the linear interpolation
            ynew = sel('y') + sel('slopes')*(
                                    v['xnew'] - sel('x'))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)

        ctx.save_for_backward(ynew, *saved_inputs)
        return ynew

    @staticmethod
    def backward(ctx, grad_out):
        inputs = ctx.saved_tensors[1:]
        gradients = torch.autograd.grad(
                        ctx.saved_tensors[0],
                        [i for i in inputs if i is not None],
                        grad_out, retain_graph=True)
        result = [None, ] * 5
        pos = 0
        for index in range(len(inputs)):
            if inputs[index] is not None:
                result[index] = gradients[pos]
                pos += 1
        return (*result,)


class Backbone(nn.Module):
    def __init__(self, action_size=100, img_size=224, embedding_size=256, intermediate_size=256):
        super(Backbone, self).__init__()
        self.embedding_size = embedding_size

        # Action Pathway
        self.action_encoder = nn.Linear(action_size, embedding_size)

        # Visual Pathway
        self.visual_encoder = resnet18(embedding_size)
        self.visual_narrower = nn.Linear(512, embedding_size)

        # Task Pathway
        self.task_id_encoder, _ = clip.load("ViT-B/32")
        self.task_id_embedding_narrower = nn.Linear(512, embedding_size)

        self.discriminator = nn.Sequential(
            nn.Linear(embedding_size * 3, intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, 1),
        )
    
    def _action_pathway_(self, action):
        action_embedding = self.action_encoder(action)
        return action_embedding

    def _img_pathway_(self, img, action_embed):
        # Comprehensive Visual Encoder. img_embedding is the square token list
        img_embedding = self.visual_encoder(img, action_embed).squeeze()
        img_embedding = self.visual_narrower(img_embedding)
        img_embedding = F.relu(img_embedding)
        return img_embedding

    def _task_id_pathway_(self, lang):
        with torch.no_grad():
            task_embedding = self.task_id_encoder.encode_text(lang)
        task_embedding = task_embedding.float()
        task_embedding = self.task_id_embedding_narrower(task_embedding)
        return task_embedding

    def forward(self, img, sentence, action):
        action = torch.flatten(action, start_dim=1)
        # Convert channels-first to channels-last
        img = img.permute(0, 2, 3, 1)

        # Action Pathway
        action_embed = self._action_pathway_(action)

        # Image Pathway
        img_embed = self._img_pathway_(img, action_embed)

        # Task Pathway
        task_embed = self._task_id_pathway_(sentence)

        x = torch.cat((action_embed, img_embed, task_embed), dim=-1)

        return self.discriminator(x)
