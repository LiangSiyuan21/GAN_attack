from lib2to3.pgen2.token import LBRACE
import torch
import torch.nn as nn

from ..attack import Attack


class BIM(Attack):
    r"""
    BIM or iterative-FGSM in the paper 'Adversarial Examples in the Physical World'
    [https://arxiv.org/abs/1607.02533]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 4/255)
        alpha (float): step size. (Default: 1/255)
        steps (int): number of steps. (Default: 0)

    .. note:: If steps set to 0, steps will be automatically decided following the paper.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.BIM(model, eps=4/255, alpha=1/255, steps=0)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, model, eps=4/255, alpha=1/255, steps=0, lb=0, ub=1):
        super().__init__("BIM", model)
        self.eps = eps * 255
        self.alpha = alpha * 255
        if steps == 0:
            self.steps = int(min(eps*255 + 4, 1.25*eps*255))
        else:
            self.steps = steps
        self.supported_mode = ['default', 'targeted']
        self.lb = lb
        self.ub = ub

    def forward(self, data, gts):
        r"""
        Overridden.
        """
        images = data['img'][0].data[0].clone().detach().to(self.device)
        
        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(images).detach().to(self.device)

        ori_images = images.clone().detach()
        
        new_data = {}
        new_data['img_metas'] = data['img_metas'][0]

        # images = images.clone().detach().to(self.device)
        # labels = labels.clone().detach().to(self.device)

        # if self.targeted:
        #     target_labels = self.get_target_label(images, labels)

        # loss = nn.CrossEntropyLoss()

        # ori_images = images.clone().detach()

        for _ in range(self.steps):
            images.requires_grad = True
            data['img'][0] = images
            
            new_data['img'] = images

            losses = self.model(return_loss=True, **new_data, gt_bboxes=gts[0].unsqueeze(0), gt_labels=gts[1].unsqueeze(0))
            
            self.model.zero_grad()
            losses['loss_cls'] = losses['loss_cls'] * (-1.0)
            losses['loss_cls'].backward()
            grad = images.grad.data

            # Update adversarial images
            adv_images = images + self.alpha*grad.sign()
            a = torch.clamp(ori_images - self.eps, min=self.lb)
            b = (adv_images >= a).float()*adv_images \
                + (adv_images < a).float()*a
            c = (b > ori_images+self.eps).float()*(ori_images+self.eps) \
                + (b <= ori_images + self.eps).float()*b
            images = torch.clamp(c, max=self.ub).detach()

        return images