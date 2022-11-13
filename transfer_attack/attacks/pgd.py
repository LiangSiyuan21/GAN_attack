from lib2to3.pgen2.token import LBRACE
import torch
import torch.nn as nn

from ..attack import Attack


class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=0.3,
                 alpha=2/255, steps=40, random_start=True, lb=0, ub=1):
        super().__init__("PGD", model)
        self.eps = eps * 255
        self.alpha = alpha * 255
        self.steps = steps
        self.random_start = random_start
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

        adv_images = images.clone().detach()
        
        new_data = {}
        new_data['img_metas'] = data['img_metas'][0]

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=self.lb, max=self.ub).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            data['img'][0] = adv_images
            
            new_data['img'] = adv_images

            losses = self.model(return_loss=True, **new_data, gt_bboxes=gts[0].unsqueeze(0), gt_labels=gts[1].unsqueeze(0))
            
            self.model.zero_grad()
            losses['loss_cls'] = losses['loss_cls'] * (-1.0)
            losses['loss_cls'].backward()
            grad = adv_images.grad.data

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=self.lb, max=self.ub).detach()

        return adv_images