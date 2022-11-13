import torch
import torch.nn as nn

from ..attack import Attack


class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.007)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=0.007)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=0.007, lb=0, ub=1):
        super().__init__("FGSM", model)
        self.eps = eps * 255
        self.supported_mode = ['default', 'targeted']
        self.lb=lb
        self.ub=ub
    # def forward(self, images, labels):
    #     r"""
    #     Overridden.
    #     """
    #     images = images.clone().detach().to(self.device)
    #     labels = labels.clone().detach().to(self.device)

    #     if self.targeted:
    #         target_labels = self.get_target_label(images, labels)

    #     loss = nn.CrossEntropyLoss()

    #     images.requires_grad = True
        # outputs = self.get_logits(images)

        # # Calculate loss
        # if self.targeted:
        #     cost = -loss(outputs, target_labels)
        # else:
        #     cost = loss(outputs, labels)

        # # Update adversarial images
        # grad = torch.autograd.grad(cost, images,
        #                            retain_graph=False, create_graph=False)[0]

        # adv_images = images + self.eps*grad.sign()
        # adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        # return adv_images
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
        self.steps = 1
        for i in range(self.steps):
            adv_images.requires_grad = True
            data['img'][0] = adv_images
            
            new_data['img'] = adv_images

            losses = self.model(return_loss=True, **new_data, gt_bboxes=gts[0].unsqueeze(0), gt_labels=gts[1].unsqueeze(0))
            
            self.model.zero_grad()
            losses['loss_cls'] = losses['loss_cls'] * (-1.0)
            losses['loss_cls'].backward()
            grad = adv_images.grad.data
            
            grad_norm = torch.norm(nn.Flatten()(grad), p=1, dim=1)
            grad = grad / grad_norm.view([-1]+[1]*(len(grad.shape)-1))
            grad = grad + momentum
            momentum = grad

            adv_images = adv_images.detach() - self.eps*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=self.lb, max=self.ub).detach()

        return adv_images