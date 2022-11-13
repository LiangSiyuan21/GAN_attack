import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from attack import Attack


class TAIG(Attack):
    def __init__(self, model, eps=8/255, alpha=2/255, steps=20, decay=0.0, resize_rate=0.9, diversity_prob=0.5, random_start=False, ub=1, lb=0):
        super(TAIG, self).__init__("TAIG", model)
        self.eps = eps *(ub-lb)
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.ub = ub
        self.lb = lb

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)
        
        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]
            
        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x

    
    def compute_ig(self, inputs, label_inputs, model):
        baseline = np.zeros(inputs['img'].shape)
        scaled_inputs = [baseline + (float(i) / 20) * (inputs['img'].detach().cpu().numpy() - baseline) for i in
                        range(0, 20  + 1)]
        scaled_inputs = np.asarray(scaled_inputs)

        scaled_inputs = torch.from_numpy(scaled_inputs)
        scaled_inputs = scaled_inputs.to(dtype=torch.float).cuda()
        scaled_inputs.requires_grad_(True)
        
        
        for i in range(len(scaled_inputs)):
            inputs['img'] = scaled_inputs[i]
            loss = model(return_loss=True, **inputs, gt_bboxes=label_inputs[0].unsqueeze(0), gt_labels=label_inputs[1].unsqueeze(0))
                
            model.zero_grad()
            
            loss['loss_cls'] = loss['loss_cls'] * (-1.0)
            loss['loss_cls'].backward()
            # if i == 0:
            #     grads = scaled_inputs.grad.data
            # else:
            #     grads = torch.stack([grads, scaled_inputs[i].grad.data], dim=0)
        grads = scaled_inputs.grad.data
        avg_grads = torch.mean(grads, dim=0)
        delta_X = scaled_inputs[-1] - scaled_inputs[0]
        integrated_grad = delta_X * avg_grads
        IG = integrated_grad.cpu().detach().numpy()
        del integrated_grad,delta_X,avg_grads,grads,loss
        return IG


    def forward(self, data, gts):
        r"""
        Overridden.
        """
        if len(data) == 2:
            images = data['img'][0].data[0].clone().detach().to(self.device)
        else:
            images = torch.from_numpy(data).to(self.device)
        
        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(images).detach().to(self.device)

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        
        new_data = {}
        new_data['img_metas'] = data['img_metas'][0]
        for i in range(self.steps):
            steps = 20

            adv_images.requires_grad = True
            data['img'][0] = adv_images
            
            new_data['img'] = adv_images

            integrated_grad = self.compute_ig(new_data, gts, self.model)

            
            self.model.zero_grad()
            input_grad=torch.from_numpy(integrated_grad)
            input_grad=input_grad.cuda()

            # losses['loss_cls'] = losses['loss_cls'] * (-1.0)
            # losses['loss_cls'].backward()
            # grad = adv_images.grad.data
            
            # grad_norm = torch.norm(nn.Flatten()(grad), p=1, dim=1)
            # grad = grad / grad_norm.view([-1]+[1]*(len(grad.shape)-1))
            # grad = grad + momentum*self.decay
            # momentum = grad
            
            adv_images = adv_images.detach() - 3 * torch.sign(input_grad)
            # adv_images = adv_images.detach() - self.alpha * torch.sign(input_grad) * 255.0
            # adv_images = adv_images.detach() - self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=self.lb, max=self.ub).detach()

        return adv_images