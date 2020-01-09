from models.resnet import resnet50
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models.segmentation import fcn_resnet50
from .unet import Unet
import numpy as np

class ResnetFPN(nn.Module):
    def __init__(self, numClass):
        super(ResnetFPN, self).__init__()
        self.resnet_backbone = resnet50(True)
        self._up_kwargs = {'mode': 'bilinear'}
        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0) # Reduce channels
        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        # Smooth layers
        self.smooth1_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth4_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth1_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth2_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth3_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth4_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        # Classify layers
        self.classify = nn.Conv2d(128*4, numClass, kernel_size=3, stride=1, padding=1)

    def _concatenate(self, p5, p4, p3, p2):
        _, _, H, W = p2.size()
        p5 = F.interpolate(p5, size=(H, W), **self._up_kwargs)
        p4 = F.interpolate(p4, size=(H, W), **self._up_kwargs)
        p3 = F.interpolate(p3, size=(H, W), **self._up_kwargs)
        return torch.cat([p5, p4, p3, p2], dim=1)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.interpolate(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), **self._up_kwargs) + y

    def forward(self, image):
        
        c2, c3, c4, c5 = self.resnet_backbone(image)
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        
        # Smooth
        p5 = self.smooth1_1(p5)
        p4 = self.smooth2_1(p4)
        p3 = self.smooth3_1(p3)
        p2 = self.smooth4_1(p2)
        
        p5 = self.smooth1_2(p5)
        p4 = self.smooth2_2(p4)
        p3 = self.smooth3_2(p3)
        p2 = self.smooth4_2(p2)

        # Classify
        ps3 = self._concatenate(p5, p4, p3, p2)
        output = self.classify(ps3)

        return output
    
class FCNResnet50(nn.Module):
    def __init__(self, num_classes):
        super(FCNResnet50, self).__init__()
        self.net = fcn_resnet50(num_classes=num_classes)
        
    def forward(self, images):
        return self.net(images)['out']
Net = ResnetFPN
class LocalRefinement(nn.Module):
    """ Network refining the larger prediction with local information
    """
    
    def __init__(self, num_classes):
        super(LocalRefinement, self).__init__()
        self.backbone = Net(num_classes)
        self.refinement = nn.Conv2d(2 * num_classes, num_classes, kernel_size=3, stride=1, padding=1)
    
    def forward(self, images, previous_prediction):
        """ Predict patch and combine with previous prediction
        
        Parameters
        ----------
        images: tensor (batch, 3, h, w)
            patch images
        previous_prediction: tensor (batch, num_classes, h, w)
            previous prediction
        
        Returns
        -------
        tensor (batch, num_classes, h, w)
            refinement patch prediction
        """
        # Predict on patch
        patch_prediction = self.backbone(images)
        
        # Combine with previous prediction
        combine_prediction = torch.cat([patch_prediction, previous_prediction], dim=1)
        
        # Refine prediction
        refinement_prediction = self.refinement(combine_prediction)
        
        return refinement_prediction
    
class InspectorNet(nn.Module):
    """ Network combining between global and local context
    """
    
    def __init__(self, num_classes, num_scaling_level):
        super(InspectorNet, self).__init__()
        
        self.global_branch = Net(num_classes)
        self.num_scaling_level = num_scaling_level
        
        for i in range(num_scaling_level):
            self.add_module("local_branch_"+str(i), LocalRefinement(num_classes))
            
    def copy_weight(self, source_level, dest_level):
        if source_level == -1:
            source_state_dict = self.global_branch.state_dict()
            getattr(self, "local_branch_"+str(dest_level)).backbone.load_state_dict(source_state_dict)
        else:
            source_state_dict = getattr(self, "local_branch_"+str(source_level)).state_dict()
            getattr(self, "local_branch_"+str(dest_level)).load_state_dict(source_state_dict)
            
    def get_training_parameters(self, training_level, decay_rate=0, learning_rate=1e-3):
        """ Training parameters for optimize
        
        Parameters
        ----------
        training_level: int
            level of local branch to be mainly trained
        decay_rate: float
            the decay rate of learning rate after each branch
        learning_rate: float
            learning rate of the training branch
        
        Returns
        -------
        list of {'params': params to optimize, 'lr': learning rate of those params}
        """
        params = []
        current_lr = learning_rate
        for i in range(self.num_scaling_level - 1, -1, -1):
            local_branch = getattr(self, "local_branch_"+str(i))
            for p in local_branch.parameters():
                p.requires_grad = False
            if i > training_level:
                continue
            for p in local_branch.parameters():
                p.requires_grad = (current_lr > 0)
            if current_lr != 0:
                print("Optimize level", i)
                params.append({'params': local_branch.parameters(), 'lr': current_lr})
            current_lr *= decay_rate
        if current_lr != 0:
            print("Optimize global")
            params.append({'params': self.global_branch.parameters(), 'lr': current_lr})
        for p in self.global_branch.parameters():
            p.requires_grad = (current_lr > 0)
        return params
        
    def forward_global(self, images, **kargs):
        """ Forward global branch
        
        Parameters
        ----------
        images: tensor (b, 3, h, w)
            rescaled images to predict
            
        Returns
        -------
        tensor (b, num_classes, h, w)
            prediction tensor
        """
        return self.global_branch(images)
    
    def forward_local(self, patches, previous_prediction, level, **kargs):
        """ Forward local patch
        
        Parameters
        ----------
        patches: tensor (b, 3, h, w)
            patches to predict
        previous_prediction: tensor (b, 3, h, w)
            prediction of previous branch
        level: int
            level of local branch
            
        Returns
        -------
        tensor (b, num_classes, h, w)
            refined patch prediction
        """
        local_branch = getattr(self, "local_branch_"+str(level))
        return local_branch(patches, previous_prediction)
    
    def forward(self, mode, **kargs):
        if mode == "global":
            return self.forward_global(**kargs)
        else:
            return self.forward_local(**kargs)