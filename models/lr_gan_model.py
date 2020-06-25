import torch
import itertools
from .base_model import BaseModel
from . import networks
import numpy as np
import torch.nn as nn
import os
import torch.nn.functional as F
import torchvision
import random 
random.seed()
import time

class LRGANModel(BaseModel):

    def initialize(self, opt):
        self.opt = opt
        self.usecudnn = opt["general"]["usecudnn"]
        self.batchsize = opt["input"]["batchsize"]
        self.statsfrequency = opt["training"]["statsfrequency"]       
        self.learningrate = opt["training"]["learningrate"]
        self.modelType = opt["training"]["learningrate"]
        self.weightdecay = opt["training"]["weightdecay"]
        self.momentum = opt["training"]["momentum"]
 
        ###### Models
        self.Encoder_fab = networks.Encoder_fab(opt)
        self.FAB = networks.FAB(1, inner_nc=256, num_masks=0, num_additional_ids=32)

        self.Encoder_W_real = networks.Encoder_W_real(opt)
        self.Encoder_W_df =  networks.Encoder_W_df(opt)

        self.Classifier_W_real = networks.Classifier_W_GRU(opt)
        self.Classifier_W_df = networks.Classifier_W_GRU(opt)
        self.FAB.parallel_init()
        
        self.model_names = [ 
            'FAB', 
            'Encoder_fab',
            'Encoder_W_df', 
            'Classifier_W_df',
            'Encoder_W_real', 
            'Classifier_W_real',    
            ]
        self.loss_names = [
            'LR_df',
            'LR_real',
            # 'LR_fusion_sf',
            # 'LR_fusion_fc',
            'KD'
            # 'recon_1',
            # 'recon_recy',
            # 'recon_s2'
            ]

        self.Encoder_fab = nn.DataParallel(self.Encoder_fab).cuda()

        self.Encoder_W_df = nn.DataParallel(self.Encoder_W_df).cuda()
        self.Encoder_W_real = nn.DataParallel(self.Encoder_W_real).cuda()

        self.Classifier_W_real = nn.DataParallel(self.Classifier_W_real).cuda()
        self.Classifier_W_df = nn.DataParallel(self.Classifier_W_df).cuda()

        self.weights_dir = 'weights'
        self.weights_dir_out = 'weights'
        if not os.path.isdir(self.weights_dir):
            os.makedirs(self.weights_dir)
        if not os.path.isdir(self.weights_dir_out):
            os.makedirs(self.weights_dir_out)

        if(opt["general"]["loadpretrainedmodel"]):
            self.weights_suffix = opt["general"]["pretrainedmodelpath"]
            self.load_networks(self.weights_suffix)
        else:
            self.weights_suffix = ''
        
        ####  Loss
        self.criterion_LR = nn.CrossEntropyLoss()
        self.criterion_NLL = nn.NLLLoss().cuda()
        self.criterion_CE = nn.CrossEntropyLoss().cuda()
        self.kl_dist = nn.KLDivLoss().cuda()   
        self.criterion_recon  = nn.L1Loss().cuda()

    def load_networks(self, suffix):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_%s.pt' % (name, suffix)
                load_path = os.path.join(self.weights_dir, load_filename)
                net = getattr(self, name)
                print('loading the model from %s' % load_path)
                load_dict = torch.load(load_path)
                net_dict = net.state_dict()
                load_dict = {k: v for k, v in load_dict.items() if k in net_dict.keys() and v.size() == net_dict[k].size()}
                missed_params = [k for k, v in net_dict.items() if not k in load_dict.keys()]
                print('miss matched params:{}'.format(missed_params))
                net_dict.update(load_dict)
                net.load_state_dict(net_dict)   
    
    def save_networks(self, suffix):
        time.sleep(10)
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_%s.pt' % (name, suffix)
                save_path = os.path.join(self.weights_dir_out, save_filename)
                net = getattr(self, name)
                torch.save(net.state_dict(), save_path)

    def set_train(self):
        for name in self.model_names:
            if isinstance(name,str):
                net = getattr(self, name)
                net.train()
    
    def set_eval(self):
        for name in self.model_names:
            if isinstance(name,str):
                net = getattr(self, name)
                net.eval()

    def set_input(self, input):
        self.input = input['x'].cuda()
        self.labels = input['label'].cuda()

    def sample_source(self, source_img, sampler):
        batch_size = source_img.size(0)
        xs = np.linspace(-1,1,sampler.size(2))
        xs = np.meshgrid(xs, xs)
        xs = np.stack(xs, 2)
        xs = torch.tensor(xs, requires_grad=False).unsqueeze(0).repeat(batch_size, 1,1,1).cuda()
        xs  = xs.float()
        sampler_xs = (sampler.permute(0,2,3,1) + xs).clamp(min=-1,max=1)
        output_img = F.grid_sample(source_img.detach(), sampler_xs)
        return output_img

    def set_fab_input(self, input):
        self.source_img = input['source_img'].cuda()
        self.target_img =  input['target_img'].cuda()
    
    def train_DFN(self):
        params = [
            {"params": self.Encoder_fab.parameters(), "lr": opt["gan"]["lr"]},
            {"params": self.FAB.parameters(), "lr": opt["gan"]["lr"] }
        ]
        self.optimizer_ALL = torch.optim.Adam(params,
                                                 betas=(opt["gan"]["beta1"], 0.999))

        self.optimizer_ALL.zero_grad()
        batch_size = self.input.size(0)
        source_f = random.randint(0, 22)
        self.source_img = self.input[:, source_f, :, :, :]
        self.target_img =  self.input[:, source_f + 3, :, :, :]
        self.input_pair = torch.cat([self.source_img.unsqueeze(1), self.target_img.unsqueeze(1)], 1)
        ew = self.Encoder_fab(self.input_pair)
        ew_source = ew[:, 0, :]
        ew_target = ew[:, 1, :]

        xc_1 = torch.cat([ew_source, ew_target], 1).view(batch_size, 512, 1, 1)
        sampler_1 = self.FAB.decoder(xc_1) 
        self.output_img_1 = self.sample_source(self.source_img, sampler_1 )
        self.loss_recon_1 = self.criterion_recon(self.output_img_1, self.target_img)

        self.loss_ALL = self.loss_recon_1
        self.loss_ALL.backward()
        print('recon_1: {:.5f}'.format(self.loss_recon_1.data))
        self.optimizer_ALL.step()

    def train_fusion(self):
        params = [
            {"params": self.Encoder_fab.parameters(), "lr": opt["gan"]["lr"]},
            # {"params": self.FAB.parameters(), "lr": opt["gan"]["lr"] },

            {"params": self.Encoder_W_real.parameters(), "lr": opt["gan"]["lr"]},
            {"params": self.Classifier_W_real.parameters(), "lr": opt["gan"]["lr"]*0.1 },

            {"params": self.Encoder_W_df.parameters(), "lr": opt["gan"]["lr"]},
            {"params": self.Classifier_W_df.parameters(), "lr": opt["gan"]["lr"] *0.1}
        ]
        self.optimizer_ALL = torch.optim.Adam(params,
                                                 betas=(opt["gan"]["beta1"], 0.999))
        self.set_requires_grad(self.FAB, False)

        self.optimizer_ALL.zero_grad()
        batch_size = self.input.size(0)
        ew = self.Encoder_fab(self.input)
        ff_ew = torch.cat([ew[:, 1:29, :], ew[:, 28:29, :] ], 1)
        xc =  torch.cat([ff_ew, ew],2).view(-1, 512, 1, 1)
        samplers = self.FAB.decoder(xc)#.detach()
        samplers = samplers.view(batch_size, 29, 2, 88, 88)
        ew_df= self.Encoder_W_df(samplers)
        sf_df, fc_df, ft_df = self.Classifier_W_df(ew_df)
        ew_real = self.Encoder_W_real(self.input)
        sf_real, fc_real, ft_real = self.Classifier_W_real(ew_real)

        T = 20.
        self.loss_KD = \
                                    self.kl_dist(  (fc_df/T).log_softmax(-1),  (fc_real.detach()/T).softmax(-1)  ) * T * T  \
                                 +  self.kl_dist(  (fc_real/T).log_softmax(-1),  (fc_df.detach()/T).softmax(-1)  ) * T * T
        print('KD: {:.5f}'.format(self.loss_KD.data),end = ' ')

        self.loss_LR_df = self.criterion_NLL(sf_df.log(), self.labels.squeeze(1))
        self.loss_LR_real = self.criterion_NLL(sf_real.log(), self.labels.squeeze(1))

        sf_fusion = (sf_df*sf_real)
        self.loss_LR_fusion_sf = self.criterion_NLL(sf_fusion, self.labels.squeeze(1))

        fc_fusion = (fc_df + fc_real) /2.
        self.loss_LR_fusion_fc = self.criterion_CE(fc_fusion, self.labels.squeeze(1))

        print('LR_fusion_fc: {:.5f}'.format(self.loss_LR_fusion_fc.data),end = ' ')
        print('LR_fusion_sf: {:.5f}'.format(self.loss_LR_fusion_sf.data),end = ' ')
        print('LR_df: {:.5f}'.format(self.loss_LR_df.data),end = ' ')
        print('LR_real: {:.5f}'.format(self.loss_LR_real.data))

        self.loss_ALL =  self.loss_LR_df + self.loss_LR_real + self.loss_KD * 10
        self.loss_ALL.backward()
        self.optimizer_ALL.step()

    def validate_fusion(self):
        batch_size = self.input.size(0)
        ew = self.Encoder_fab(self.input)
        ff_ew = torch.cat([ew[:, 1:29, :], ew[:, 28:29, :] ], 1)
        xc =  torch.cat([ff_ew, ew],2).view(-1, 512, 1, 1)
        samplers = self.FAB.decoder(xc)#.detach()
        samplers = samplers.view(batch_size, 29, 2, 96, 96)
        ew_df= self.Encoder_W_df(samplers)
        sf_df, fc_df, ft_df = self.Classifier_W_df(ew_df)
        ew_real = self.Encoder_W_real(self.input)
        sf_real, fc_real, ft_real = self.Classifier_W_real(ew_real)
        sf_fusion = sf_df * sf_real
        fc_fusion = (sf_df + sf_real) /2.
        output = [sf_fusion, sf_real, sf_df,fc_fusion ]
        loss = [0, 0, 0, 0]
        cnt = [0, 0, 0, 0]
        for k in range(len(output)):
            loss[k] = self.criterion_NLL(output[k].log(), self.labels.squeeze(1)).detach().cpu().numpy()
            maxvalues, maxindices = torch.max(output[k].data, 1)
            cnt[k] = 0
            for i in range(0, self.labels.squeeze(1).size(0)):
                if maxindices[i] == self.labels.squeeze(1)[i]:
                    cnt[k] += 1
        return np.array(cnt), np.array(loss)
    
    def train_baseline(self):
        params = [
            {"params": self.Encoder_W_real.parameters(), "lr": opt["gan"]["lr"]},
            {"params": self.Classifier_W_real.parameters(), "lr": opt["gan"]["lr"]*0.1 },
        ]
        self.optimizer_ALL = torch.optim.Adam(params,
                                                 betas=(opt["gan"]["beta1"], 0.999))

        self.optimizer_ALL.zero_grad()
        batch_size = self.input.size(0)
        ew_real = self.Encoder_W_real(self.input)
        sf_real, fc_real = self.Classifier_W_real(ew_real)
        self.loss_LR_real = self.criterion_NLL(sf_real.log(), self.labels.squeeze(1))
        print('LR_real: {:.5f}'.format(self.loss_LR_real.data))
        self.loss_ALL = self.loss_LR_real
        self.loss_ALL.backward()
        self.optimizer_ALL.step()
    
    def validate_baseline(self):
        batch_size = self.input.size(0)
        ew_real = self.Encoder_W_real(self.input)
        sf_real, fc_real = self.Classifier_W_real(ew_real)
        output = sf_real
        loss = self.criterion_NLL(output.log(), self.labels.squeeze(1)).detach().cpu().numpy()
        maxvalues, maxindices = torch.max(output.data, 1)
        cnt = 0
        for i in range(0, self.labels.squeeze(1).size(0)):
            if maxindices[i] == self.labels.squeeze(1)[i]:
                cnt += 1
        return cnt, loss

    def train_df(self):
        params = [
            # {"params": self.Encoder_fab.parameters(), "lr": opt["gan"]["lr"]},
            {"params": self.Encoder_W_df.parameters(), "lr": opt["gan"]["lr"]},
            {"params": self.Classifier_W_df.parameters(), "lr": opt["gan"]["lr"] *0.1}
        ]
        self.optimizer_ALL = torch.optim.Adam(params,
                                                 betas=(opt["gan"]["beta1"], 0.999))
        self.set_requires_grad(self.FAB, False)

        self.optimizer_ALL.zero_grad()
        batch_size = self.input.size(0)
        ew = self.Encoder_fab(self.input)

        ff_ew = torch.cat([ew[:, 1:29, :], ew[:, 28:29, :] ], 1)
        xc =  torch.cat([ff_ew, ew],2).view(-1, 512, 1, 1)
        samplers = self.FAB.decoder(xc).detach()
        samplers = samplers.view(batch_size, 29, 2, 96, 96)

        def flow2rgb(flow_map, max_value= None):
            flow_map_np = flow_map.detach().cpu().numpy()
            _, h, w = flow_map_np.shape
            flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
            rgb_map = np.ones((3,h,w)).astype(np.float32)
            if max_value is not None:
                normalized_flow_map = flow_map_np / max_value
            else:
                normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
            rgb_map[0] += normalized_flow_map[0]
            rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
            rgb_map[2] += normalized_flow_map[1]
            return torch.from_numpy(rgb_map.clip(0,1)).cuda() 
        # self.flow2rgb = flow2rgb(sampler_1[0, :, :, :])#.unsqueeze(0)
        self.flow_frame = flow2rgb(samplers[0,0, :, :, :]).unsqueeze(0)
        for i in range(28):
            self.flow_frame = torch.cat([self.flow_frame, flow2rgb(samplers[0,i+1, :, :, :]).unsqueeze(0) ], 0)

        ew_df= self.Encoder_W_df(samplers)
        sf_df, fc_df = self.Classifier_W_df(ew_df)
        self.loss_LR_df = self.criterion_NLL(sf_df.log(), self.labels.squeeze(1))
        print('LR_df: {:.5f}'.format(self.loss_LR_df.data))
        self.loss_ALL =self.loss_LR_df
        self.loss_ALL.backward()
        self.optimizer_ALL.step()

    def validate_df(self):
        batch_size = self.input.size(0)
        ew = self.Encoder_fab(self.input)
        ff_ew = torch.cat([ew[:, 1:29, :], ew[:, 28:29, :] ], 1)
        xc =  torch.cat([ff_ew, ew],2).view(-1, 512, 1, 1)
        samplers = self.FAB.decoder(xc).detach()
        samplers = samplers.view(batch_size, 29, 2, 96, 96)
        ew_df= self.Encoder_W_df(samplers)
        sf_df, fc_df = self.Classifier_W_df(ew_df)
        output = fc_df
        loss = self.criterion_LR(output, self.labels.squeeze(1)).detach().cpu().numpy()
        maxvalues, maxindices = torch.max(output.data, 1)
        cnt = 0
        for i in range(0, self.labels.squeeze(1).size(0)):
            if maxindices[i] == self.labels.squeeze(1)[i]:
                cnt += 1
        return cnt, loss

    def visual_DFN(self, writer, iteration):
        img_source = self.source_img[0:1,:,:,:]
        img_target = self.target_img[0:1,:,:,:]
        img_output = self.output_img_1[0:1,:,:,:]

        writer.add_image('Image_train/{}_source'.format(iteration), torchvision.utils.make_grid(img_source.data), iteration)
        writer.add_image('Image_train/{}_target'.format(iteration), torchvision.utils.make_grid(img_target.data), iteration)
        writer.add_image('Image_train/{}_output'.format(iteration), torchvision.utils.make_grid(img_output.data), iteration)

    def visual_frame_img_flow(self, writer, iteration):
        img_save = torch.cat([
                                            self.input[0, 0:8,: ,: ,:], 
                                            self.input[0, 8:16,: ,: ,:], 
                                            self.input[0, 16:24,: ,: ,:], 
                                            self.input[0, 24:29,: ,: ,:],                 
                                            ]
                                            , 0)
        writer.add_image('Image_train/{}_input_aug'.format(iteration), torchvision.utils.make_grid(img_save.data), iteration)
        writer.add_image('Image_train/{}_input_aug_flow'.format(iteration), torchvision.utils.make_grid(self.flow_frame.data), iteration)