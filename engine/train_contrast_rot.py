import sys
import os
import argparse
import wandb

from sklearn.manifold import TSNE
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pytorch3d
import torch.optim as optim
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.train_utils import *
from tqdm import tqdm

from datasets.load_data_contrastive import get_data_loaders_from_cfg, process_batch_zero
from configs.config import get_config

from contrast.Cont_split_rot import Model_Rot_all as Model
from contrast.utils import AvgMeter, get_lr
from scipy.stats import spearmanr, pearsonr

def train_epoch(cfg, clip_model, train_loader, optimizer, lr_scheduler, step):
    clip_model.train()
    loss_meter = AvgMeter()
    
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    
    for batch in tqdm_object:
        ''' load data '''
        batch_sample = process_batch_zero(
            batch_sample = batch, 
            device=cfg.device, 
            pose_mode=cfg.pose_mode, 
            PTS_AUG_PARAMS=cfg.PTS_AUG_PARAMS, 
        )
        
        loss = clip_model(batch_sample)
        if loss is None:
            return None
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()
        
        count = batch_sample["zero_mean_pts_1"].size(0)
        loss_meter.update(loss.item(), count)
        
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(cfg, clip_model, valid_loader):
    loss_meter = AvgMeter()
    
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))

    for batch in tqdm_object:
        
        batch_sample = process_batch_zero(
            batch_sample = batch, 
            device=cfg.device, 
            pose_mode=cfg.pose_mode, 
            PTS_AUG_PARAMS=None, 
        )

        rot_loss = clip_model(batch_sample)
            
        count = batch_sample["zero_mean_pts_1"].size(0)
        loss_meter.update(rot_loss.item(), count)
        
        tqdm_object.set_postfix(val_loss=loss_meter.avg)
        
    return loss_meter


def test_epoch(cfg, clip_model, test_loader):
    loss_meter = AvgMeter()
    
    feat = []
    gt_rot_red = []
    gt_rot_green = []
    labels = []
    sym = []
    tqdm_object = tqdm(test_loader, total=len(test_loader))
    
    for batch in tqdm_object:
        
        batch_sample = process_batch_zero(
            batch_sample = batch, 
            device=cfg.device, 
            pose_mode=cfg.pose_mode, 
            PTS_AUG_PARAMS=None, 
        )
              
        rot_loss, pts_features, gt_green, gt_red = clip_model(batch_sample, umap=True, for_decoder=False)
        count = batch_sample["zero_mean_pts_1"].size(0)
        loss_meter.update(rot_loss.item(), count)
        
        tqdm_object.set_postfix(test_loss=loss_meter.avg)
        
        sym.append(batch_sample['sym'].cpu().detach().numpy())
        labels.append(batch_sample['id'].cpu().detach().numpy())
        feat.append(pts_features.cpu().detach().numpy())    
        gt_rot_red.append(gt_red.cpu().detach().numpy())
        gt_rot_green.append(gt_green.cpu().detach().numpy())
    
    sym = np.concatenate(sym, axis=0)
    labels = np.concatenate(labels, axis=0)
    feat = np.concatenate(feat, axis=0)    
    gt_rot_green = np.concatenate(gt_rot_green, axis=0)  
    gt_rot_red = np.concatenate(gt_rot_red, axis=0)  
    return loss_meter, feat, gt_rot_green, gt_rot_red, labels, sym

def rot_cos_sym(rot):
        
    # make sure the last row is [0, 0, 0, 1]
    rot = torch.from_numpy(rot)
    cos = nn.CosineSimilarity(dim=2, eps=1e-8)
    rot_x1 = rot.unsqueeze(1)  # bs*1*3
    rot_x2 = rot.unsqueeze(0)  # 1*bs*3
    rot_losses = 1.0 - cos(rot_x1, rot_x2)
    return rot_losses

def corr(feat, green, red, ids, sym):
    
    spearman_corr, pearsonr_corr = 0.0, 0.0
    all_ids = np.unique(ids)
    sym = torch.tensor(sym)
    for i in all_ids:
        ind = np.where(ids == i)[0]
        sym_ind = (sym[ind, 0] == 0).nonzero(as_tuple=True)[0] # find non-sym objects
        feat_id, green_id, red_id = feat[ind], green[ind], red[ind]
        
        if len(sym_ind) == 0: # sym obj
            label_diff = rot_cos_sym(green_id)
        else:
            label_diff = rot_cos_sym(green_id) + rot_cos_sym(red_id)
        feat_sim = - (torch.from_numpy(feat_id[:, None, :]) - torch.from_numpy(feat_id[None, :, :])).norm(2, dim=-1) 
        
        spearman_corr += spearmanr(label_diff.flatten(), feat_sim.flatten())[0]
        pearsonr_corr += pearsonr(label_diff.flatten(), feat_sim.flatten())[0]
        
    return spearman_corr / len(all_ids), pearsonr_corr / len(all_ids)

def pose_error(pose):
    
    rot = pytorch3d.transforms.rotation_6d_to_matrix(pose[:,:6])
    div = torch.pow((torch.linalg.det(rot)), 1/3)
    div = div.unsqueeze(dim=-1)
    div = div.unsqueeze(dim=-1)
    rot = rot / div
    t = pose[:, 6:]
    
    rot_trans = torch.transpose(rot, 1, 2)  # N*3*3
    rot = rot.unsqueeze(1) # N*1*3*3

    R = torch.matmul(rot, rot_trans)  # 2000*30000*3*3
    R_trace = torch.diagonal(R, offset=0, dim1=-1, dim2=-2).sum(-1)  # 2000*30000
    cos_theta = (R_trace - 1) / 2  # 2000*30000 , [0, 0] = -0.9055
    theta = torch.arccos(torch.clip(cos_theta, -1.0, 1.0)) * 180 / torch.pi
    
    t_reshaped = torch.tile(t.unsqueeze(1), (1, t.shape[0], 1))
    shift = torch.linalg.norm(t_reshaped - t, dim=-1) * 100

    return theta, shift

    
def main(cfg_clip):
    ### Setup CPUs ##############
    os.environ['OMP_NUM_THREADS'] = str(cfg_clip.cpu)
    os.environ['MKL_NUM_THREADS'] = str(cfg_clip.cpu)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cfg_clip.cpu)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cfg_clip.cpu)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cfg_clip.cpu)
    
    torch.set_num_threads(cfg_clip.cpu)
    
    ''' Init data loader '''
    if cfg_clip.train:
        data_loaders = get_data_loaders_from_cfg(cfg=cfg_clip, data_type=['train', 'val', 'test'])
        train_loader = data_loaders['train_loader']
        val_loader = data_loaders['val_loader']
        test_loader = data_loaders['test_loader']
        print('train_set: ', len(train_loader))
        print('val_set: ', len(val_loader))
        print('test_set: ', len(test_loader))
    else:
        data_loaders = get_data_loaders_from_cfg(cfg=cfg_clip, data_type=['test'])
        test_loader = data_loaders['test_loader']   
        print('test_set: ', len(test_loader))
  

    ''' Init CLIP trianing agent'''
    clip_model = Model().to(cfg_clip.device)
    start_epoch = 0
    if cfg_clip.pretrained_clip_model_path:
        clip_model.load_state_dict(torch.load(cfg_clip.pretrained_clip_model_path))
        start_epoch = cfg_clip.start_epoch
    
    if not os.path.exists(os.path.join(cfg_clip.results_path,'CLIP',cfg_clip.wandb_name)):
        os.makedirs(os.path.join(cfg_clip.results_path,'CLIP',cfg_clip.wandb_name))

    optimizer = torch.optim.AdamW(
        clip_model.parameters(), lr=cfg_clip.lr, weight_decay=cfg_clip.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=cfg_clip.patience, factor=cfg_clip.factor
    )
    step = "epoch"
    if cfg_clip.is_train:
        best_loss = float('inf')
        
        for epoch in range(start_epoch, cfg_clip.n_epochs):
            adjust_learning_rate(cfg_clip, optimizer, epoch)            
            train_loss= train_epoch(cfg_clip, clip_model, train_loader, optimizer, lr_scheduler, step)
            if train_loss is None:
                continue
            wandb.log({"train_loss":train_loss.__get_value__()
                       })
            clip_model.eval()
            with torch.no_grad():
                valid_loss = valid_epoch(cfg_clip, clip_model, val_loader)
            wandb.log({"val_loss": valid_loss.__get_value__()
                       })
            clip_model.eval()
            with torch.no_grad():
                test_loss, feat, gt_rot_green, gt_rot_red, labels, sym = test_epoch(cfg_clip, clip_model, test_loader)                
                spearman_corr, pearsonr_corr = corr(feat, gt_rot_green, gt_rot_red, labels, sym)
            wandb.log({"test_loss": test_loss.__get_value__(),
                       "spearman": spearman_corr,
                       "pearson":pearsonr_corr
                       })

            if best_loss >= valid_loss.__get_value__():
                best_loss = valid_loss.__get_value__()
                torch.save(clip_model.state_dict(), os.path.join(cfg_clip.results_path,'CLIP',cfg_clip.wandb_name,"best_epoch.pt"))
            # save latest epoch
            torch.save(clip_model.state_dict(), os.path.join(cfg_clip.results_path,'CLIP',cfg_clip.wandb_name,"latest_epoch.pt"))
            lr_scheduler.step(valid_loss.avg)    
                
            
    
        
        
if __name__ == '__main__':
    # load config
    cfg_clip = get_config()
    
    wandb.init(config=cfg_clip,
               project=cfg_clip.wandb_proj, 
               name=cfg_clip.wandb_name)
    main(cfg_clip)


