import torch
from torch import nn
import torch.nn.functional as F
import pytorch3d
import sys
sys.path.append('..')
from .rnc_loss import RnCLoss_trans
from configs.config import get_config 
from backbone.pts_encoder.pointnet2 import Pointnet2ClsMSG

CFG = get_config()

class Projection(nn.Module):
    
    def __init__(
        self,
        pts_embedding=CFG.pts_embedding
        ):
        super(Projection, self).__init__()
        self.projection_dim = pts_embedding
        self.w1 = nn.Linear(pts_embedding, pts_embedding, bias=False)
        self.bn1 = nn.BatchNorm1d(pts_embedding)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(pts_embedding, self.projection_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(self.projection_dim, affine=False)
    
    def forward(self, embedding):
        
        return self.bn2(self.w2(self.relu(self.bn1(self.w1(embedding)))))
        
  
        
class Model_Trans_all(nn.Module):
    def __init__(
        self,
        k1=CFG.k1,
        k2=CFG.k2,
        temperature=CFG.temperature,
        pts_embedding=CFG.pts_embedding,
        pose_embedding=256,
    ):
        super(Model_Trans_all, self).__init__()
        ''' encode point clouds '''
        
        self.pts_encoder = Pointnet2ClsMSG(0)

        self.project_head = Projection(1024)
        self.temperature = temperature
        self.clrk = Class_Rank(temperature=self.temperature,base_temperature=self.temperature)
    
    
       
    def forward(self, batch, umap=False, for_test=False, for_decoder=False):
        
        bs = batch['pts'].shape[0]
        pts_features = self.project_head(self.pts_encoder(batch['pts'])) #bs*N*3
        
        
        if torch.all(torch.isnan(pts_features))==False and torch.all(torch.isinf(pts_features))==False:
            if not for_decoder and not umap:
                # Getting point cloud and gt pose Features
                gt_pose = batch['gt_pose'][:, 9:]
                labels = batch['id']
                trans_loss = self.clrk(pts_features, labels, gt_pose)
                
                return trans_loss
            
            
            if for_decoder and not umap:
                return pts_features
            
            if umap:
                # Getting point cloud and gt pose Features
                gt_pose = batch['gt_pose'][:, 9:]
                labels = batch['id']
                trans_loss = self.clrk(pts_features, labels, gt_pose)
                return trans_loss, pts_features, gt_pose
        else:
            import pdb;pdb.set_trace()
            return None                
                
class Class_Rank(nn.Module):
    def __init__(self, temperature=2,
                 base_temperature=2, layer_penalty=None, loss_type='hmce'):
        super(Class_Rank, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        if not layer_penalty:
            self.layer_penalty = self.pow_2
        else:
            self.layer_penalty = layer_penalty
        # self.sup_con_loss = SupConLoss(temperature=self.temperature, contrast_mode='all', base_temperature=self.temperature, feature_sim='l2')
        self.loss_type = loss_type
        self.rnc_loss = RnCLoss_trans(temperature=self.temperature, label_diff='l1', feature_sim='l2')
        
        
    def pow_2(self, value):
        return torch.pow(2, value)

    def forward(self, features, labels, gt_T):
        device = features.device
        bs = labels.shape[0]
        
        t_layer_loss = torch.tensor(0.0).to(device)
        all_ids = torch.unique(labels)
        
        for i in all_ids:
            
            ind = torch.where(labels == i)[0]

            feat_id, t_id= features[ind], gt_T[ind]
            t_layer_loss += self.rnc_loss(feat_id, t_id)
            

        return t_layer_loss / len(all_ids)



