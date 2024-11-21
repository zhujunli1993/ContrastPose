import torch.nn as nn
import torch
import torch.nn.functional as F
import absl.flags as flags
from absl import app
from .Cross_Atten import CrossAttention
FLAGS = flags.FLAGS

# Point_center  encode the segmented point cloud
# one more conv layer compared to original paper

class Pose_Ts(nn.Module):
    def __init__(self):
        super(Pose_Ts, self).__init__()
        self.f = FLAGS.feat_c_ts
        self.k = FLAGS.Ts_c
        if FLAGS.use_clip_global==0.0:
            self.conv1 = torch.nn.Conv1d(self.f, 1024, 1)

            self.conv2 = torch.nn.Conv1d(1024, 256, 1)
            self.conv3 = torch.nn.Conv1d(256, 256, 1)
            self.conv4 = torch.nn.Conv1d(256, self.k, 1)
            self.drop1 = nn.Dropout(0.2)
            self.bn1 = nn.BatchNorm1d(1024)
            self.bn2 = nn.BatchNorm1d(256)
            self.bn3 = nn.BatchNorm1d(256)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()
        else:
            
            self.conv1 = torch.nn.Conv1d(self.f, 1024, 1)

            self.conv2 = torch.nn.Conv1d(1024, 256, 1)
            self.conv3 = torch.nn.Conv1d(256+1024, 256, 1)
            self.conv4 = torch.nn.Conv1d(256, self.k, 1)
            self.drop1 = nn.Dropout(0.2)
            self.bn1 = nn.BatchNorm1d(1024)
            self.bn2 = nn.BatchNorm1d(256)
            self.bn3 = nn.BatchNorm1d(256)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()
    def forward(self, x, clip_feat_t=None, use_clip=False, use_clip_global=False):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        if clip_feat_t is not None and use_clip is True and use_clip_global is True:
            clip_feat_t = torch.unsqueeze(clip_feat_t, dim=-1)
            
            x = torch.cat([x, clip_feat_t], dim=1)
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.drop1(x)
        x = self.conv4(x)

        x = x.squeeze(2)
        x = x.contiguous()
        xt = x[:, 0:3]
        xs = x[:, 3:6]
        return xt, xs
class Pose_Ts_nonLinear(nn.Module):
    def __init__(self):
        super(Pose_Ts_nonLinear, self).__init__()
        self.f = FLAGS.feat_c_ts
        self.k = FLAGS.Ts_c
        if FLAGS.use_clip_global==0.0:
            self.conv1 = torch.nn.Conv1d(self.f, 1024, 1)

            self.conv2 = torch.nn.Conv1d(1024, 256, 1)
            self.conv3 = torch.nn.Conv1d(256, 256, 1)
            self.conv4 = torch.nn.Conv1d(256, self.k, 1)
            self.drop1 = nn.Dropout(0.2)
            self.bn1 = nn.BatchNorm1d(1024)
            self.bn2 = nn.BatchNorm1d(256)
            self.bn3 = nn.BatchNorm1d(256)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()
            
            self.conv_clip = torch.nn.Conv1d(1024, self.k, 1)
        else:
            
            self.conv1 = torch.nn.Conv1d(self.f, 1024, 1)

            self.conv2 = torch.nn.Conv1d(1024, 256, 1)
            self.conv3 = torch.nn.Conv1d(256+1024, 256, 1)
            self.conv4 = torch.nn.Conv1d(256, self.k, 1)
            self.drop1 = nn.Dropout(0.2)
            self.bn1 = nn.BatchNorm1d(1024)
            self.bn2 = nn.BatchNorm1d(256)
            self.bn3 = nn.BatchNorm1d(256)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()
    def forward(self, x, clip_feat_t=None, use_clip=False, use_clip_global=False):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        if clip_feat_t is not None and use_clip is True and use_clip_global is False:
            clip_feat_t = torch.unsqueeze(clip_feat_t, dim=-1)
            x_clip = self.conv_clip(clip_feat_t)
            x_clip = x_clip.squeeze(2)
            x_clip = x_clip.contiguous()
            
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.drop1(x)
        x = self.conv4(x)

        x = x.squeeze(2)
        x = x.contiguous()
        
        final_x = (x + x_clip) / 2.0
        xt = final_x[:, 0:3]
        xs = final_x[:, 3:6]
        return xt, xs

class Pose_Ts_atten(nn.Module):
    def __init__(self):
        super(Pose_Ts_atten, self).__init__()
        self.f = FLAGS.feat_c_ts
        self.k = FLAGS.Ts_c
        if FLAGS.use_clip_global==0.0:
            self.conv1 = torch.nn.Conv1d(self.f, 1024, 1)

            self.conv2 = torch.nn.Conv1d(1024, 256, 1)
            self.conv3 = torch.nn.Conv1d(256, 256, 1)
            self.conv4 = torch.nn.Conv1d(256, self.k, 1)
            self.drop1 = nn.Dropout(0.2)
            self.bn1 = nn.BatchNorm1d(1024)
            self.bn2 = nn.BatchNorm1d(256)
            self.bn3 = nn.BatchNorm1d(256)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()
        else:
            self.cross = CrossAttention(dim=256, heads=FLAGS.heads)
            self.conv_clip = torch.nn.Conv1d(1024, 256, 1)
            self.conv1 = torch.nn.Conv1d(self.f, 1024, 1)
            self.conv2 = torch.nn.Conv1d(1024, 256, 1)
            self.conv3 = torch.nn.Conv1d(256, 256, 1)
            self.conv4 = torch.nn.Conv1d(256, self.k, 1)
            self.drop1 = nn.Dropout(0.2)
            self.bn1 = nn.BatchNorm1d(1024)
            self.bn2 = nn.BatchNorm1d(256)
            self.bn3 = nn.BatchNorm1d(256)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()
    def forward(self, x, clip_feat_t=None, use_clip=False, use_clip_global=False):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        
        if clip_feat_t is not None and use_clip is True and use_clip_global is True:
            clip_feat_t = torch.unsqueeze(clip_feat_t, dim=-1)
            clip_feat_t = self.conv_clip(clip_feat_t)
            
        x = x.permute(0, 2, 1)
        clip_feat_t = clip_feat_t.permute(0, 2, 1)
        x = self.cross(x_kv=x, x_q=clip_feat_t)  
         
        x = x.permute(0, 2, 1)  
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.drop1(x)
        x = self.conv4(x)

        x = x.squeeze(2)
        x = x.contiguous()
        xt = x[:, 0:3]
        xs = x[:, 3:6]
        return xt, xs


def main(argv):
    feature = torch.rand(3, 3, 1000)
    obj_id = torch.randint(low=0, high=15, size=[3, 1])
    net = Pose_Ts()
    out = net(feature, obj_id)
    t = 1

if __name__ == "__main__":
    app.run(main)
