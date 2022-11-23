import torch
from torch import nn

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, act_fn = nn.ReLU, use_batchnorm=True):
        super(ConvBNAct, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3)
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.act = act_fn()
        self.use_batchnorm = use_batchnorm
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
        
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, pool_fn = nn.MaxPool2d):
        super(DownSample, self).__init__()
        self.pool = pool_fn(2)
        self.conv1 = ConvBNAct(in_channels, out_channels)
        self.conv2 = ConvBNAct(out_channels, out_channels)
        
    def forward(self, x):
        return self.conv2(self.conv1(self.pool(x)))

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, bn_at_last=True):
        super(UpSample, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv1 = ConvBNAct(out_channels + out_channels, out_channels, use_batchnorm=bn_at_last)
        self.conv2 = ConvBNAct(out_channels, out_channels, use_batchnorm=bn_at_last)
        
    def forward(self, x, snapshot):
        x = self.upsample(x)
        crop_idx = (snapshot.shape[-1] - x.shape[-1]) // 2
        x = self.conv2(self.conv1(torch.cat((snapshot[:, :, crop_idx:-crop_idx, crop_idx:-crop_idx], x), dim=-3)))
        return x
        
        
class UNet(nn.Module):
    def __init__(self, initial_channels = 32):
        super(UNet, self).__init__()
        self.initial_convs = nn.Sequential(ConvBNAct(7, initial_channels),
                                           ConvBNAct(initial_channels, initial_channels))
        self.down_layers = nn.ModuleList([DownSample((initial_channels << i), (initial_channels << (i+1))) for i in range(6)])
        self.up_layers = nn.ModuleList([UpSample((initial_channels << (6-i)), (initial_channels << (5-i))) for i in range(6)])
        self.last_conv = nn.Conv2d(initial_channels, 6, 3)
        
    def forward(self, img):
        x = self.initial_convs(img)
        self.snapshots = []
        for layer in self.down_layers:
            self.snapshots.append(x)
            x = layer(x)
        for i, layer in enumerate(self.up_layers):
            x = layer(x, self.snapshots[-(i+1)])
        del self.snapshots
        x = self.last_conv(x).squeeze(-3)
        return x

class ConvBNActTime(nn.Module):
    def __init__(self, img_dim, time_dim, out_channels, act_fn = nn.ReLU, use_batchnorm=True):
        super(ConvBNActTime, self).__init__()
        self.img_conv = nn.Conv2d(img_dim, out_channels, 3)
        self.time_conv = nn.Conv2d(time_dim, out_channels, 3, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) # if use_batchnorm else nn.Identity()
        self.act = act_fn()
        
    def forward(self, img, time=None):
        x = self.img_conv(img)
        if time is not None:
            f_time = self.time_conv.weight[:, time].transpose(0, 1).sum(dim=[-2,-1], keepdim=True)
            x += f_time
        return self.act(x)
    
class UNetV2(nn.Module):
    def __init__(self, img_dim, time_dim = 0, initial_channels = 32, use_batchnorm_at_first=False, use_batchnorm_at_last=True):
        super(UNetV2, self).__init__()
        self.initial_conv = ConvBNActTime(img_dim, time_dim, initial_channels, use_batchnorm=use_batchnorm_at_first)
        self.second_conv = ConvBNAct(initial_channels, initial_channels, use_batchnorm=use_batchnorm_at_first)
        self.down_layers = nn.ModuleList([DownSample((initial_channels << i), (initial_channels << (i+1))) for i in range(6)])
        self.up_layers = nn.ModuleList([UpSample((initial_channels << (6-i)), (initial_channels << (5-i)), bn_at_last = (use_batchnorm_at_last or (i < 5))) for i in range(6)])
        
    def forward(self, img, time=None):
        x = self.second_conv(self.initial_conv(img, time))
        self.snapshots = []
        for layer in self.down_layers:
            self.snapshots.append(x)
            x = layer(x)
        for i, layer in enumerate(self.up_layers):
            x = layer(x, self.snapshots[-(i+1)])
        del self.snapshots
        return x

class UNetWithASOC(nn.Module):
    '''
    Self attention for every station in each timestamp 
    transformer through time per station
    '''
    def __init__(self, num_classes, img_dim, time_dim = 0, initial_channels=32, input_dim=7, prefeats=None, feat_num=18, hidden_dim=64, bn_at_first=True):
        super(UNetAttentionV1, self).__init__()
        self.unet = UNetV2(img_dim, time_dim, initial_channels, use_batchnorm_at_first=bn_at_first)
        
        self.input_dim = input_dim
        self.feat_num = feat_num
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        self.register_buffer('pre_aws', prefeats - 380)
        self.lstmcell = nn.LSTMCell(initial_channels + self.feat_num + 12, self.hidden_dim)
        
        self.mha = nn.MultiheadAttention(self.hidden_dim, self.hidden_dim // 16)
        self.mlp = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim * 4), nn.ReLU(), nn.Linear(self.hidden_dim * 4, self.hidden_dim))
        self.last_fc = nn.Linear(initial_channels + self.hidden_dim + 12, num_classes)
        
    def forward(self, img, time=None, feats=None, pre_date=None):
        # img: #batch, #channel, #coor_x, #coor_y
        # time: #batch
        # feats: #batch, #stn, (#inputdim * #featnum)
        # pre_date: #batch, 10
        x = self.unet(img, time) # (#batch, #channel, #coor_x, #coor_y)
        sampled_x = x[:, :, self.pre_aws[:, 0], self.pre_aws[:, 1]].permute(2, 0, 1).contiguous() # (#stn, #batch, #channel)
        num_stns, batch_size, _ = sampled_x.shape
        feats = feats.view(batch_size, num_stns, self.input_dim, self.feat_num).permute(2, 1, 0, 3).contiguous() # (#inputdim, #stn, #batch, #featnum)
        time_feats = pre_date.unsqueeze(0).repeat(num_stns, 1, 1) # (#stn, #batch, 10)
        loc_feats = (self.pre_aws.float() / x.shape[-1]).unsqueeze(1).repeat(1, batch_size, 1) # (#stn, #batch, 2)
        aux_feats = torch.cat((sampled_x, time_feats, loc_feats), dim=-1) # (#stn, #batch, #channel+10+2)
        
        feats = torch.cat((feats, aux_feats.unsqueeze(0).repeat(self.input_dim, 1, 1, 1)), dim=-1) # (#inputdim, #stn, #batch, #channel + #featnum + 10 + 2)
        curr_hidden_state = torch.zeros(num_stns, batch_size, self.hidden_dim).to(feats.device) # (#stn, #batch, #hdim)
        curr_cell_state = torch.zeros(num_stns * batch_size, self.hidden_dim).to(feats.device) # ((#stn * #batch), #hdim)
        
        for i in range(self.input_dim):
            curr_hidden_state, curr_cell_state = self.lstmcell(feats[i].view(batch_size * num_stns, -1), (curr_hidden_state.view(-1, self.hidden_dim), curr_cell_state)) # ((#stn * #batch), #hdim)
            curr_hidden_state = curr_hidden_state.view(num_stns, batch_size, self.hidden_dim)
            curr_hidden_state = curr_hidden_state + self.mha(curr_hidden_state, curr_hidden_state, curr_hidden_state)[0]
            curr_hidden_state = curr_hidden_state + self.mlp(curr_hidden_state)
        
        encoded_feats = torch.cat((curr_hidden_state, aux_feats), dim=-1).permute(1, 0, 2).contiguous() # (#batch, #stn, #hdim + #channel + (10+2))
        return self.last_fc(encoded_feats)
    
class ASOCOnly(nn.Module):
    '''
    Self attention for every station in each timestamp 
    transformer through time per station
    '''
    def __init__(self, num_classes, img_dim, time_dim = 0, initial_channels=32, input_dim=7, prefeats=None, feat_num=18, hidden_dim=64, bn_at_first=True):
        super(OnlyAttentionV1, self).__init__()
        self.input_dim = input_dim
        self.feat_num = feat_num
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        self.register_buffer('pre_aws', prefeats - 380)
        self.lstmcell = nn.LSTMCell(self.feat_num + 12, self.hidden_dim)
        
        self.mha = nn.MultiheadAttention(self.hidden_dim, self.hidden_dim // 16)
        self.mlp = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim * 4), nn.ReLU(), nn.Linear(self.hidden_dim * 4, self.hidden_dim))
        self.last_fc = nn.Linear(self.hidden_dim + 12, num_classes)
        
    def forward(self, time=None, feats=None, pre_date=None):
        # x = self.unet(img, time) # (#batch, #channel, #coor_x, #coor_y)
        # sampled_x = x[:, :, self.pre_aws[:, 0], self.pre_aws[:, 1]].permute(2, 0, 1).contiguous() # (#stn, #batch, #channel)
        # num_stns, batch_size, _ = sampled_x.shape
        batch_size = time.shape[0]
        num_stns = 714
        
        feats = feats.view(batch_size, num_stns, self.input_dim, self.feat_num).permute(2, 1, 0, 3).contiguous() # (#inputdim, #stn, #batch, #featnum)
        time_feats = pre_date.unsqueeze(0).repeat(num_stns, 1, 1) # (#stn, #batch, 10)
        loc_feats = (self.pre_aws.float() / 706).unsqueeze(1).repeat(1, batch_size, 1) # (#stn, #batch, 2)
        aux_feats = torch.cat((time_feats, loc_feats), dim=-1) # (#stn, #batch, #channel+10+2)
        
        feats = torch.cat((feats, aux_feats.unsqueeze(0).repeat(self.input_dim, 1, 1, 1)), dim=-1) # (#inputdim, #stn, #batch, #channel + #featnum + 10 + 2)
        curr_hidden_state = torch.zeros(num_stns, batch_size, self.hidden_dim).to(feats.device) # (#stn, #batch, #hdim)
        curr_cell_state = torch.zeros(num_stns * batch_size, self.hidden_dim).to(feats.device) # ((#stn * #batch), #hdim)
        
        for i in range(self.input_dim):
            curr_hidden_state, curr_cell_state = self.lstmcell(feats[i].view(batch_size * num_stns, -1), (curr_hidden_state.view(-1, self.hidden_dim), curr_cell_state)) # ((#stn * #batch), #hdim)
            curr_hidden_state = curr_hidden_state.view(num_stns, batch_size, self.hidden_dim)
            curr_hidden_state = curr_hidden_state + self.mha(curr_hidden_state, curr_hidden_state, curr_hidden_state)[0]
            curr_hidden_state = curr_hidden_state + self.mlp(curr_hidden_state)
        
        encoded_feats = torch.cat((curr_hidden_state, aux_feats), dim=-1).permute(1, 0, 2).contiguous() # (#batch, #stn, #hdim + #channel + (10+2))
        return self.last_fc(encoded_feats)