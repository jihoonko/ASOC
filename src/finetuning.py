from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
import time

from dataset import RadarAndRainMultiDatasetV2
from model import UNetWithASOC

import os
import pickle
from pyproj import Proj
import pdb

def load_coors(args):
    def location(lat, long):
            p = Proj("+proj=lcc +lat_1=30 +lat_2=60 +lat_0=38.0 +lon_0=126.0 +x_0=1024000 +y_0=1024000 +no_defs +ellps=WGS84 +units=km", preserve_units=True)
            x0, y0 = p(126.0, 38.0)
            x, y = p(long, lat)
            return (int(round(1024 + y - y0)), int(round(1024 + x - x0)))

    _cr, _cc = 1024, 1024
    _rr = 734

    aws_infos = []
    with open(f"{args.data_path}/aws_stn_info.txt", "r") as f:
        for line in f:
            _y, _x = location(float(line.split()[2]), float(line.split()[1]))
            _y, _x = _rr - (_y - _cr), _rr + (_x - _cc)
            aws_infos.append([_y, _x])
    # print(aws_infos)
    ret = torch.LongTensor(np.array(aws_infos))
    # print(ret.shape)
    return ret
            
def train(args):
    filename="finetuning"
    feat_mask=[i for i in range(0,18)]
    train_dataset = RadarAndRainMultiDatasetV2(sampled_path = args.sampled_path, data_path = args.data_path, year_from=2014, year_to=2018, feat_mask=feat_mask)
    val_dataset = RadarAndRainMultiDatasetV2(sampled_path = args.sampled_path, data_path = args.data_path, year_from=2019, year_to=2019, feat_mask=feat_mask)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, collate_fn = train_dataset.collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10, collate_fn = val_dataset.collate_fn, drop_last=True)
    coors = load_coors(args)
    coors_map = {(coors[i][0].item(), coors[i][1].item()): i for i in range(714)}
    model = UNetWithASOC(num_classes=3, img_dim=7, time_dim=36, initial_channels=32, prefeats=coors, feat_num=len(feat_mask), bn_at_first=False).to(args.default_device)
    if args.all_devices is not None:
        model = torch.nn.DataParallel(model, device_ids=args.all_devices, output_device=args.default_device)
        model.module.unet.load_state_dict(torch.load(args.pretrained_unet_weights_path))
    else:
        model.module.unet.load_state_dict(torch.load(args.pretrained_unet_weights_path))
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    step_cnt = 0
    train_loss_sum, train_acc = 0., 0
    TP1, TN1, FP1, FN1 = 0, 0, 0, 0
    TP2, TN2, FP2, FN2 = 0, 0, 0, 0

    model.train()
    f_log = open(f"./logs/{filename}.log", "a")
    while step_cnt < args.n_steps:
        start = time.time()
        for imgs, ff, indices, rows, cols, vals, wheres, pre_date in tqdm(iter(train_loader)):
            optimizer.zero_grad()
            
            imgs = imgs.to(args.default_device)
            indices, vals = indices.to(args.default_device), vals.to(args.default_device)
            
            _preds = model(imgs, (wheres * 6 + 5), ff, pre_date)
            converted_indices = torch.LongTensor([coors_map.get((_r, _c), 0) for _r, _c in zip(rows.numpy().tolist(), cols.numpy().tolist())]).to(args.default_device)
            
            if step_cnt == 0:
                print((converted_indices == 0).long().sum(), (converted_indices > 0).long().sum())
                exit(0)
            
            preds = torch.softmax(_preds[indices, converted_indices, :], dim=-1)
            labels = torch.argmax(preds, dim=-1).to(args.default_device)
            wheres = wheres.to(args.default_device)[indices]
            
            loss = 0
            for i in range(1, 2+1):
                smooth_TP = (preds[:, i:].sum(-1) * (vals > i-1).float()).sum()
                smooth_TN = (preds[:, :i].sum(-1) * (vals < i).float()).sum()
                smooth_FP = (preds[:, i:].sum(-1) * (vals < i).float()).sum()
                smooth_FN = (preds[:, :i].sum(-1) * (vals > i-1).float()).sum()
                loss -= (smooth_TP / (smooth_TP + smooth_FN + smooth_FP + 1e-6)) 
            
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            train_acc += (labels == vals).float().sum().item()
            
            TP1 += ((labels > 0) & (vals > 0)).float().sum().item()
            TN1 += ((labels == 0) & (vals == 0)).float().sum().item()
            FP1 += ((labels > 0) & (vals == 0)).float().sum().item()
            FN1 += ((labels == 0) & (vals > 0)).float().sum().item()
            
            TP2 += ((labels > 1) & (vals > 1)).float().sum().item()
            TN2 += ((labels < 2) & (vals < 2)).float().sum().item()
            FP2 += ((labels > 1) & (vals < 2)).float().sum().item()
            FN2 += ((labels < 2) & (vals > 1)).float().sum().item()
            del _preds, preds, loss, labels
            
            step_cnt += 1
            if step_cnt % 100 == 0:
                f_log.write(f'Step #{step_cnt}: train_loss is {train_loss_sum / 100.} train_acc: {train_acc / (TP1 + FN1 + FP1 + TN1)} ')
                f_log.write(f'F1_1mm: {(2 * TP1) / (2 * TP1 + FN1 + FP1)} F1_10mm: {(2 * TP2) / (2 * TP2 + FN2 + FP2)}\t')
                f_log.write(f'CSI_1mm: {(TP1) / (TP1 + FN1 + FP1)} CSI_10mm: {(TP2) / (TP2 + FN2 + FP2)}\n')

                f_log.flush()
                train_loss_sum, train_acc = 0., 0
                TP1, TN1, FP1, FN1 = 0, 0, 0, 0
                TP2, TN2, FP2, FN2 = 0, 0, 0, 0
    
            if step_cnt % 1000 == 0:
                end = time.time()
                f_log.write(f'Duration time is {end - start}\n')
                f_log.flush()
                model.eval()
                torch.save(model.module.state_dict(), f'./checkpoints/{filename}_{step_cnt}.pkt')
                val_loss_sum, val_acc = 0., 0
                _TP1, _TN1, _FP1, _FN1 = 0, 0, 0, 0
                _TP2, _TN2, _FP2, _FN2 = 0, 0, 0, 0
                _TP = np.array([0 for _ in range(12)])
                _TN = np.array([0 for _ in range(12)])
                _FP = np.array([0 for _ in range(12)])
                _FN = np.array([0 for _ in range(12)])
                
                with torch.no_grad():
                    for imgs, ff, indices, rows, cols, vals, wheres, pre_date in tqdm(iter(val_loader)):
                        imgs = imgs.to(args.default_device)
                        indices, vals = indices.to(args.default_device), vals.to(args.default_device)
                        
                        _preds = model(imgs, (wheres * 6 + 5), ff, pre_date)
                        converted_indices = torch.LongTensor([coors_map.get((_r, _c), 0) for _r, _c in zip(rows.numpy().tolist(), cols.numpy().tolist())]).to(args.default_device)
                        preds = torch.softmax(_preds[indices, converted_indices, :], dim=-1)
                        labels = torch.argmax(preds, dim=-1).to(args.default_device)
                        
                        loss = 0
                        for i in range(1, 2+1):
                            smooth_TP = (preds[:, i:].sum(-1) * (vals > i-1).float()).sum()
                            smooth_TN = (preds[:, :i].sum(-1) * (vals < i).float()).sum()
                            smooth_FP = (preds[:, i:].sum(-1) * (vals < i).float()).sum()
                            smooth_FN = (preds[:, :i].sum(-1) * (vals > i-1).float()).sum()
                            loss -= i*(smooth_TP / (smooth_TP + smooth_FN + smooth_FP + 1e-6)) 
                        
                        val_loss_sum += loss.item()                        
                        val_acc += (labels == vals).float().sum().item()
                        
                        _TP1 += ((labels > 0) & (vals > 0)).float().sum().item()
                        _TN1 += ((labels == 0) & (vals == 0)).float().sum().item()
                        _FP1 += ((labels > 0) & (vals == 0)).float().sum().item()
                        _FN1 += ((labels == 0) & (vals > 0)).float().sum().item()
                        _TP2 += ((labels > 1) & (vals > 1)).float().sum().item()
                        _TN2 += ((labels < 2) & (vals < 2)).float().sum().item()
                        _FP2 += ((labels > 1) & (vals < 2)).float().sum().item()
                        _FN2 += ((labels < 2) & (vals > 1)).float().sum().item()
                        
                        wheres = wheres.to(args.default_device)[indices]
                        for i in range(1, 2+1):
                            for j in range(6):
                                _TP[(i-1) * 6 + j] += ((wheres == j).float() * preds[:, i:].sum(-1) * (vals > i-1).float()).sum().item()
                                _TN[(i-1) * 6 + j] += ((wheres == j).float() * preds[:, :i].sum(-1) * (vals < i).float()).sum().item()
                                _FP[(i-1) * 6 + j] += ((wheres == j).float() * preds[:, i:].sum(-1) * (vals < i).float()).sum().item()
                                _FN[(i-1) * 6 + j] += ((wheres == j).float() * preds[:, :i].sum(-1) * (vals > i-1).float()).sum().item()
                        
                        del _preds, preds, loss, labels
                        
                _CSI = ' '.join(map(str, (_TP / (_TP + _FN + _FP + 1e-6)).tolist()))
                f_log.write(f'val_loss is {val_loss_sum} val_acc: {val_acc / (_TP1 + _TN1 + _FP1 + _FN1)} | ')
                f_log.write(_CSI + '\n')
                f_log.flush()
                model.train()
                start = time.time()
            if step_cnt == args.n_steps:
                break    
            
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='fine-tuning (U-Net)')
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="learning rate")
    parser.add_argument("--n-steps", type=int, default=50000,
                        help="number of training steps")
    parser.add_argument("--batch-size", type=int, default=24,
                        help="number of batch size")
    parser.add_argument("--data-path", type=str, default=None,
                        help="path of radar data")
    parser.add_argument("--sampled-path", type=str, default='./sampled/Train_(2014-2019)',
                        help="path of sampled data for training")
    parser.add_argument("--pretrained-unet-weights-path", type=str, default=f'./example_checkpoints/pretrained.pkt',
                        help="path of pretrained weights")
    parser.add_argument("--gpus", type=str, default=None,
                        help="gpu id for execution")
    
    args = parser.parse_args()
    if torch.cuda.is_available():
        if args.gpus == 'cpu':
            args.all_devices = None
            args.default_device = 'cpu:0'
        elif args.gpus is not None:
            args.all_devices = list(map(int, args.gpus.split(',')))
            args.default_device = args.all_devices[0]
        else:
            args.all_devices = [i for i in range(torch.cuda.device_count())]
            args.default_device = torch.cuda.current_device()
    else:
        args.all_devices = None
        args.default_device = 'cpu:0'
    
    if args.data_path is None:
        print('Path of the RADAR data should be provided!')
        exit(0)
    
    print(f'enabled gpu_devices: {args.all_devices}, default device: {args.default_device}')
    print(args)
    train(args)