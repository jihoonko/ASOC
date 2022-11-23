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
import sys

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
    ret = torch.LongTensor(np.array(aws_infos))
    return ret
            
def evaluation(args):
    feat_mask=[i for i in range(0,18)]
    f_log = sys.stdout
    
    test_dataset = RadarAndRainMultiDatasetV2(sampled_path = args.sampled_path, data_path = args.data_path, year_from=2020, year_to=2020, feat_mask=feat_mask)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, collate_fn = test_dataset.collate_fn)
    coors = load_coors(args)
    coors_map = {(coors[i][0].item(), coors[i][1].item()): i for i in range(714)}
    model = UNetWithASOC(num_classes=3, img_dim=7, time_dim=36, initial_channels=32, prefeats=coors, feat_num=len(feat_mask), bn_at_first=False).to(args.default_device)
    if args.all_devices is not None:
        model = torch.nn.DataParallel(model, device_ids=args.all_devices, output_device=args.default_device)
        model.module.load_state_dict(torch.load(args.finetuned_weights_path))
    else:
        model.load_state_dict(torch.load(args.finetuned_weights_path))
        
    _preds_history, _vals_history, _interval_history = [], [], []
    model.eval()
    with torch.no_grad():
        cntt = 0
        for imgs, ff, indices, rows, cols, vals, wheres, pre_date in tqdm(iter(test_loader)):
            cntt += 1
            imgs = imgs.to(args.default_device)
            indices, vals = indices.to(args.default_device), vals.to(args.default_device)

            _preds = model(imgs, (wheres * 6 + 5), ff, pre_date)
            converted_indices = torch.LongTensor([coors_map.get((_r, _c), 0) for _r, _c in zip(rows.numpy().tolist(), cols.numpy().tolist())]).to(args.default_device)
            preds = torch.softmax(_preds[indices, converted_indices, :], dim=-1)
            labels = torch.argmax(preds, dim=-1).to(args.default_device)

            _preds_history += labels.detach().cpu().numpy().tolist()
            _vals_history += vals.detach().cpu().numpy().tolist()
            _interval_history += wheres[indices].detach().cpu().numpy().tolist()
            del _preds, preds, labels
            
        all_mats = np.array([[0 for _1 in range(3)] for _2 in range(3)])
        confusion_matrix = np.array([[[0 for _1 in range(3)] for _2 in range(3)] for _3 in range(6)])
        for r, c, i in zip(_preds_history, _vals_history, _interval_history):
            confusion_matrix[i][r][c] += 1
            all_mats[r][c] += 1
            
        sum_loss = 0.
        for i in range(6):
            sum_loss += confusion_matrix[i][2][2] / (confusion_matrix[i][2][2] + confusion_matrix[i][2][1] + confusion_matrix[i][2][0] + confusion_matrix[i][1][2] + confusion_matrix[i][0][2])
            sum_loss += np.sum(confusion_matrix[i][1:,1:]) / (np.sum(confusion_matrix[i]) - confusion_matrix[i][0][0])
        
        f_log.write(f"{all_mats[2][2] / (all_mats[2][2] + all_mats[2][1] + all_mats[2][0] + all_mats[1][2] + all_mats[0][2])} \n")
        f_log.write(f"{np.sum(all_mats[1:,1:]) / (np.sum(all_mats) - all_mats[0][0])} \n")
        
        f_log.write(f'confusion matrix (1h):\n {np.array(confusion_matrix[0])} \n')
        f_log.write(f'confusion matrix (2h):\n {np.array(confusion_matrix[1])} \n')
        f_log.write(f'confusion matrix (3h):\n {np.array(confusion_matrix[2])} \n')
        f_log.write(f'confusion matrix (4h):\n {np.array(confusion_matrix[3])} \n')
        f_log.write(f'confusion matrix (5h):\n {np.array(confusion_matrix[4])} \n')
        f_log.write(f'confusion matrix (6h):\n {np.array(confusion_matrix[5])} \n')
        
        ## CSI Measure
        f_log.write("CSI Score\n")
        for i in range(6):
            f_log.write(f'CSI score ({i+1}h, >= 10mm) {confusion_matrix[i][2][2] / (confusion_matrix[i][2][2] + confusion_matrix[i][2][1] + confusion_matrix[i][2][0] + confusion_matrix[i][1][2] + confusion_matrix[i][0][2])}\n')
            f_log.write(f'CSI score ({i+1}h, >= 1mm) {(confusion_matrix[i][1][1]+confusion_matrix[i][1][2]+confusion_matrix[i][2][1]+confusion_matrix[i][2][2]) / (np.sum(confusion_matrix[i]) - confusion_matrix[i][0][0])}\n')
            f_log.write(f'CSI score ({i+1}h, < 1mm) {(confusion_matrix[i][0][0]/(np.sum(confusion_matrix[i]) - (confusion_matrix[i][1][1]+confusion_matrix[i][1][2]+confusion_matrix[i][2][1]+confusion_matrix[i][2][2])))}\n')

        ## F1 Score
        f_log.write("F1 Score\n")
        for i in range(6):
            f_log.write(f'F1 score ({i+1}h, >= 10mm) {2*confusion_matrix[i][2][2] / (2*confusion_matrix[i][2][2] + confusion_matrix[i][2][1] + confusion_matrix[i][2][0] + confusion_matrix[i][1][2] + confusion_matrix[i][0][2])}\n')
            x = (confusion_matrix[i][1][1]+confusion_matrix[i][1][2]+confusion_matrix[i][2][1]+confusion_matrix[i][2][2])
            f_log.write(f'F1 score ({i+1}h, >= 1mm) {2*x / (2*x + confusion_matrix[i][0][1] + confusion_matrix[i][0][2] + confusion_matrix[i][1][0] + confusion_matrix[i][2][0])}\n')
            f_log.write(f'F1 score ({i+1}h, < 1mm) {2* confusion_matrix[i][0][0]/(2* confusion_matrix[i][0][0] + confusion_matrix[i][0][1] + confusion_matrix[i][0][2] + confusion_matrix[i][1][0] + confusion_matrix[i][2][0])}\n')
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='evaluation (U-Net)')
    parser.add_argument("--batch-size", type=int, default=24,
                        help="number of batch size")
    parser.add_argument("--data-path", type=str, default=None,
                        help="path of radar data")
    parser.add_argument("--sampled-path", type=str, default='./sampled/Test_2020',
                        help="path of sampled data for evaluation")
    parser.add_argument("--finetuned-weights-path", type=str, default=f'./example_checkpoints/finetuned.pkt',
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
    
    evaluation(args)