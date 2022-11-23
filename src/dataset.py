import gzip
import numpy as np
import torch
import torch.nn as nn
from itertools import chain
import pickle
import os
import random
import tqdm
import pdb
import math

class RadarAndRainMultiDatasetV2(torch.utils.data.Dataset):
    """ For UnetMultiClassificationV2 """
    def __init__(self, sampled_path, data_path, year_from=2014, year_to=2018, feat_mask = [i for i in range(0,18)], interval=60, min_interval=10, max_interval=360, input_dim=7, center=(1024, 1024), radius=734):
        assert type(year_from) == int and type(year_to) == int and type(input_dim) == int
        assert type(interval) == int and interval % 10 == 0
        
        self.date = []
        self.days = [30, 31, 31, 30]
        self.dates = list(chain.from_iterable([chain.from_iterable([map(str, np.arange(year * 10000 + (month + 1) * 100 + 1,
                                                                                       year * 10000 + (month + 1) * 100 + 1 +
                                                                                       self.days[month-5] + int((month == 1) and (year % 4 == 0))))
                                                                    for month in range(6-1, 9)]) for year in range(year_from, year_to + 1)]))
        self.inv_dates = {v: i for i, v in enumerate(self.dates)}
        self.interval, self.input_dim = interval, input_dim
        self.min_interval, self.max_interval = min_interval, max_interval
        self.num_time_slots = (self.max_interval - self.min_interval) // 10 + 1
        self.center = center
        self.radius = radius
        self.data_path = data_path
        self.feat_mask = feat_mask
        
        with open(sampled_path, "rb") as f:
            raw_raindata = pickle.load(f)
            multi_raindata = list(chain.from_iterable([[((k, i), v) for k, v in raw_raindata] for i in range(1, 6+1)]))
            self.raindata = list(filter(lambda x: (year_from <= int(x[0][0][:4]) <= year_to) and self.load_images(x[0][0], x[0][1], just_check=True), multi_raindata))
            random.shuffle(self.raindata)
    
    def load_feats(self, timestamp):
        def parse_feats(_idx):
            datestr = "%s%02d%1d0" % (self.dates[_idx // 144], (_idx % 144) // 6, _idx % 6)
            with open(f"{self.data_path}/{datestr[:4]}/{datestr[:6]}/ground_{datestr}.pkl", "rb") as f:
                _feat = pickle.load(f)
                return _feat      
        
        return torch.FloatTensor(parse_feats(timestamp)[:,self.feat_mask])
    
    def load_images(self, timestamp, inv, just_check=False):
        def load_image_inner_check(_idx):
            datestr = "%s%02d%1d0" % (self.dates[_idx // 144], (_idx % 144) // 6, _idx % 6)
            return os.path.exists(f'{self.data_path}/{datestr[:6]}/{datestr[:10]}/radar_{datestr}.bin.gz')
        
        def load_image_inner(_idx):
            datestr = "%s%02d%1d0" % (self.dates[_idx // 144], (_idx % 144) // 6, _idx % 6)
            with gzip.open(f'{self.data_path}/{datestr[:6]}/{datestr[:10]}/radar_{datestr}.bin.gz', 'rb') as f:
                f.seek(1024)
                target = f.read()
                result = np.frombuffer(target, 'i2').reshape(2048, 2048)
                img = np.maximum(result[(self.center[0]+self.radius):(self.center[0]-self.radius):-1, (self.center[1]-self.radius):(self.center[1]+self.radius)], 0) / 10000.
            return img
        
        date, hour = timestamp[:-4], int(timestamp[-4:-2])
        idx = self.inv_dates[date] * 144 + hour * 6
        
        if just_check:
            valid = True
            for i in range(self.input_dim):
                if not load_image_inner_check(idx - ((self.input_dim - 1) + (inv * (self.interval // 10))) + i):
                    valid = False
                    break
            return valid
        else:
            history = torch.FloatTensor(np.stack([load_image_inner(idx - ((self.input_dim - 1) + (inv * (self.interval // 10))) + i) for i in range(self.input_dim)], axis=0))
            feats = torch.cat([self.load_feats(idx - ((self.input_dim - 1) + (inv * (self.interval // 10))) + i) for i in range(self.input_dim)], axis=-1)
            return history, feats
    
    def __len__(self) -> int:
        return len(self.raindata)
    
    def __getitem__(self, idx):
        history, feats = self.load_images(self.raindata[idx][0][0], self.raindata[idx][0][1])
        row, col, val = zip(*map(lambda x: (self.radius - (x[0] - self.center[0]), self.radius + (x[1] - self.center[1]), x[2]), self.raindata[idx][1]))
        val = [0 if v < 1. else (1 if v < 10 else 2) for v in val]
        where = [(self.raindata[idx][0][1] - 1)]
        
        _year, _month, _day, _timee = int(self.raindata[idx][0][0][0:4]), int(self.raindata[idx][0][0][4:6]), int(self.raindata[idx][0][0][6:8]), (int(self.raindata[idx][0][0][8:10]) * 6 + int(self.raindata[idx][0][0][10]))
        _pii = 3.14159265358979
        _dayy = _day
        for i in range(_month-1-5):
            _dayy += self.days[i]
        timefeat = torch.FloatTensor([math.sin(_dayy / (366. if (_year % 4 == 0) else 365.) * _pii / 180.),
                                      math.cos(_dayy / (366. if (_year % 4 == 0) else 365.) * _pii / 180.),
                                      math.sin(_timee / 144. * _pii  / 180.),
                                      math.cos(_timee / 144. * _pii  / 180.),
                                      1. if (self.raindata[idx][0][1] - 1) == 0 else 0.,
                                      1. if (self.raindata[idx][0][1] - 1) == 1 else 0.,
                                      1. if (self.raindata[idx][0][1] - 1) == 2 else 0.,
                                      1. if (self.raindata[idx][0][1] - 1) == 3 else 0.,
                                      1. if (self.raindata[idx][0][1] - 1) == 4 else 0.,
                                      1. if (self.raindata[idx][0][1] - 1) == 5 else 0.])
        
        return history, feats, torch.LongTensor(row), torch.LongTensor(col), torch.LongTensor(val), torch.LongTensor(where), timefeat
    
    def collate_fn(self, samples):
        historys, feats, rows, cols, vals, wheres, timefeat = zip(*samples)
        historys = torch.stack(historys, dim=0)
        feats = torch.stack(feats, dim=0)
        indices = torch.LongTensor(list(chain.from_iterable([([i] * len(_row)) for i, _row in enumerate(rows)])))
        rows, cols, vals, wheres = map(lambda xs: torch.cat(xs, dim=0), [rows, cols, vals, wheres])
        timefeat = torch.stack(timefeat)
        return historys, feats, indices, rows, cols, vals, wheres, timefeat
    
class AWSOnlyDataset(torch.utils.data.Dataset):
    """ For UnetMultiClassificationV2 """
    def __init__(self, sampled_path, data_path, year_from=2014, year_to=2018, feat_mask = [i for i in range(0,18)], interval=60, min_interval=10, max_interval=360, input_dim=7, center=(1024, 1024), radius=734):
        assert type(year_from) == int and type(year_to) == int and type(input_dim) == int
        assert type(interval) == int and interval % 10 == 0
        
        self.date = []
        self.days = [30, 31, 31, 30]
        self.dates = list(chain.from_iterable([chain.from_iterable([map(str, np.arange(year * 10000 + (month + 1) * 100 + 1,
                                                                                       year * 10000 + (month + 1) * 100 + 1 +
                                                                                       self.days[month-5] + int((month == 1) and (year % 4 == 0))))
                                                                    for month in range(6-1, 9)]) for year in range(year_from, year_to + 1)]))
        self.inv_dates = {v: i for i, v in enumerate(self.dates)}
        self.interval, self.input_dim = interval, input_dim
        self.min_interval, self.max_interval = min_interval, max_interval
        self.num_time_slots = (self.max_interval - self.min_interval) // 10 + 1
        self.center = center
        self.radius = radius
        self.data_path = data_path
        self.feat_mask = feat_mask
        
        with open(sampled_path, "rb") as f:
            raw_raindata = pickle.load(f)
            multi_raindata = list(chain.from_iterable([[((k, i), v) for k, v in raw_raindata] for i in range(1, 6+1)]))
            self.raindata = list(filter(lambda x: (year_from <= int(x[0][0][:4]) <= year_to) and self.load_images(x[0][0], x[0][1], just_check=True), multi_raindata))
            random.shuffle(self.raindata)
    
    def load_feats(self, timestamp):
        def parse_feats(_idx):
            datestr = "%s%02d%1d0" % (self.dates[_idx // 144], (_idx % 144) // 6, _idx % 6)
            with open(f"{self.data_path}/{datestr[:4]}/{datestr[:6]}/ground_{datestr}.pkl", "rb") as f:
                _feat = pickle.load(f)
                return _feat      
        
        return torch.FloatTensor(parse_feats(timestamp)[:,self.feat_mask])
    
    def load_images(self, timestamp, inv, just_check=False):
        def load_image_inner_check(_idx):
            datestr = "%s%02d%1d0" % (self.dates[_idx // 144], (_idx % 144) // 6, _idx % 6)
            return os.path.exists(f'{self.data_path}/{datestr[:6]}/{datestr[:10]}/radar_{datestr}.bin.gz')
        
        date, hour = timestamp[:-4], int(timestamp[-4:-2])
        idx = self.inv_dates[date] * 144 + hour * 6
        
        if just_check:
            valid = True
            for i in range(self.input_dim):
                if not load_image_inner_check(idx - ((self.input_dim - 1) + (inv * (self.interval // 10))) + i):
                    valid = False
                    break
            return valid
        else:
            feats = torch.cat([self.load_feats(idx - ((self.input_dim - 1) + (inv * (self.interval // 10))) + i) for i in range(self.input_dim)], axis=-1)
            return feats
    
    def __len__(self) -> int:
        return len(self.raindata)
    
    def __getitem__(self, idx):
        feats = self.load_images(self.raindata[idx][0][0], self.raindata[idx][0][1])
        row, col, val = zip(*map(lambda x: (self.radius - (x[0] - self.center[0]), self.radius + (x[1] - self.center[1]), x[2]), self.raindata[idx][1]))
        val = [0 if v < 1. else (1 if v < 10 else 2) for v in val]
        where = [(self.raindata[idx][0][1] - 1)]
        
        _year, _month, _day, _timee = int(self.raindata[idx][0][0][0:4]), int(self.raindata[idx][0][0][4:6]), int(self.raindata[idx][0][0][6:8]), (int(self.raindata[idx][0][0][8:10]) * 6 + int(self.raindata[idx][0][0][10]))
        _pii = 3.14159265358979
        _dayy = _day
        for i in range(_month-1-5):
            _dayy += self.days[i]
        timefeat = torch.FloatTensor([math.sin(_dayy / (366. if (_year % 4 == 0) else 365.) * _pii / 180.),
                                      math.cos(_dayy / (366. if (_year % 4 == 0) else 365.) * _pii / 180.),
                                      math.sin(_timee / 144. * _pii / 180.),
                                      math.cos(_timee / 144. * _pii / 180.),
                                      1. if (self.raindata[idx][0][1] - 1) == 0 else 0.,
                                      1. if (self.raindata[idx][0][1] - 1) == 1 else 0.,
                                      1. if (self.raindata[idx][0][1] - 1) == 2 else 0.,
                                      1. if (self.raindata[idx][0][1] - 1) == 3 else 0.,
                                      1. if (self.raindata[idx][0][1] - 1) == 4 else 0.,
                                      1. if (self.raindata[idx][0][1] - 1) == 5 else 0.])
        
        return feats, torch.LongTensor(row), torch.LongTensor(col), torch.LongTensor(val), torch.LongTensor(where), timefeat
    
    def collate_fn(self, samples):
        feats, rows, cols, vals, wheres, timefeat = zip(*samples)
        feats = torch.stack(feats, dim=0)
        indices = torch.LongTensor(list(chain.from_iterable([([i] * len(_row)) for i, _row in enumerate(rows)])))
        rows, cols, vals, wheres = map(lambda xs: torch.cat(xs, dim=0), [rows, cols, vals, wheres])
        timefeat = torch.stack(timefeat)
        return feats, indices, rows, cols, vals, wheres, timefeat