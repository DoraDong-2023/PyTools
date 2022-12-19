"""
author:DZY
date:221219
usage: accelerating tricks
"""

import time
from multiprocessing import Pool
import joblib

file = './blank.txt'
with open(file, 'r') as f:
    lines = f.readlines()

def targetfunc(input):
    # process file
    info = []
    return info

###########
# original
###########
t1 = time.time()
data_infos = []
for line in lines:
    info = targetfunc(line)
    data_infos.append(info)
print(time.time()-t1)
     
###########
# for loop: map
###########
t1 = time.time()
data_infos = map(targetfunc, lines)
data_infos = list(data_infos)
print(time.time()-t1)
     
###########
# for loop: multiprocessing
###########
"""
t1 = time.time()
num_processing_lines = 4
with Pool(num_processing_lines) as p:
    data_infos = p.map(targetfunc, lines)
print(time.time()-t1)
"""

###########
# for loop: p_tqdm
###########
t1 = time.time()
#p_map 并行有序
#p_umap 并行无序
#t_map 串行有序
data_infos = p_map(targetfunc, lines)
print(time.time()-t1)

###########
# for loop: joblib basic
###########
t1 = time.time()
# n_jobs is the number of parallel jobs
joblib.Parallel(n_jobs=2)(joblib.delayed(targetfunc)(line) for line in lines)
print('{:.4f} s'.format(time.time()-t1))

###########
# joblib sklearn
###########
from sklearn.externals import joblib
# joblib 中的 dump 函数用于下载模型
best_est = clf_cv.best_estimator_# 搭配cv使用
joblib.dump(value=best_est, filename='mybest_dt_model.m')
model = joblib.load(filename='mybest_dt_model.m')



###########
# DataLoader : BackgroundGenerator
###########
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __iter__(self):# 线程不会再等待，每个线程都有数据在加载
        return BackgroundGenerator(super().__iter__())

##########
# DataLoader : data_prefetcher:
# https://github.com/NVIDIA/apex/issues/439
# https://www.zhihu.com/column/c_1166381834538835968
###########
# default device change would be in the same stream
# in order to process parallelly the batch data, let pin_memory=True, and load data on another stream
class DataPrefetcher():
    def __init__(self, loader, opt):
        self.loader = iter(loader)
        self.opt = opt
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            for k in self.batch:
                if k != 'meta':
                    self.batch[k] = self.batch[k].to(device=self.opt.device, non_blocking=True)

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

# usage
#for iter_id, batch in enumerate(data_loader):
prefetcher = DataPrefetcher(data_loader, opt)
batch = prefetcher.next()
iter_id = 0
while batch is not None:
    iter_id += 1
    if iter_id >= num_iters:
        break
    main_training()
    batch = prefetcher.next()


#########
# GPU preprocessing data: Nvidia-dali
###########
# GPU are waiting for CPU to process data and load into gpu, which causes wasting of resources
# to use dali, one need to store the file into the format as ImageNet
# https://blog.csdn.net/minstyrain/article/details/99696557


#########
# GPU FP16 加速: apex 加速
# https://zhuanlan.zhihu.com/p/79887894
############
from apex import amp
model, optimizer = amp.initialize(model, optimizer, opt_level="O1") # 这里是“欧一”，不是“零一”
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()


##########
# Small dataset: 数据集挂到内存运行
###########
# linux command
# sudo mount tmpfs /path/to/your/data -t tmpfs -o size=30G


##########
# Larger dataset
#############
# TFRecord：https://zhuanlan.zhihu.com/p/114982658

##########
# 调参
#############
# num_worker=#cpu
# n_jobs=-1

#########
#
############
