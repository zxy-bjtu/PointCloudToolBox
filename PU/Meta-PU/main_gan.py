import os, sys
import time
import math
import socket
import argparse
import importlib
import warnings
import numpy as np
from glob import glob
from tqdm import tqdm
from geomloss import SamplesLoss
import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, './model'))
import data_loader as data_loader
import data_utils as d_utils
import networks as MODEL_GEN


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='test', help='train or test [default: test]')
parser.add_argument('--gpu', default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model/networks', help='Model name [default: networks]')
parser.add_argument('--log_dir', default='models/logs', help='Log dir')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [1024] [default: 1024]')
parser.add_argument('--up_ratio', type=int, default=4, help='Upsampling Ratio [default: 4]')
parser.add_argument('--max_epoch', type=int, default=80, help='Epoch to run [default: 80]')
parser.add_argument('--batch_size', type=int, default=24, help='Batch Size during training [default: 24]')
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--dataset', default=None)
parser.add_argument('--gan', default=False, action='store_true')
parser.add_argument('--model_path', type=int, default=0, help='The num of epoch to restore the models from')
parser.add_argument('--lambd', default=10000, type=float)
parser.add_argument('--max_sinkhorn_iters', default=32, help="Maximum number of Sinkhorn iterations")
parser.add_argument('--FWWD', default=False, action='store_true',
                    help="move WD loss cal in g forward (for memory balance)")

parser.add_argument('--replace', default=False, action='store_true')
parser.add_argument('--nowarmup', default=False, action='store_true')
parser.add_argument('--test_scale', type=float, default=4, help='up ratio during testing [default: 4]')
parser.add_argument('--num_workers_each_gpu', type=int, default=4, help='[default: 4]')

USE_DATA_NORM = True
USE_RANDOM_INPUT = True
ASSIGN_MODEL_PATH = 0

FLAGS = parser.parse_args()
PHASE = FLAGS.phase
GPU_INDEX = FLAGS.gpu
MODEL_DIR = FLAGS.log_dir
RESTORE_MODEL_DIR = FLAGS.log_dir
NUM_POINT = FLAGS.num_point
UP_RATIO = FLAGS.up_ratio
MAX_EPOCH = FLAGS.max_epoch
BATCH_SIZE = FLAGS.batch_size
BASE_LEARNING_RATE = FLAGS.learning_rate
ASSIGN_MODEL_PATH = FLAGS.model_path
max_sinkhorn_iters = FLAGS.max_sinkhorn_iters
Replace = FLAGS.replace

print(socket.gethostname())
print(FLAGS)
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_INDEX

    
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim

device = torch.device('cuda:{}'.format(int(GPU_INDEX)) if torch.cuda.is_available() else 'cpu')
print('device: {}'.format(device))
if ASSIGN_MODEL_PATH > 0:
    ori_dir = MODEL_DIR
    MODEL_DIR = os.path.join(MODEL_DIR, 'models_{}'.format(ASSIGN_MODEL_PATH))
  

def log_string(LOG_FOUT, out_str):
    LOG_FOUT.write(out_str)
    LOG_FOUT.flush()

def weight_init(m):          
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
    

def weight_clip(m):
    if hasattr(m, 'weight'):
        m.weight.data.clamp_(-0.01, 0.01)

def load_checkpoint(model, optimizer, fc_optimizer=None, name='g'):
    _file = os.path.join(RESTORE_MODEL_DIR, '{}_model_{}.pth'.format(name, ASSIGN_MODEL_PATH))
    print("=> loading checkpoint '{}'".format(_file))
    if os.path.isfile(_file):
        try:
            checkpoint = torch.load(_file)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print('model loaded...')
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
            if name == 'g':
                fc_optimizer.load_state_dict(checkpoint['fc_optimizer'])
                for state in fc_optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
                print('fc optimizer loaded...')
            print('optimizer loaded...')
            
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(_file, checkpoint['epoch']))
        except:
            try:
                checkpoint = torch.load(_file)
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    
                    name = k.replace("module.", "")  
                    
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
                print('model loaded...')
                optimizer.load_state_dict(checkpoint['optimizer'])
                if name == 'g':
                    fc_optimizer.load_state_dict(checkpoint['fc_optimizer'])
                    for state in fc_optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda()
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
                print('optimizer loaded...')
                start_epoch = checkpoint['epoch']
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(_file, checkpoint['epoch']))
            except:
                print('load model error')
    else:
        print("=> no checkpoint found at '{}'".format(_file))

    return model, optimizer, fc_optimizer

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id + int(time.time()))



def train(assign_model_path=None, bn_decay=0.95):
    torch.backends.cudnn.benchmark = False  
    learning_rate = BASE_LEARNING_RATE

    try:
        os.makedirs(MODEL_DIR)
        os.makedirs(os.path.join(MODEL_DIR, 'train'))
    except:
        print('log dir is not empty!!!!!!!')

    log_writer = SummaryWriter()
    g_learning_rate = learning_rate
    LDset = data_loader.LoaderDataset(FLAGS.dataset, BATCH_SIZE, USE_DATA_NORM, [0], transforms=None)

    g_model = MODEL_GEN.GenModel(use_normal=False, use_bn=False, use_ibn=False, bn_decay=bn_decay, up_ratio=UP_RATIO,
                                 multi_gpus=False, device=device)
    g_model.train()

    fc_parameters = list(filter(lambda kv: 'fc' in kv[0], g_model.named_parameters()))
    conv_parameters = list(filter(lambda kv: ('conv' in kv[0] and 'fc' not in kv[0]), g_model.named_parameters()))
    
    others_para = list(filter(lambda kv: ('conv' not in kv[0] and 'fc' not in kv[0]), g_model.named_parameters()))
    
    fc_parameters = list(fc[1] for fc in fc_parameters)
    conv_parameters = list(con[1] for con in conv_parameters)
    others_para = list(p[1] for p in others_para)

    
    parameters = [{'params': conv_parameters, 'lr': g_learning_rate, 'weight_decay': (1e-5)}, {'params': others_para}]

    g_optimizer = optim.Adam(parameters, lr=g_learning_rate, betas=(0.9, 0.999))
    g_fc_optimizer = optim.Adam(fc_parameters, lr=g_learning_rate * 10, weight_decay=1e-5, betas=(0.9, 0.999))
    g_model.apply(weight_init)


    g_model = g_model.to(device)
        
    restore_epoch = 0
    MAX_EPOCH = FLAGS.max_epoch
    if ASSIGN_MODEL_PATH > 0:
        restore_epoch += ASSIGN_MODEL_PATH
        MAX_EPOCH += ASSIGN_MODEL_PATH
        print("Load pre-train model from %s" % ASSIGN_MODEL_PATH)
        g_model, g_optimizer, g_fc_optimizer = load_checkpoint(g_model, g_optimizer, g_fc_optimizer, 'g')

    WD = SamplesLoss(loss='sinkhorn', p=2, blur=.001, reach=.2)
    g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=FLAGS.max_epoch,
                                                                eta_min=1e-6)  
    g_fc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(g_fc_optimizer, T_max=FLAGS.max_epoch,
                                                                eta_min=1e-5)

    dloader = torch.utils.data.DataLoader(LDset, batch_size=1, shuffle=False, pin_memory=True,
                                          num_workers=FLAGS.num_workers_each_gpu,
                                          worker_init_fn=worker_init_fn,
                                          drop_last=True)  
    for epoch in tqdm(range(restore_epoch, MAX_EPOCH + 1), ncols=55):
        g_scheduler.step()
        g_fc_scheduler.step()
        torch.cuda.empty_cache()
        np.random.seed(int(time.time()))
        for i_batch, sample_batched in enumerate(tqdm(dloader)):
            pointclouds_pl, pointclouds_gt, radius, this_scale = sample_batched
            this_scale = this_scale[0].numpy() - (1e-05)  
            pointclouds_pl, pointclouds_gt, radius = pointclouds_pl[0].cuda(non_blocking=True), pointclouds_gt[0].cuda(
                non_blocking=True), radius[0].cuda(non_blocking=True)
            
            assert pointclouds_pl.shape[0] > 0 and pointclouds_pl.shape[1] > 10
            pointclouds_gt = pointclouds_gt[:, :, 0:3]
            g_optimizer.zero_grad()
            g_fc_optimizer.zero_grad()

            if not FLAGS.FWWD:
                pred, _, reg_loss, uniform_loss, repulsion_loss = g_model(pointclouds_pl)
            else:
                pred, _, WD_loss, reg_loss, uniform_loss, repulsion_loss, _ = g_model(pointclouds_pl, WD,
                                                                                      pointclouds_gt,
                                                                                      this_scale=this_scale)
                WD_loss = WD_loss / radius
                del radius
                WD_loss = torch.mean(WD_loss, dim=0, keepdim=True)
                if WD_loss.shape.__len__ != 1 and WD_loss.shape[0] != 1:
                    print('WD_loss wrong shape!!!!!!!!!!!!!!!!!!!!!!!!!')
                try:
                    uniform_loss = torch.mean(uniform_loss, dim=0, keepdim=True)
                    repulsion_loss = torch.mean(repulsion_loss, dim=0, keepdim=True)
                except:
                    pass
                if reg_loss.shape[0] != 1:
                    reg_loss = reg_loss.mean()
 
            pre_gen_loss = WD_loss + reg_loss * (1e-3) + uniform_loss * (1e-3) + repulsion_loss * (5e-3)
            pre_gen_loss.backward()

            g_optimizer.step()
            g_fc_optimizer.step()

            n_iter = i_batch + epoch * LDset.num_batches

            log_writer.add_scalar('CDLoss/train', WD_loss.cpu().detach(), n_iter)
            log_writer.add_scalar('UniLoss/train', uniform_loss.cpu().detach(), n_iter)

        
        if epoch % 10 == 0 and epoch > restore_epoch:
            print('saving model!')
            state = {'epoch': epoch + 1, 'state_dict': g_model.state_dict(),
                     'optimizer': g_optimizer.state_dict(), 'fc_optimizer': g_fc_optimizer.state_dict()}
            torch.save(state, os.path.join(MODEL_DIR, 'g_model_{}.pth'.format(epoch)))


def prediction_whole_model(use_normal=False, this_scale=4):
    torch.backends.cudnn.benchmark = True
    BATCH_SIZE = 1
    multi_gpus = False
    if ',' in GPU_INDEX:
        print('dont use multi gpu!!!!!!!')
        gpu_ids = [int(id) for id in GPU_INDEX.split(',')]
        multi_gpus = True
    else:
        gpu_ids = [int(GPU_INDEX)]   

    device = torch.device('cuda:{}'.format(gpu_ids[0]) if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))

    data_folder = FLAGS.dataset
    phase = data_folder.split('/')[-2] + data_folder.split('/')[-1]
    save_path = os.path.join(ori_dir, 'result/' + phase)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    samples = glob(data_folder + "/*.xyz")
    samples.sort(reverse=True)
    print('in',data_folder,'num of samples: ',len(samples))

    g_model = MODEL_GEN.GenModel(use_normal=False, use_bn=False, use_ibn=False, bn_decay=0.95, up_ratio=this_scale,
                                 device=device, training=False)
    g_model.eval()

    if multi_gpus:
        g_model = torch.nn.DataParallel(g_model, device_ids=gpu_ids).to(device)
    else:
        g_model = g_model.to(device)

    print('loading models...')
    try:
        print(os.path.join(ori_dir, 'g_model_{}.pth'.format(ASSIGN_MODEL_PATH)))
        dic = torch.load(os.path.join(ori_dir, 'g_model_{}.pth'.format(ASSIGN_MODEL_PATH)))
        g_model.load_state_dict(dic['state_dict'])
    except:
        try:
            weight = torch.load(os.path.join(ori_dir, 'g_model_{}.pth'.format(ASSIGN_MODEL_PATH)),
                                map_location=lambda storage, loc: storage)
            
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in weight['state_dict'].items():
                
                name = k.replace("module.", "")  
                
                new_state_dict[name] = v
            g_model.load_state_dict(new_state_dict)
        except:
            print('load model error')

    total_time = 0
    for i, item in enumerate(samples):
        input = np.loadtxt(item)
        input = np.expand_dims(input, axis=0)
        if not use_normal:
            input = input[:, :, 0:3]
        print(item, input.shape)
        with torch.no_grad():
            input_torch = torch.from_numpy(input).type(torch.cuda.FloatTensor).detach().to(device)
            beg = time.time()
            pred, _, _, _, _ = g_model(input_torch, this_scale=this_scale)
            end = time.time()
            pred = pred.detach().cpu()
            path = os.path.join(save_path, item.split('/')[-1])
            if use_normal:
                norm_pl = np.zeros_like(pred)
                data_loader.save_pl(path, np.hstack((pred[0, ...], norm_pl[0, ...])))
            else:
                data_loader.save_pl(path, pred[0, ...])
            path = path[:-4] + '_input.xyz'
            data_loader.save_pl(path, input[0])
        total_time += (end - beg)
    print('total time is: {}'.format(total_time))


if __name__ == "__main__":
    if Replace == True:
        try:
            import shutil
            shutil.rmtree(os.path.join(MODEL_DIR, 'code/'))
        except:
            pass
    if PHASE == 'train':
        assert not os.path.exists(os.path.join(MODEL_DIR, 'code/'))
        os.makedirs(os.path.join(MODEL_DIR, 'code/'))
        train(assign_model_path=ASSIGN_MODEL_PATH)
    else:
        prediction_whole_model(this_scale=FLAGS.test_scale)