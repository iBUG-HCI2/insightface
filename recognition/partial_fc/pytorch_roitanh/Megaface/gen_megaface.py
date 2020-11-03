import os
import sys
import os.path as osp
work_dir = osp.abspath(__file__)
work_dir = osp.abspath(osp.join(work_dir, '..', '..'))
sys.path.append(work_dir)
from backbones.iresnet import iresnet50,iresnet100

from easydict import EasyDict as edict
import time
import sys
import numpy as np
import argparse
import struct
import cv2
import sklearn
from sklearn.preprocessing import normalize
import torch 
import mxnet as mx
from mxnet import ndarray as nd



def read_img(image_path):
  img = cv2.imread(image_path)
  return img

def get_feature(imgs, nets):
  count = len(imgs)
  data = np.zeros((count*2, 3, imgs[0].shape[0], imgs[0].shape[1]), dtype=np.uint8)
  for idx, img in enumerate(imgs):
    img = img[:,:,::-1] #to rgb
    img = np.transpose( img, (2,0,1) )
    for flipid in [0,1]:
      _img = np.copy(img)
      if flipid==1:
        _img = _img[:,:,::-1]
      data[count*flipid+idx] = _img

  F = []
  for net in nets:
    # db = mx.io.DataBatch(data=(data,))
    net.eval()
    imgs=torch.Tensor(data).cuda()
    imgs.div_(255).sub_(0.5).div_(0.5)
    with torch.no_grad():
        x = net(imgs).cpu().numpy()
    # feat = feat.reshape([self.batch_size, 2 * feat.shape[1]]).cpu().numpy()
    # net.model.forward(db, is_train=False)
    # x = net.model.get_outputs()[0].asnumpy()
    embedding = x[0:count,:] + x[count:,:]
    embedding = sklearn.preprocessing.normalize(embedding)
    #print('emb', embedding.shape)
    F.append(embedding)
  F = np.concatenate(F, axis=1)
  F = sklearn.preprocessing.normalize(F)
  #print('F', F.shape)
  return F


def write_bin(path, feature):
  feature = list(feature)
  with open(path, 'wb') as f:
    f.write(struct.pack('4i', len(feature),1,4,5))
    f.write(struct.pack("%df"%len(feature), *feature))

def get_and_write(buffer, nets):
  imgs = []
  for k in buffer:
    imgs.append(k[0])
  features = get_feature(imgs, nets)
  #print(np.linalg.norm(feature))
  assert features.shape[0]==len(buffer)
  for ik,k in enumerate(buffer):
    out_path = k[1]
    feature = features[ik].flatten()
    write_bin(out_path, feature)

def main(args):

  print(args)
  gpuid = args.gpu
  ctx = mx.gpu(gpuid)
  nets = []
  image_shape = [int(x) for x in args.image_size.split(',')]
  for model in args.model.split('|'):
    vec = model.split(',')
    assert len(vec)>1
    prefix = vec[0]
    epoch = int(vec[1])
    weight = torch.load(prefix)
    print('loading',prefix, epoch)
    net = iresnet100().cuda()
    net.load_state_dict(weight)
    nets.append(net)

  facescrub_out = os.path.join(args.output, 'facescrub')
  megaface_out = os.path.join(args.output, 'megaface')

  i = 0
  succ = 0
  buffer = []
  for line in open(args.facescrub_lst, 'r'):
    if i%1000==0:
      print("writing fs",i, succ)
    i+=1
    image_path = line.strip()
    _path = image_path.split('/')
    a,b = _path[-2], _path[-1]
    out_dir = os.path.join(facescrub_out, a)
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)
    image_path = os.path.join(args.facescrub_root, image_path)
    img = read_img(image_path)
    if img is None:
      print('read error:', image_path)
      continue
    out_path = os.path.join(out_dir, b+"_%s.bin"%(args.algo))
    item = (img, out_path)
    buffer.append(item)
    if len(buffer)==args.batch_size:
      get_and_write(buffer, nets)
      buffer = []
    succ+=1
  if len(buffer)>0:
    get_and_write(buffer, nets)
    buffer = []
  print('fs stat',i, succ)

  i = 0
  succ = 0
  buffer = []
  for line in open(args.megaface_lst, 'r'):
    if i%1000==0:
      print("writing mf",i, succ)
    i+=1
    image_path = line.strip()
    _path = image_path.split('/')
    a1, a2, b = _path[-3], _path[-2], _path[-1]
    out_dir = os.path.join(megaface_out, a1, a2)
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)
      #continue
    #print(landmark)
    image_path = os.path.join(args.megaface_root, image_path)
    img = read_img(image_path)
    if img is None:
      print('read error:', image_path)
      continue
    out_path = os.path.join(out_dir, b+"_%s.bin"%(args.algo))
    item = (img, out_path)
    buffer.append(item)
    if len(buffer)==args.batch_size:
      get_and_write(buffer, nets)
      buffer = []
    succ+=1
  if len(buffer)>0:
    get_and_write(buffer, nets)
    buffer = []
  print('mf stat',i, succ)


def parse_arguments(argv):
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--batch_size', type=int, help='', default=8)
  parser.add_argument('--image_size', type=str, help='', default='3,112,112')
  parser.add_argument('--gpu', type=int, help='', default=0)
  parser.add_argument('--algo', type=str, help='', default='insightface')
  parser.add_argument('--facescrub-lst', type=str, help='', default='./data/facescrub_lst')
  parser.add_argument('--megaface-lst', type=str, help='', default='./data/megaface_lst')
  parser.add_argument('--facescrub-root', type=str, help='', default='./data/facescrub_images')
  parser.add_argument('--megaface-root', type=str, help='', default='./data/megaface_images')
  parser.add_argument('--output', type=str, help='', default='./feature_out')
  parser.add_argument('--model', type=str, help='', default='')
  return parser.parse_args(argv)

if __name__ == '__main__':
  main(parse_arguments(sys.argv[1:]))

