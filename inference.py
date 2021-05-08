from torch.utils import data
from dataset import get_dataset
from CDCNN import get_model
from train import test
from utils import get_device, metrics, logger
import argparse
import numpy as np
import warnings
import torch
import datetime
# 忽略警告
warnings.filterwarnings("ignore")
# 生成日志
file_date = datetime.datetime.now().strftime('%Y-%m-%d')
log_date = datetime.datetime.now().strftime('%Y-%m-%d:%H:%M')
logger = logger('./logs/infer_logs-'+file_date+'.txt')
logger.info("---------------------------------------------------------------------")
logger.info("-----------------------------Next infer log----------------------------")
logger.info("---------------------------{}--------------------------".format(log_date))
logger.info("---------------------------------------------------------------------")
# Test options
parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                             " various hyperspectral datasets")
parser.add_argument('--model', type=str, default='MCSSN',
                    help="Model to train.")
parser.add_argument('--cuda', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")
# Testing options
group_test = parser.add_argument_group('Test')
group_test.add_argument('--test_stride', type=int, default=1,
                        help="Sliding window step stride during inference (default = 1)")
group_test.add_argument('--restore_folder', type=str, default='./checkpoints/', nargs='?',
                        help="The folder where the model is stored")
group_test.add_argument('--checkpoint', type=str, default=None,
                        help="Choose the weight of the model to load")

# Dataset options
group_dataset = parser.add_argument_group('Dataset')
group_dataset.add_argument('--folder', type=str, default='./dataset/', nargs='?',
                           help="Folder where all samples are located.")
group_dataset.add_argument('--dataset', type=str, default=None, nargs='?',
                           help="Path to an image on which to run inference.")
group_dataset.add_argument('--patch_size', type=int,
                           help="Size of the spatial neighbourhood (optional, if "
                           "absent will be set by the model)")
group_dataset.add_argument('--batch_size', type=int,
                           help="Batch size (optional, if absent will be set by the model")
# 对参数进行解析
args = parser.parse_args()
# 获取设备名称
CUDA_DEVICE = get_device(logger, args.cuda)
# 获取模型名称
MODEL = args.model
# 获取数据集名称
FOLDER = args.folder
# Dataset name
DATASET = args.dataset
# 加载数据集
img, gt, LABEL_VALUES, IGNORED_LABELS = get_dataset(logger, DATASET, FOLDER)
# Number of classes
N_CLASSES = len(LABEL_VALUES)
# Number of spectral bands
N_BANDS = img.shape[-1]
# 指定滑动窗口步长
TEST_STRIDE = args.test_stride
# 指定权重
RESTORE_FOLDER = args.restore_folder
CHECKPOINT = RESTORE_FOLDER + args.checkpoint
hyperparams = vars(args)
hyperparams.update({'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'device': CUDA_DEVICE, 'ignored_labels': IGNORED_LABELS})
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

logger.info('----------Model parameters----------')

for k,v in hyperparams.items():
	logger.info("{}:{}".format(k,v))

model, _, _, hyperparams = get_model(**hyperparams)
model.load_state_dict(torch.load(CHECKPOINT))
probabilities = test(model, img, hyperparams)
prediction = np.argmax(probabilities, axis=-1)

results = metrics(prediction, gt, ignored_labels=hyperparams['ignored_labels'], n_classes=N_CLASSES)
mask = np.zeros(gt.shape, dtype='bool')
for l in IGNORED_LABELS:
    mask[gt == l] = True
prediction[mask] = 0

logger.info('The network inference successfully!!!')

logger.info('----------Inference result----------')
logger.info("\nConfusion matrix:\n{}".format(results['Confusion matrix']))
logger.info("\nAccuracy:\n{}".format(results['Accuracy']))
logger.info("\nF1 scores:\n{}".format(results['F1 scores']))
logger.info("\nKappa:\n{}".format(results['Kappa']))