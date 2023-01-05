import argparse
import ast


def get_args_train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ms_loss_scaler', type=str, default='static', help='train loss scaler, static/dynamic/none')
    parser.add_argument('--ms_loss_scaler_value', type=float, default=1024.0, help='static loss scale value')
    parser.add_argument('--ms_optim_loss_scale', type=float, default=1.0, help='optimizer loss scale')
    parser.add_argument('--ms_grad_sens', type=float, default=1024.0, help='gard sens')
    parser.add_argument('--overflow_still_update', type=ast.literal_eval, default=False, help='overflow still update')
    parser.add_argument('--ema', type=ast.literal_eval, default=True, help='ema')
    parser.add_argument('--is_distributed', type=ast.literal_eval, default=True, help='Distribute train or not')
    parser.add_argument('--device_target', type=str, default='Ascend', help='device target, Ascend/GPU/CPU')
    parser.add_argument('--weights', type=str, default='yolov3_backbone.ckpt', help='initial weights path')
    parser.add_argument('--ema_weight', type=str, default='', help='initial ema weights path')
    parser.add_argument('--data_dir', type=str, default='D:\datasets\coco128', help='path to dataset folder')
    parser.add_argument('--cfg', type=str, default='./config/network/yolov3_relu.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='./config/data/coco.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='./config/data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=128, help='total batch size for all device')
    parser.add_argument('--img_size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')

    parser.add_argument('--rect', type=ast.literal_eval, default=False, help='rectangular training')
    parser.add_argument('--evolve', type=ast.literal_eval, default=False, help='evolve hyperparameters')
    parser.add_argument('--cache_images', type=ast.literal_eval, default=False, help='cache images for faster training')
    parser.add_argument('--image_weights', type=ast.literal_eval, default=False, help='use weighted image selection for training')
    parser.add_argument('--multi_scale', type=ast.literal_eval, default=False, help='vary img-size +/- 50%%')
    parser.add_argument('--single_cls', type=ast.literal_eval, default=False, help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, default='momentum', help='select optimizer')
    parser.add_argument('--sync_bn', type=ast.literal_eval, default=False, help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist_ok', type=ast.literal_eval, default=False, help='existing project/name ok, do not increment')
    parser.add_argument('--quad', type=ast.literal_eval, default=False, help='quad dataloader')
    parser.add_argument('--linear_lr', type=ast.literal_eval, default=False, help='linear LR')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers')

    # args for ModelArts
    parser.add_argument('--enable_modelarts', type=ast.literal_eval, default=False, help='enable modelarts')
    parser.add_argument('--file_url', type=str, default='/home/work/user-job-dir/V0001',
                        help='ModelArts: obs path to files')
    parser.add_argument('--data_url', type=str, default='/cache/data/', help='ModelArts: obs path to dataset folder')
    parser.add_argument('--train_url', type=str, default='/home/work/user-job-dir/V0001', help='ModelArts: obs path to dataset folder')
    parser.add_argument('--data_dir_modelarts', type=str, default='/cache/data/', help='ModelArts: obs path to dataset folder')
    parser.add_argument('--ckpt_url', type=str, default='/cache/yolov3_backbone.ckpt', help='model to save/load')
    opt = parser.parse_args()
    return opt


def get_args_test():
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--ms_mode', type=str, default='graph', help='train mode, graph/pynative')
    parser.add_argument('--device_target', type=str, default='GPU', help='device target, Ascend/GPU/CPU')
    parser.add_argument('--weights', type=str, default='EMA_yolov3_300.ckpt', help='model.pt path    (s)')
    parser.add_argument('--data', type=str, default='./config/data/coco.yaml', help='*.data path')
    parser.add_argument('--cfg', type=str, default='./config/network/yolov3.yaml', help='model.yaml path')
    parser.add_argument('--hyp', type=str, default='./config/data/hyp.scratch.yaml', help='hyperparameters p    ath')
    parser.add_argument('--batch_size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--single_cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save_txt', action='store_true', help='save results to *.txt')

    parser.add_argument('--save_hybrid', action='store_true', help='save label+prediction hybrid results to     *.txt')
    parser.add_argument('--save_conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save_json', action='store_true', help='save a cocoapi-compatible JSON results fil    e')
    parser.add_argument('--project', default='./run_test', help='save to project/name')
    parser.add_argument('--exist_ok', action='store_true', help='existing project/name ok, do not increment'    )
    parser.add_argument('--v5_metric', action='store_true', help='assume maximum recall as 1.0 in AP calcula    tion')
    opt = parser.parse_args()
    return opt
