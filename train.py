import argparse
import time
import numpy as np
import random
import paddle
import paddle.optimizer as optim
from paddle.io import DataLoader
from model import *
from mydataloader import *
# from mydataloader_clsaware import *
from utils import *
from loss import *
from metrics import *
from config import *
from utils import GetAttriPool


#---------------------------------------------------#
#   获得检测任务中的类别数
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def fit_one_epoch(net, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, config):
    total_r_loss = 0
    total_c_loss = 0
    total_a_loss = 0 # attris
    total_loss = 0
    val_loss = 0

    net.train()
    print('Start Training')
    for iteration, batch in enumerate(gen()):
        if iteration % (config.print_interval) == 0:
            print(time.strftime('%Y-%m-%d %H:%M:%S'), f'Epoch: {epoch}, Iter: {iteration}/{epoch_size}')
        if iteration >= epoch_size:
            break
        with paddle.no_grad():
            batch = [paddle.to_tensor(ann, dtype='float32') for ann in batch[:-1]]

        batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, attris_label = batch

        optimizer.clear_grad()

        detection_output, attris_output = net(batch_images)
        #print(attris_output)
        hm, wh, offset = detection_output
        c_loss = focal_loss(hm, batch_hms)
        wh_loss = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
        off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)
        attri_loss = BCELoss(attris_output, attris_label)

        if config['attri_loss'] == 'focal':
            attri_loss = 10 * attri_focal_loss(attris_output, attris_label, config['n_class'])
        elif config['attri_loss'] == 'bce':
            attri_loss = BCELoss(attris_output, attris_label)
        if config['attri_loss'] == 'focal+bce':
            attri_loss = 10 * attri_focal_loss(attris_output, attris_label, config['n_class']) + BCELoss(attris_output, attris_label)
        else:
            assert('config error: attri loss')
        
        if config['task'] == 'cls_det':
            loss = 0.1 * c_loss + wh_loss + off_loss + attri_loss
        else:
            loss = attri_loss

        total_loss += loss.numpy()
        total_a_loss += attri_loss.numpy()
        total_c_loss += c_loss.numpy()
        total_r_loss += wh_loss.numpy() + off_loss.numpy()
        
        loss.backward()
        optimizer.step()
        
    print(f'total_loss:{total_loss / (iteration + 1)}, total_a_loss:{total_a_loss / (iteration + 1)}, total_r_loss:{total_r_loss / (iteration + 1)}, total_c_loss:{total_c_loss / (iteration + 1)}')


    net.eval()
    print('Start Validation')
    all_label, all_pred, imgnames = [], [], []
    for iteration, batch in enumerate(genval()):
        if iteration % (config.print_interval) == 0:
            print(time.strftime('%Y-%m-%d %H:%M:%S'), f'Epoch: {epoch}, Iter: {iteration}/{epoch_size_val}')
        if iteration >= epoch_size_val:
            break
        with paddle.no_grad():
            batch = [paddle.to_tensor(ann, dtype='float32') for ann in batch[:-1]] + [batch[-1]]

            batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, attris_label, imgname = batch

            imgnames += imgname

            detection_output, attris_output = net(batch_images)
            hm, wh, offset = detection_output
            c_loss = focal_loss(hm, batch_hms)
            wh_loss = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
            off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)
            attri_loss = BCELoss(attris_output, attris_label)
            if config['task'] == 'cls_det':
                loss = 0.1 * c_loss + wh_loss + off_loss + attri_loss
            else:
                loss = attri_loss
            val_loss += loss.numpy()

            all_label.append(attris_label.numpy())
            all_pred.append(attris_output.numpy())

            
    print('Finish Validation')
    print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))
    if (epoch+1) % config['save_interval'] == 0:  # 每5个epoch保存一次模型
        save_dir_name = config['save_dir']
        if not os.path.exists(f'work/{save_dir_name}'):
            os.makedirs(f'work/{save_dir_name}')
        print('Saving state, iter:', str(epoch + 1))

        # 动态图模型
        pdparams_save_path = f'work/{save_dir_name}/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pdparams'%((epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1))
        paddle.save(net.state_dict(), pdparams_save_path)
        pdopt_save_path = f'work/{save_dir_name}/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pdopt'%((epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1))
        paddle.save(optimizer.state_dict(), pdopt_save_path)

        # #保存静态图模型, 用于部署
        # input_spec = paddle.static.InputSpec(shape=[None, 3, 256, 128], name='img') # 定制化预测模型导出
        # model = paddle.jit.to_static(net, input_spec=[input_spec])
        # paddle.jit.save(model, f"work/{save_dir_name}/Epoch%d-Total_Loss%.4f-Val_Loss%.4f"%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    
    all_label = np.concatenate(all_label, 0)
    all_pred = np.concatenate(all_pred, 0)
    attris_eval(all_label, all_pred, imgnames)

    return val_loss / (epoch_size_val+1)
    

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    paddle.seed(seed)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('-y', '--yaml', type=str, help='path of .yaml')
    args = parse.parse_args()
    config = get_config(args.yaml)

    # fix seed
    set_seed(config['seed'])

    # paddle.set_device("gpu:0")
    # paddle.set_device("cpu")
    input_shape = config['input_shape']
    classes_path = config['classes_path']
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    pretrain = config['pretrained']

    model = CenterNet_ResNet(num_classes, pretrain=pretrain, config=config)

    # 断点继续训练.....
    if config['continue_train']:
        print('Loading weights into state dict...')
        model_path = config['last_params']
        # dynamic
        pretrained_dict = paddle.load(model_path + '.pdparams')
        # # static
        # pretrained_dict = paddle.load(model_path)
        # for k in pretrained_dict:
        #     print(k)
        model.load_dict(pretrained_dict)
        print('Finished!')

    model.train()

    annotation_train_path = config['annotation_train_path']
    annotation_val_path = config['annotation_val_path']

    # 训练集
    with open(annotation_train_path) as f:
        train_lines = f.readlines()
    np.random.shuffle(train_lines)
    num_train = int(len(train_lines))

    # 验证集
    with open(annotation_val_path) as f:
        val_lines = f.readlines()
    np.random.shuffle(val_lines)
    num_val = int(len(val_lines))

    
    if True:
        lr = config['lr_freeze']
        Batch_size = config['batch_size']
        Init_Epoch = config['Init_Epoch']
        Freeze_Epoch = config['Freeze_Epoch']

        if config['lr_s'] == 'ReduceOnPlateau':
            lr_scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=lr, factor=0.5, patience=300, verbose=True)
        elif config['lr_s'] == 'CosineAnnealingDecay':
            lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=lr, eta_min=lr/10, T_max=Freeze_Epoch, verbose=True)
        else:
            assert('config error: lr_s')
        
        if config['optim'] == 'Momentum':
            optimizer = optim.Momentum(parameters=model.parameters(),
                        learning_rate=lr_scheduler, momentum=0.9, weight_decay=config['weight_decay'], use_nesterov=True)
        elif config['optim'] == 'SGD':
            optimizer = optim.SGD(parameters=model.parameters(),learning_rate=lr_scheduler,
                              weight_decay=config['weight_decay'])
        else:
            assert('config error: optim')

        # gen = dataloader(train_lines, input_shape, num_classes, False, Batch_size, config)
        # gen_val = dataloader(val_lines, input_shape, num_classes, False, Batch_size, config)

        # paddle.io.dataloader 方式读取数据
        train_dataset = CenternetDataset(train_lines, input_shape, num_classes, False, config)
        val_dataset = CenternetDataset(val_lines, input_shape, num_classes, False, config)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=config['num_workers'],
                                drop_last=True, collate_fn=centernet_dataset_collate,
                                use_shared_memory=False)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=config['num_workers'], 
                                drop_last=True, collate_fn=centernet_dataset_collate,
                                use_shared_memory=False)

        epoch_size = num_train//Batch_size
        epoch_size_val = num_val//Batch_size
        #------------------------------------#
        #   冻结backbone训练
        #------------------------------------#
        model.freeze_backbone()

        for epoch in range(Init_Epoch, Freeze_Epoch):
            val_loss = fit_one_epoch(model, epoch,epoch_size, epoch_size_val, gen, gen_val, Freeze_Epoch, config)
            lr_scheduler.step(val_loss)
    
    
    if True:
        lr = config['lr_unfreeze']
        Batch_size = config['batch_size']  # 解冻后batch_size要比之前冻结时小一些
        Freeze_Epoch = config['Freeze_Epoch']
        Unfreeze_Epoch = config['Unfreeze_Epoch']

        if config['lr_s'] == 'ReduceOnPlateau':
            lr_scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=lr, factor=0.5, patience=300, verbose=True)
        elif config['lr_s'] == 'CosineAnnealingDecay':
            lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=lr, eta_min=lr/10, T_max=Unfreeze_Epoch, verbose=True)
        else:
            assert('config error: lr_s')

        ignored_params = list(map(id, model.backbone.parameters()))
        classifier_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

        if config['optim'] == 'Momentum':
            optimizer = optim.Momentum(parameters=model.parameters(),
                        learning_rate=lr_scheduler, momentum=0.9, weight_decay=config['weight_decay'], use_nesterov=True)
        elif config['optim'] == 'SGD':
            optimizer = optim.SGD(parameters=model.parameters(),learning_rate=lr_scheduler,
                              weight_decay=config['weight_decay'])
        else:
            assert('config error: optim')

        # gen = dataloader(train_lines, input_shape, num_classes, False, Batch_size, config)
        # gen_val = dataloader(val_lines, input_shape, num_classes, False, Batch_size, config)

        # paddle.io.dataloader 方式读取数据
        train_dataset = CenternetDataset(train_lines, input_shape, num_classes, False, config)
        val_dataset = CenternetDataset(val_lines, input_shape, num_classes, False, config)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=config['num_workers'],
                                drop_last=True, collate_fn=centernet_dataset_collate,
                                use_shared_memory=False)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=config['num_workers'], 
                                drop_last=True, collate_fn=centernet_dataset_collate,
                                use_shared_memory=False)

        epoch_size = num_train//Batch_size
        epoch_size_val = num_val//Batch_size
        #------------------------------------#
        #   解冻backbone后训练
        #------------------------------------#
        model.unfreeze_backbone()

        for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
            val_loss = fit_one_epoch(model, epoch, epoch_size, epoch_size_val, gen, gen_val, Unfreeze_Epoch, config)
            lr_scheduler.step(val_loss)