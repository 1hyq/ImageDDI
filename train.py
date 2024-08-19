import os
import torch
import pickle
import argparse
import torch.optim

import numpy as np

import torch.nn.parallel

import torch.utils.data
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from dataloader.ddi_dataloader import DDIDataset, get_datasets
from model.model_utils import load_model, train_one_epoch_multitask, evaluate_on_multitask, save_finetune_ckpt
from model.train_utils import fix_train_random_seed
from utils.public_utils import cal_torch_model_params, setup_device
from model_ddi import MolDDI
import warnings

warnings.filterwarnings("ignore")

torch.manual_seed(6789)
np.random.seed(6789)
torch.cuda.manual_seed_all(6789)
os.environ['PYTHONHASHSEED'] = str(6789)


def load_norm_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_tra = [transforms.CenterCrop(args.imageSize),
               transforms.RandomHorizontalFlip(),
               transforms.RandomGrayscale(p=0.2),
               transforms.RandomRotation(degrees=360)]
    tile_tra = [transforms.RandomHorizontalFlip(),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomRotation(degrees=360),
                transforms.ToTensor()]
    return normalize, img_tra, tile_tra


# 保存特征和标签
def save_features(features, labels, filename):
    with open(filename, 'wb') as f:
        pickle.dump({'features': features, 'labels': labels}, f)


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of ImageMol')

    # basic
    parser.add_argument('--dataset', type=str, default="Deng's_dataset", help='dataset name')
    parser.add_argument('--dataroot', type=str, default="./dataset/", help='data root')
    parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use (default: 1)')
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 2)')
    # optimizer
    parser.add_argument('--lr', default=0.00025, type=float, help='learning rate (default: 0.01)')
    # parser.add_argument('--lr', type=float, default="0.001,0.0015,0.002,0.0025",
    #                     help='learning rates for grid search, separated by commas')

    parser.add_argument('--weight_decay', default="-6", type=float, help='weight decay pow (default: -5)')
    parser.add_argument('--momentum', default=0.9, type=float, help='moment um (default: 0.9)')

    # train
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42) to split dataset')
    parser.add_argument('--runseed', type=int, default=2021, help='random seed to run model (default: 2021)')
    parser.add_argument('--epochs', type=int, default="100", help='number of total epochs to run (default: 100)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=128, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--resume', default='ckpts/ImageMol.pth.tar', type=str, metavar='PATH',
                        help='path to checkpoint (default: None)')
    parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
    parser.add_argument('--image_model', type=str, default="ResNet18", help='e.g. ResNet18, ResNet34')
    parser.add_argument('--num_classes', type=int, default="65")
    parser.add_argument('--hidden_dim', type=int, default="512")
    parser.add_argument('--vacb_size', type=int, default="5456", help='number of motif')
    parser.add_argument('--num_cliques_types', type=int, default="2192", help='number of cliques')
    parser.add_argument('--max_distance_dim', type=int, default="64", help='max_distance_dim')
    parser.add_argument('--max_motif_len', type=int, default="8", help='length of max_motif')

    parser.add_argument('--save_finetune_ckpt', type=int, default=1, choices=[0, 1],
                        help='1 represents saving best ckpt, 0 represents no saving best ckpt')
    parser.add_argument('--task_type', type=str, default="classification", choices=["classification", "regression"],
                        help='task type')
    # log
    parser.add_argument('--log_dir', default='./logs/train/', help='path to log')

    return parser.parse_args()

def get_file_path(dataroot, dataset):
    '''

    :param dataroot: './dataset'
    :param dataset:  Deng's_dataset'
    :return:
    '''
    train_type_file  = os.path.join(dataroot,dataset,'4/ddi_training1xiao.csv')
    val_type_file = os.path.join(dataroot,dataset,'4/ddi_validation1xiao.csv')
    test_type_file = os.path.join(dataroot, dataset, '4/ddi_test1xiao.csv')
    drug_smiles = os.path.join(dataroot,dataset,'drug_smiles.csv')
    motif_seq_file = os.path.join(dataroot,"motif_vacb.csv")

    assert (os.path.exists(train_type_file) and os.path.exists(val_type_file)
            and os.path.exists(test_type_file) and os.path.exists(motif_seq_file) and os.path.exists(drug_smiles))

    return train_type_file, val_type_file, test_type_file, drug_smiles, motif_seq_file



def main(args):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args.image_folder, args.txt_file = get_datasets(args.dataset, args.dataroot, data_type="pretraining")

    dataroot, dataset = args.dataroot, args.dataset
    train_type_file, val_type_file, test_type_file, drug_smiles, motif_seq_file =(
        get_file_path(dataroot=dataroot, dataset=dataset))

    args.train_type_file = train_type_file
    args.val_type_file = val_type_file
    args.test_type_file = test_type_file
    args.drug_smiles = drug_smiles
    args.motif_seq_file = motif_seq_file
    assert os.path.isfile(args.train_type_file), "{} is not a file.".format(train_type_file)
    args.verbose = True

    device, device_ids = setup_device(args.ngpu)

    # fix random seeds
    fix_train_random_seed(args.runseed)

    # architecture name
    if args.verbose:
        print('Architecture: {}'.format(args.image_model))

    ##################################### initialize some parameters #####################################

    eval_precision = "precision"
    eval_f1_macro = "f1_macro"
    eval_recall = "recall"
    eval_acc = "acc"
    eval_kappa = "kappa"
    valid_select = "max"
    min_value = -np.inf

    print(" eval_precision: {}".format(eval_precision))
    print(" eval_f1_macro: {}".format(eval_f1_macro))
    print(" eval_recall: {}".format(eval_recall))
    print(" eval_acc: {}".format(eval_acc))
    print(" eval_kappa: {}".format(eval_kappa))

    ##################################### load data #####################################
    # image_aug

    img_transformer_train = [transforms.CenterCrop(args.imageSize), transforms.RandomHorizontalFlip(),
                             transforms.RandomGrayscale(p=0.2), transforms.RandomRotation(degrees=360),
                             transforms.ToTensor()]
    img_transformer_test = [transforms.CenterCrop(args.imageSize), transforms.RandomHorizontalFlip(),
                            transforms.RandomGrayscale(p=0.2), transforms.RandomRotation(degrees=360),
                            transforms.ToTensor()]
    # img_transformer_train = [transforms.CenterCrop(args.imageSize), transforms.ToTensor()]
    # img_transformer_test = [transforms.CenterCrop(args.imageSize), transforms.ToTensor()]

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = DDIDataset(args.image_folder, args.txt_file, args.drug_smiles, args.motif_seq_file, fold=4,
                                 data_type='training', img_transformer=transforms.Compose(img_transformer_train),
                                 normalize=normalize, args=args)
    args.vacb_size = train_dataset.vacb_size
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch,
                                                   shuffle=True,
                                                   num_workers=args.workers,
                                                   pin_memory=True)
    # val dataset
    '''
    names_val_1,names_val_2,labels_val = load_filenames_and_label(args.image_folder, args.txt_file, args.val_type_file)
    names_val_1,names_val_2,labels_val = np.array(names_val_1),np.array(names_val_2),np.array(labels_val)
    '''
    val_dataset = DDIDataset(args.image_folder, args.txt_file, args.drug_smiles, args.motif_seq_file, fold=4,
                               data_type='validation', img_transformer=transforms.Compose(img_transformer_test),
                               normalize=normalize, args=args)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.batch,
                                                 shuffle=True,
                                                 num_workers=args.workers,
                                                 pin_memory=True)
    # test dataset
    '''
    names_test_1,names_test_2,labels_test = load_filenames_and_label(args.image_folder, args.txt_file, args.test_type_file)
    names_test_1,names_test_2,labels_test = np.array(names_test_1),np.array(names_test_2),np.array(labels_test)
    '''
    test_dataset = DDIDataset(args.image_folder, args.txt_file, args.drug_smiles, args.motif_seq_file, fold=4,
                                data_type='test', img_transformer=transforms.Compose(img_transformer_test),
                                normalize=normalize, args=args)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch,
                                                  shuffle=True,
                                                  num_workers=args.workers,
                                                  pin_memory=True)
    ##################################### load model #####################################
    baseModel = load_model(args.image_model, imageSize=args.imageSize, num_classes=args.num_classes)
    if args.resume:
        if os.path.isfile(args.resume):  # only support ResNet18 when loading resume
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)

            ckp_keys = list(checkpoint['state_dict'])
            cur_keys = list(baseModel.state_dict())
            model_sd = baseModel.state_dict()
            if args.image_model == "ResNet18":
                ckp_keys = ckp_keys[:120]
                cur_keys = cur_keys[:120]

            for ckp_key, cur_key in zip(ckp_keys, cur_keys):
                model_sd[cur_key] = checkpoint['state_dict'][ckp_key]

            baseModel.load_state_dict(model_sd)
            arch = checkpoint['arch']
            print("resume model info: arch: {}".format(arch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    print("params: {}".format(cal_torch_model_params(baseModel)))
    model = MolDDI(baseModel, args.num_classes, args.hidden_dim, args.vacb_size, args.max_motif_len)
    print(model)
    print("params: {}".format(cal_torch_model_params(model)))
    model = model.cuda()
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        '''
    baseModel = load_model(args.image_model, imageSize=args.imageSize, num_classes=args.num_classes)

    model = MolDDI(baseModel,args.num_classes,args.hidden_dim,args.vacb_size,args.max_motif_len)
    args.resume = './ckpts/ddi/valid_best (8).pth'
    if args.resume:
        if os.path.isfile(args.resume):  # only support ResNet18 when loading resume
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)          
            print(f'checkpoint: {checkpoint.keys()}')
            ckp_keys = list(checkpoint['model_state_dict'])
            cur_keys = list(model.state_dict())
            model_sd = model.state_dict()
            for ckp_key, cur_key in zip(ckp_keys, cur_keys):
                model_sd[cur_key] = checkpoint['model_state_dict'][ckp_key]
            model.load_state_dict(model_sd)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    print("params: {}".format(cal_torch_model_params(model)))
    print(model)
    model = model.cuda()
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        '''

    ##################################### initialize optimizer #####################################
    optimizer = torch.optim.Adam(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=10 ** args.weight_decay,
    )

    #### nn.CrossEntroy  label long
    criterion = torch.nn.CrossEntropyLoss()

    ##################################### train #####################################
    val_precision_list = []
    val_f1_list = []
    val_recall_list = []
    val_acc_list = []
    val_loss_list = []
    train_loss_list = []
    test_precision_list = []
    test_f1_list = []
    test_recall_list = []
    test_acc_list = []
    test_loss_list = []
    train_loss_list = []

    results = {'highest_valid': min_value,
               'final_train': min_value,
               'final_test': min_value,
               'highest_train': min_value,
               'highest_valid_desc': None,
               "final_train_desc": None,
               "final_test_desc": None}
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_val_recall = 0.0

    early_stop = 0
    patience = 30
    log_file_path = "result/results_image_Ryu0.txt"

    for epoch in range(args.start_epoch, args.epochs):
        # train
        train_one_epoch_multitask(model=model, optimizer=optimizer, data_loader=train_dataloader, criterion=criterion,
                                  device=device, epoch=epoch, task_type=args.task_type)
        # evaluate
        train_loss, train_results, train_data_dict = evaluate_on_multitask(model=model, data_loader=train_dataloader,
                                                                           criterion=criterion, device=device,
                                                                           epoch=epoch, return_data_dict=True)
        val_loss, val_results, val_data_dict = evaluate_on_multitask(model=model, data_loader=val_dataloader,
                                                                     criterion=criterion, device=device, epoch=epoch,
                                                                     task_type=args.task_type, return_data_dict=True)
        test_loss, test_results, test_data_dict = evaluate_on_multitask(model=model, data_loader=test_dataloader,
                                                                        criterion=criterion, device=device, epoch=epoch,
                                                                        task_type=args.task_type, return_data_dict=True)

        # train_result_precision = train_results[eval_precision]
        # valid_result_precision = val_results[eval_precision]
        # test_result_precision = test_results[eval_precision]
        print({"epoch": epoch, "patience": early_stop, "Loss": train_loss})
        print({'Train_precision': train_results['precision'], 'Train_f1_macro': train_results['f1_macro'],
               'Train_recall': train_results['recall'], 'Train_acc': train_results['acc'],
               'Train_kappa': train_results['kappa']})
        print({'Val_precision': val_results['precision'], 'Val_f1_macro': val_results['f1_macro'],
               'Val_recall': val_results['recall'], 'Val_acc': val_results['acc'], 'Val_kappa': val_results['kappa']})
        print({'Test_precision': test_results['precision'], 'Test_f1_macro': test_results['f1_macro'],
               'Test_recall': test_results['recall'], 'Test_acc': test_results['acc'],
               'Test_kappa': test_results['kappa']})

        val_precision_list.append(val_results['precision'])
        val_f1_list.append(val_results['f1_macro'])
        val_recall_list.append(val_results['recall'])
        val_acc_list.append(val_results['acc'])
        train_loss_list.append(train_loss)
        # 检查是否在验证集上达到了更高的 ACC 或 F1
        if val_results['f1_macro'] > best_val_f1:

            best_val_recall = max(best_val_f1, val_results['f1_macro'])

            results['highest_valid'] = val_results
            results['final_train'] = train_results
            results['final_test'] = test_results

            results['highest_valid_desc'] = val_results
            results['final_train_desc'] = train_results
            results['final_test_desc'] = test_results

            if args.save_finetune_ckpt == 1:
                save_finetune_ckpt(model, optimizer, round(train_loss, 4), epoch, args.log_dir, "valid_best",
                                   lr_scheduler=None, result_dict=results)
            early_stop = 0
        else:
            early_stop += 1
            if early_stop > patience:
                break

    y_true = val_data_dict['y_true']

    features = val_data_dict['features']
    save_features(val_data_dict['features'], val_data_dict['y_true'], 'val_features_test.pkl')
    # the result
    print("final results: highest_valid_precision: {:.3f}, final_train_precision: {:.3f}, final_test_precision: {:.3f}"
          .format(results["highest_valid"]['precision'], results["final_train"]['precision'],
                  results["final_test"]['precision']))
    print("final results: highest_valid_acc: {:.3f}, final_train_acc: {:.3f}, final_test_acc: {:.3f}"
          .format(results["highest_valid"]['acc'], results["final_train"]['acc'], results["final_test"]['acc']))

    # print the f1 score and recall
    print("final results: highest_valid_f1: {:.3f}, final_train_f1: {:.3f}, final_test_f1: {:.3f}"
          .format(results["highest_valid"]['f1_macro'], results["final_train"]['f1_macro'],
                  results["final_test"]['f1_macro']))
    print("final results: highest_valid_recall: {:.3f}, final_train_recall: {:.3f}, final_test_recall: {:.3f}"
          .format(results["highest_valid"]['recall'], results["final_train"]['recall'],
                  results["final_test"]['recall']))

    # create a figure
    fig1, ax1 = plt.subplots()


    ax1.plot(val_precision_list, label='Precision')
    ax1.plot(val_f1_list, label='F1 Score')
    ax1.plot(val_recall_list, label='Recall')
    ax1.plot(val_acc_list, label='Accuracy')


    ax1.legend()


    ax1.set_title('Validation Metrics Over Time')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Metric Value')


    plt.savefig('result/metrics_chart.png')

    fig2, ax2 = plt.subplots()
    ax2.plot(train_loss_list, label='Train Loss')
    ax2.legend()
    ax2.set_title('Train Loss Over Time')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    plt.savefig('result/loss_chart.png')


if __name__ == "__main__":
    args = parse_args()
    main(args)
