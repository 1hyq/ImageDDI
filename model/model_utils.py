import logging
import os
import sys
import torch
import torchvision
from sklearn import metrics
from tqdm import tqdm
from model.evaluate import metric as utils_evaluate_metric



def get_support_model_names():
    return ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]


def load_model(modelname="ResNet18", imageSize=224, num_classes=2):
    assert modelname in get_support_model_names()
    if modelname == "ResNet18":
        model = torchvision.models.resnet18(pretrained=False)
       # model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet34":
        model = torchvision.models.resnet34(pretrained=False)
       # model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet50":
        model = torchvision.models.resnet50(pretrained=False)
       # model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet101":
        model = torchvision.models.resnet101(pretrained=False)
       # model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet152":
        model = torchvision.models.resnet152(pretrained=False)
      #   model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise Exception("{} is undefined".format(modelname))
    return model


# evaluation for classification
def metric(y_true, y_pred, y_prob):
    acc = metrics.accuracy_score(y_true, y_pred)
    auc = metrics.roc_auc_score(y_true, y_prob)

    f1 = metrics.f1_score(y_true, y_pred)
    precision_list, recall_list, _ = metrics.precision_recall_curve(y_true, y_prob)
    aupr = metrics.auc(recall_list, precision_list)
    precision = metrics.precision_score(y_true, y_pred, zero_division=1)
    recall = metrics.recall_score(y_true, y_pred, zero_division=1)
    kappa = metrics.cohen_kappa_score(y_true, y_pred)
    matthews = metrics.matthews_corrcoef(y_true, y_pred)
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_prob)

    return {
        "accuracy": acc,
        "ROCAUC": auc,
        "f1": f1,
        "AUPR": aupr,
        "precision": precision,
        "recall": recall,
        "kappa": kappa,
        "matthews": matthews,
        "fpr": fpr,  # list
        "tpr": tpr,  # list
        "precision_list": precision_list,
        "recall_list": recall_list
    }



def train_one_epoch_multitask(model, optimizer, data_loader, criterion, device, epoch, task_type):
    '''
    :param model:
    :param optimizer:
    :param data_loader:
    :param criterion:
    :param device:
    :param epoch:
    :param criterion_lambda:
    :return:
    '''

    model.train()
    accu_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images_1, images_2, motif_seq, labels = data
        images_1, images_2, labels = images_1.to(device), images_2.to(device), labels.to(device)
        # motif_tensor = torch.tensor(motif_seq)
        motif_seq = motif_seq.to(device)
        # combined_cliques = combined_cliques.to(device)
        # distance_matrix = distance_matrix.to(device)
        sample_num = images_1.shape[0] + images_2.shape[0]

        pred, feature = model(images_1, images_2, motif_seq)
        # labels = labels.view(pred.shape).to(torch.float64)
        is_valid = labels != -1
        labels = labels.long()
        loss_mat = criterion(pred.double(), labels)  ### labels.long()
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1)


@torch.no_grad()
def evaluate_on_multitask(model, data_loader, criterion, device, epoch, task_type="classification",
                          return_data_dict=False):
    model.eval()

    accu_loss = torch.zeros(1).to(device)
    y_scores, y_true, y_pred, y_prob, features = [], [], [], [], []
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images_1, images_2, motif_seq, labels = data
        images_1, images_2, labels = images_1.to(device), images_2.to(device), labels.to(device)
        motif_seq = motif_seq.to(device)

        with torch.no_grad():
            pred, feature = model(images_1, images_2, motif_seq)
            features.append(feature.cpu())

            if task_type == "classification":
                is_valid = labels != -1
                labels = labels.long()
                loss_mat = criterion(pred, labels)
                loss_mat = torch.where(is_valid, loss_mat,
                                       torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
                loss = torch.sum(loss_mat) / torch.sum(is_valid)
            accu_loss += loss.detach()
            data_loader.desc = "[valid epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))

        y_true.append(labels.cpu())
        y_pred.append(pred.cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    features = torch.cat(features, dim=0).numpy()

    if return_data_dict:
        data_dict = {"y_true": y_true, "y_pred": y_pred, "features": features}
        return accu_loss.item() / (step + 1), utils_evaluate_metric(y_true, y_pred, empty=-1), data_dict
    else:
        return accu_loss.item() / (step + 1), utils_evaluate_metric(y_true, y_pred, empty=-1), features


def save_finetune_ckpt(model, optimizer, loss, epoch, save_path, filename_pre, lr_scheduler=None, result_dict=None,
                       logger=None):
    log = logger if logger is not None else logging
    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    lr_scheduler = None if lr_scheduler is None else lr_scheduler.state_dict()
    state = {
        'epoch': epoch,
        'model_state_dict': model_cpu,
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler,
        'loss': loss,
        'result_dict': result_dict
    }
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        log.info("Directory {} is created.".format(save_path))

    filename = '{}/{}.pth'.format(save_path, filename_pre)
    torch.save(state, filename)
    log.info('model has been saved as {}'.format(filename))

