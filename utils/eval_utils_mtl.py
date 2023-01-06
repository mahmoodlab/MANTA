import numpy as np

import torch
from models.model_attention_mil import MIL_Attention_fc_mtl
import os
import pandas as pd
from utils.utils import *
from utils.core_utils_mtl import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc
import h5py
from sklearn.preprocessing import label_binarize

def initiate_model(args, ckpt_path=None):
    print('Init Model')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}

    if args.model_size is not None and args.model_type in ['clam', 'attention_mil', 'clam_simple']:
        model_dict.update({"size_arg": args.model_size})

    if args.model_type == 'attention_mil':
        model = MIL_Attention_fc_mtl(**model_dict)
    else: 
        raise NotImplementedError

    print_network(model)

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path)
        ckpt_clean = {}
        for key in ckpt.keys():
            if 'instance_loss_fn' in key:
                continue
            ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
        model.load_state_dict(ckpt_clean, strict=True)
    model.relocate()
    model.eval()
    return model


def eval_(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)


    print('Init Loaders')
    loader = get_simple_loader(dataset, collate_fn='MIL_mtl')
    results_dict = summary(model, loader, args)

    print('test_error_task1: ', results_dict['test_error_task1'])
    print('auc_task1: ',        results_dict['auc_task1'])
    print('test_error_task2: ', results_dict['test_error_task2'])
    print('auc_task2: ',        results_dict['auc_task2'])
    print('test_error_task3: ', results_dict['test_error_task3'])
    print('auc_task3: ',        results_dict['auc_task3'])

    return model, results_dict
    # patient_results, test_error, auc, aucs, df

def infer(dataset, args, ckpt_path, class_labels, site_labels):
    model = initiate_model(args, ckpt_path)
    df = infer_dataset(model, dataset, args, class_labels, site_labels)
    return model, df

# Code taken from pytorch/examples for evaluating topk classification on on ImageNet
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res

def summary(model, loader, args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger_task1 = Accuracy_Logger(n_classes=args.n_classes[0])
    logger_task2 = Accuracy_Logger(n_classes=args.n_classes[1])
    logger_task3 = Accuracy_Logger(n_classes=args.n_classes[2])
    model.eval()

    test_error_task1 = 0.
    test_loss_task1  = 0.
    test_error_task2 = 0.
    test_loss_task2  = 0.
    test_error_task3 = 0.
    test_loss_task3  = 0.

    all_probs_task1  = np.zeros((len(loader), args.n_classes[0]))
    all_labels_task1 = np.zeros(len(loader))
    all_probs_task2  = np.zeros((len(loader), args.n_classes[1]))
    all_labels_task2 = np.zeros(len(loader))
    all_probs_task3  = np.zeros((len(loader), args.n_classes[2]))
    all_labels_task3 = np.zeros(len(loader))

    if not args.patient_level:
        slide_ids = loader.dataset.slide_data['slide_id']
        patient_results = {}

        for batch_idx, (data, label_task1, label_task2, label_task3) in enumerate(loader):
            data =  data.to(device)	    
            label_task1 = label_task1.to(device)
            label_task2 = label_task2.to(device)            
            label_task3 = label_task3.to(device)

            slide_id = slide_ids.iloc[batch_idx]
            with torch.no_grad():
                model_results_dict = model(data)

            logits_task1, Y_prob_task1, Y_hat_task1  = model_results_dict['logits_task1'], model_results_dict['Y_prob_task1'], model_results_dict['Y_hat_task1']
            logits_task2, Y_prob_task2, Y_hat_task2  = model_results_dict['logits_task2'], model_results_dict['Y_prob_task2'], model_results_dict['Y_hat_task2']
            logits_task3, Y_prob_task3, Y_hat_task3  = model_results_dict['logits_task3'], model_results_dict['Y_prob_task3'], model_results_dict['Y_hat_task3']
            del model_results_dict

            logger_task1.log(Y_hat_task1, label_task1)
            logger_task2.log(Y_hat_task2, label_task2)
            logger_task3.log(Y_hat_task3, label_task3)

            probs_task1 = Y_prob_task1.cpu().numpy()
            all_probs_task1[batch_idx] = probs_task1
            all_labels_task1[batch_idx] = label_task1.item()

            probs_task2 = Y_prob_task2.cpu().numpy()
            all_probs_task2[batch_idx] = probs_task2
            all_labels_task2[batch_idx] = label_task2.item()

            probs_task3 = Y_prob_task3.cpu().numpy()
            all_probs_task3[batch_idx] = probs_task3
            all_labels_task3[batch_idx] = label_task3.item()

            patient_results.update({slide_id: {'slide_id': np.array(slide_id),
				    'prob_task1': probs_task1, 'label_task1': label_task1.item(),
				    'prob_task2': probs_task2, 'label_task2': label_task2.item(),
				    'prob_task3': probs_task3, 'label_task3': label_task3.item() }})

            error_task1 = calculate_error(Y_hat_task1, label_task1)
            test_error_task1 += error_task1
            error_task2 = calculate_error(Y_hat_task2, label_task2)
            test_error_task2 += error_task2
            error_task3 = calculate_error(Y_hat_task3, label_task3)
            test_error_task3 += error_task3
    else:
        case_ids = loader.dataset.slide_data['case_id']
        patient_results = {}

        for batch_idx, (data, label_task1, label_task2, label_task3) in enumerate(loader):
            data =  data.to(device)
            label_task1 = label_task1.to(device)
            label_task2 = label_task2.to(device)
            label_task3 = label_task3.to(device)

            case_id = case_ids.iloc[batch_idx]
            with torch.no_grad():
                model_results_dict = model(data)

            logits_task1, Y_prob_task1, Y_hat_task1  = model_results_dict['logits_task1'], model_results_dict['Y_prob_task1'], model_results_dict['Y_hat_task1']
            logits_task2, Y_prob_task2, Y_hat_task2  = model_results_dict['logits_task2'], model_results_dict['Y_prob_task2'], model_results_dict['Y_hat_task2']
            logits_task3, Y_prob_task3, Y_hat_task3  = model_results_dict['logits_task3'], model_results_dict['Y_prob_task3'], model_results_dict['Y_hat_task3']
            del model_results_dict

            logger_task1.log(Y_hat_task1, label_task1)
            logger_task2.log(Y_hat_task2, label_task2)
            logger_task3.log(Y_hat_task3, label_task3)

            probs_task1 = Y_prob_task1.cpu().numpy()
            all_probs_task1[batch_idx] = probs_task1
            all_labels_task1[batch_idx] = label_task1.item()

            probs_task2 = Y_prob_task2.cpu().numpy()
            all_probs_task2[batch_idx] = probs_task2
            all_labels_task2[batch_idx] = label_task2.item()

            probs_task3 = Y_prob_task3.cpu().numpy()
            all_probs_task3[batch_idx] = probs_task3
            all_labels_task3[batch_idx] = label_task3.item()

            patient_results.update({case_id: {'case_id': np.array(case_id),
                                    'prob_task1': probs_task1, 'label_task1': label_task1.item(),
                                    'prob_task2': probs_task2, 'label_task2': label_task2.item(),
                                    'prob_task3': probs_task3, 'label_task3': label_task3.item() }})

            error_task1 = calculate_error(Y_hat_task1, label_task1)
            test_error_task1 += error_task1
            error_task2 = calculate_error(Y_hat_task2, label_task2)
            test_error_task2 += error_task2
            error_task3 = calculate_error(Y_hat_task3, label_task3)
            test_error_task3 += error_task3


    test_error_task1 /= len(loader)
    test_error_task2 /= len(loader)
    test_error_task3 /= len(loader)

    all_preds_task1 = np.argmax(all_probs_task1, axis=1)
    all_preds_task2 = np.argmax(all_probs_task2, axis=1)
    all_preds_task3 = np.argmax(all_probs_task3, axis=1)

        
    # TASK 1
    if len(np.unique(all_labels_task1)) == 1:
        task1_auc = -1
        task1_aucs = []
    else:
        if args.n_classes[0] == 2:
            task1_auc = roc_auc_score(all_labels_task1, all_probs_task1[:, 1])
            task1_aucs = []
        else:
            task1_aucs = []
            binary_labels = label_binarize(all_labels_task1, classes=[i for i in range(args.n_classes[0])])
            for class_idx in range(args.n_classes[0]):
                if class_idx in all_labels_task1:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs_task1[:, class_idx])
                    task1_aucs.append(auc(fpr,tpr))
                else:
                    task1_aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels_task1, classes=[i for i in range(args.n_classes[0])])
                valid_classes = np.where(np.any(binary_labels, axis=0))[0]
                binary_labels = binary_labels[:, valid_classes]
                valid_probs_task1 = all_probs_task1[:, valid_classes]
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), valid_probs_task1.ravel())
                task1_auc = auc(fpr, tpr)
            else:
                task1_auc = np.nanmean(np.array(task1_aucs))



    # TASK 2
    if len(np.unique(all_labels_task2)) == 1:
        task2_auc = -1
        task2_aucs = []
    else:
        if args.n_classes[1] == 2:
            task2_auc = roc_auc_score(all_labels_task2, all_probs_task2[:, 1])
            task2_aucs = []

        else:
            task2_aucs = []
            binary_labels = label_binarize(all_labels_task2, classes=[i for i in range(args.n_classes[1])])
            for class_idx in range(args.n_classes[1]):
                if class_idx in all_labels_task2:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs_task2[:, class_idx])
                    task2_aucs.append(auc(fpr,tpr))
                else:
                    task2_aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels_task2, classes=[i for i in range(args.n_classes[1])])
                valid_classes = np.where(np.any(binary_labels, axis=0))[0]
                binary_labels = binary_labels[:, valid_classes]
                valid_probs_task2 = all_probs_task2[:, valid_classes]
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), valid_probs_task2.ravel())
                task2_auc = auc(fpr, tpr)
            else:
               task2_auc = np.nanmean(np.array(task2_aucs))



    # TASK 3
    if len(np.unique(all_labels_task3)) == 1:
        task3_auc = -1
        task3_aucs = []
    else:
        if args.n_classes[2] == 2:
            task3_auc = roc_auc_score(all_labels_task3, all_probs_task3[:, 1])
            task3_aucs = []
        else:
            task3_aucs = []
            binary_labels = label_binarize(all_labels_task3, classes=[i for i in range(args.n_classes[2])])
            for class_idx in range(args.n_classes[2]):
                if class_idx in all_labels_task3:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs_task3[:, class_idx])
                    task3_aucs.append(auc(fpr,tpr))
                else:
                    task3_aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels_task3, classes=[i for i in range(args.n_classes[2])])
                valid_classes = np.where(np.any(binary_labels, axis=0))[0]
                binary_labels = binary_labels[:, valid_classes]
                valid_probs_task3 = all_probs_task3[:, valid_classes]
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), valid_probs_task3.ravel())
                task3_auc = auc(fpr, tpr)
            else:
               task3_auc = np.nanmean(np.array(task3_aucs))         
	

    if not args.patient_level:
        results_dict = {'slide_id': slide_ids,
			'Y_task1': all_labels_task1, 'Y_hat_task1': all_preds_task1,
			'Y_task2': all_labels_task2, 'Y_hat_task2': all_preds_task2,
			'Y_task3': all_labels_task3, 'Y_hat_task3': all_preds_task3}
    else:
        results_dict = {'case_id': case_ids,
                        'Y_task1': all_labels_task1, 'Y_hat_task1': all_preds_task1,
                        'Y_task2': all_labels_task2, 'Y_hat_task2': all_preds_task2,
                        'Y_task3': all_labels_task3, 'Y_hat_task3': all_preds_task3}
    

    for c in range(args.n_classes[0]):
       results_dict.update({ f"p{c}_task1" : all_probs_task1[:,c]  })

    for c in range(args.n_classes[1]):
       results_dict.update({ f"p{c}_task2" : all_probs_task2[:,c]  })

    for c in range(args.n_classes[2]):
       results_dict.update({ f"p{c}_task3" : all_probs_task3[:,c]  })


    df = pd.DataFrame(results_dict)

    if args.patient_level:
        df = df.drop_duplicates(subset=['case_id'])

    inference_results = {'patient_results': patient_results,
                        'test_error_task1': test_error_task1, 'auc_task1': task1_auc,
                        'test_error_task2': test_error_task2, 'auc_task2': task2_auc,
                        'test_error_task3': test_error_task3, 'auc_task3': task3_auc,
                        'loggers': (logger_task1, logger_task2, logger_task3), 'df':df}

    return inference_results


def infer_dataset(model, dataset, args, class_labels, site_labels, k=3):
    model.eval()
    all_probs_cls = np.zeros((len(dataset), k))
    all_probs_site = np.zeros((len(dataset),2))

    all_preds_cls = np.zeros((len(dataset), k))
    all_preds_cls_str = np.full((len(dataset), k), ' ', dtype=object)
    all_preds_site = np.full((len(dataset)), ' ', dtype=object)

    slide_ids = dataset.slide_data
    for batch_idx, data in enumerate(dataset):
        data = data.to(device)
        with torch.no_grad():
            results_dict = model(data)

        Y_prob, Y_hat = results_dict['Y_prob'], results_dict['Y_hat']
        site_prob, site_hat = results_dict['site_prob'], results_dict['site_hat']
        del results_dict
        probs, ids = torch.topk(Y_prob, k)
        probs = probs.cpu().numpy()
        site_prob = site_prob.cpu().numpy()
        ids = ids.cpu().numpy()
        all_probs_cls[batch_idx] = probs
        all_preds_cls[batch_idx] = ids
        all_preds_cls_str[batch_idx] = np.array(class_labels)[ids]

        all_probs_site[batch_idx] = site_prob
        all_preds_site[batch_idx] = np.array(site_labels)[site_hat.item()]

    del data
    results_dict = {'slide_id': slide_ids}
    for c in range(k):
        results_dict.update({'Pred_{}'.format(c): all_preds_cls_str[:, c]})
        results_dict.update({'p_{}'.format(c): all_probs_cls[:, c]})
    results_dict.update({'Site_Pred': all_preds_site, 'Site_p': all_probs_site[:, 1]})
    df = pd.DataFrame(results_dict)
    return df

def compute_features(dataset, args, ckpt_path, save_dir, model=None, feature_dim=512):
    if model is None:
        model = initiate_model(args, ckpt_path)

    names = dataset.get_list(np.arange(len(dataset))).values
    file_path = os.path.join(save_dir, 'features.h5')

    initialize_features_hdf5_file(file_path, len(dataset), feature_dim=feature_dim, names=names)
    for i in range(len(dataset)):
        print("Progress: {}/{}".format(i, len(dataset)))
        save_features(dataset, i, model, args, file_path)

def save_features(dataset, idx, model, args, save_file_path):
    name = dataset.get_list(idx)
    print(name)
    features, label_task1, label_task2, label_task3 = dataset[idx]
    features = features.to(device)
    with torch.no_grad():
        results_dict = model(features, return_features=True)
        Y_prob_task1, Y_hat_task1 = results_dict['Y_prob_task1'], results_dict['Y_hat_task1']
        Y_prob_task2, Y_hat_task2 = results_dict['Y_prob_task2'], results_dict['Y_hat_task2']
        Y_prob_task3, Y_hat_task3 = results_dict['Y_prob_task3'], results_dict['Y_hat_task3']

        feat_task1 = results_dict['features'][0]
        feat_task2 = results_dict['features'][1]
        feat_task3 = results_dict['features'][2]

    del results_dict
    del features

    Y_hat_task1  = Y_hat_task1.item()
    Y_prob_task1 = Y_prob_task1.view(-1).cpu().numpy()
    Y_hat_task2  = Y_hat_task2.item()
    Y_prob_task2 = Y_prob_task2.view(-1).cpu().numpy()
    Y_hat_task3  = Y_hat_task3.item()
    Y_prob_task3 = Y_prob_task3.view(-1).cpu().numpy()

    feat_task1 = feat_task1.view(1, -1).cpu().numpy()
    feat_task2 = feat_task2.view(1, -1).cpu().numpy()
    feat_task3 = feat_task3.view(1, -1).cpu().numpy()

    with h5py.File(save_file_path, 'r+') as file:
        print('label_task1', label_task1)
        file['features_task1'][idx, :] = feat_task1
        file['features_task2'][idx, :] = feat_task2
        file['features_task3'][idx, :] = feat_task3
        file['label_task1'][idx] = label_task1
        file['Y_hat_task1'][idx] = Y_hat_task1
        file['Y_prob_task1'][idx] = Y_prob_task1[1]
        file['label_task2'][idx] = label_task2
        file['Y_hat_task2'][idx] = Y_hat_task2
        file['Y_prob_task2'][idx] = Y_prob_task2[1]
        file['label_task3'][idx] = label_task3
        file['Y_hat_task3'][idx] = Y_hat_task3
        file['Y_prob_task3'][idx] = Y_prob_task3[1]



def initialize_features_hdf5_file(file_path, length, feature_dim=512, names = None):

    file = h5py.File(file_path, "w")

    dset = file.create_dataset('features_task1',
                                shape=(length, feature_dim), chunks=(1, feature_dim), dtype=np.float32)
    dset = file.create_dataset('features_task2',
                                shape=(length, feature_dim), chunks=(1, feature_dim), dtype=np.float32)
    dset = file.create_dataset('features_task3',
                                shape=(length, feature_dim), chunks=(1, feature_dim), dtype=np.float32)

    if names is not None:
        dt = h5py.string_dtype()
        label_dset = file.create_dataset('names', shape=(length, ), chunks=(1, ), dtype=dt)
        file['names'][:] = names

    label_dset = file.create_dataset('label_task1', shape=(length, ), chunks=(1, ), dtype=np.int32)
    pred_dset = file.create_dataset( 'Y_hat_task1', shape=(length, ), chunks=(1, ), dtype=np.int32)
    prob_dset = file.create_dataset( 'Y_prob_task1', shape=(length, ), chunks=(1, ), dtype=np.float32)
    label_dset = file.create_dataset('label_task2', shape=(length, ), chunks=(1, ), dtype=np.int32)
    pred_dset = file.create_dataset( 'Y_hat_task2', shape=(length, ), chunks=(1, ), dtype=np.int32)
    prob_dset = file.create_dataset( 'Y_prob_task2', shape=(length, ), chunks=(1, ), dtype=np.float32)
    label_dset = file.create_dataset('label_task3', shape=(length, ), chunks=(1, ), dtype=np.int32)
    pred_dset = file.create_dataset( 'Y_hat_task3', shape=(length, ), chunks=(1, ), dtype=np.int32)
    prob_dset = file.create_dataset( 'Y_prob_task3', shape=(length, ), chunks=(1, ), dtype=np.float32)

    file.close()
    return file_path
