from __future__ import print_function

import numpy as np

import argparse
import torch
import os
import pandas as pd
from utils.utils import *
#, save_splits
from datasets.dataset_mtl import Generic_MIL_MTL_Dataset
#, save_splits
from utils.eval_utils_mtl import compute_features, eval_ as eval_mtl
from utils.eval_utils_mtl_ms import compute_features, eval_ as eval_mtl_ms

# Training settings
parser = argparse.ArgumentParser(description='MANTA Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default='',
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small',
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['attention_mil'], default='attention_mil',
                    help='type of model (default: attention_mil)')
parser.add_argument('--drop_out', action='store_true', default=False,
                    help='whether model uses dropout')
parser.add_argument('--calc_features', action='store_true', default=False,
                    help='calculate features for pca/tsne')
parser.add_argument('--k', type=int, default=1, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False,
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--mtl', action='store_true', default=True, help='flag to enable multi-task problem')
parser.add_argument('--patient_level', action='store_true', default=False, help='To enable computing scores at the patient-level. I.e. all patients slides are treated as a single bag with a single label')
parser.add_argument('--stain_level', action='store_true', default=False)
parser.add_argument('--fusion', type=str, default='tensor')

parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str,
choices=['kidney-mtl'])

args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


encoding_size = 768

args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(os.path.join(args.save_dir, 'attention_scores'), exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir,
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size,
            'micro_average': args.micro_average}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)

if args.task == 'kidney-mtl':
    args.n_classes=[2,2,3]  
    dataset = Generic_MIL_MTL_Dataset(csv_path = 'dataset_csv/KidneySimpleLabels_edited_all_slides.csv',
                            data_dir= os.path.join(args.data_root_dir, 'kidney-features'),
                            shuffle = False,
                            print_info = True,
                            label_dicts = [{'no_cell':0, 'cell':1},
                                            {'no_amr':0, 'amr':1},
                                            {'mild_ifta':0, 'moderate_ifta':1, 'advanced_ifta':2}],
                            label_cols=['label_cell','label_amr','label_ifta'],
                            patient_strat=False,
                            ignore=[],
                            patient_level = args.patient_level,
                            stain_level = args.stain_level,
                            fusion = args.fusion)


elif os.path.isdir(args.task):
    print('reading directory for fast inference')

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}



def main_mtl(args):
    all_task1_auc = []
    all_task1_acc = []
    all_task2_auc = []
    all_task2_acc = []
    all_task3_auc = []
    all_task3_acc = []
    results_dict_all = {}

    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]

        model, results_dict = eval_mtl(split_dataset, args, ckpt_paths[ckpt_idx])
        results_dict_all[ckpt_idx] = results_dict

        all_task1_auc.append(results_dict['auc_task1'])
        all_task1_acc.append(1-results_dict['test_error_task1'])
        all_task2_auc.append(results_dict['auc_task2'])
        all_task2_acc.append(1-results_dict['test_error_task2'])
        all_task3_auc.append(results_dict['auc_task3'])
        all_task3_acc.append(1-results_dict['test_error_task3'])

        df = results_dict['df']
        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)

        if args.calc_features:
            compute_features(split_dataset, args, ckpt_paths[ckpt_idx], args.save_dir, model=model)

    df_dict = {'folds': folds,
                'task1_test_auc': all_task1_auc, 'task1_test_acc': all_task1_acc,
                'task2_test_auc': all_task2_auc, 'task2_test_acc': all_task2_acc,
                'task3_test_auc': all_task3_auc, 'task3_test_acc': all_task3_acc}


    final_df = pd.DataFrame(df_dict)
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.save_dir, save_name))



def main_mtl_ms(args):
    all_task1_auc = []
    all_task1_acc = []
    all_task2_auc = []
    all_task2_acc = []
    all_task3_auc = []
    all_task3_acc = []
    results_dict_all = {}

    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]

        model, results_dict = eval_mtl_ms(split_dataset, args, ckpt_paths[ckpt_idx])
        results_dict_all[ckpt_idx] = results_dict

        all_task1_auc.append(results_dict['auc_task1'])
        all_task1_acc.append(1-results_dict['test_error_task1'])
        all_task2_auc.append(results_dict['auc_task2'])
        all_task2_acc.append(1-results_dict['test_error_task2'])
        all_task3_auc.append(results_dict['auc_task3'])
        all_task3_acc.append(1-results_dict['test_error_task3'])

        df = results_dict['df']
        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)

        if args.calc_features:
            compute_features(split_dataset, args, ckpt_paths[ckpt_idx], args.save_dir, model=model)

    df_dict = {'folds': folds,
                'task1_test_auc': all_task1_auc, 'task1_test_acc': all_task1_acc,
                'task2_test_auc': all_task2_auc, 'task2_test_acc': all_task2_acc,
                'task3_test_auc': all_task3_auc, 'task3_test_acc': all_task3_acc}
    import pickle
    with open(os.path.join(args.save_dir, 'results_dict.pkl'), 'wb') as out:
        pickle.dump(results_dict_all, out)


    final_df = pd.DataFrame(df_dict)
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.save_dir, save_name))

    import pickle



if __name__ == "__main__":
    if args.stain_level:
        main_mtl_ms(args)
    elif args.mtl:
        main_mtl(args)

    print("finished!")
    print("end script")        
