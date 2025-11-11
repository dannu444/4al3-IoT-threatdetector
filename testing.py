import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, make_scorer

import torch
import torch.nn as nn
import torch.optim as optim
import random

import train

from argparse import ArgumentParser 
from itertools import product

class Testing():
    def __init__(self):

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

        self.input_dim = 83 
        self.output_dim = 13 
        self.check_every = 5000

    def model_structure(self):
        reduction_prob = 0.5
        layer_sizes = [16, 32, 64, 128, 256, 512] 
        hidden_dim = []

        layer_count = random.randint(2, 4) #min 2 layers, max 4 layers
        rand_idx = random.randint(1, 5)
        current_layer_size = layer_sizes[rand_idx]
        hidden_dim.append(current_layer_size)

        for i in range(layer_count - 1):
            if current_layer_size > 16  and random.random() < reduction_prob:
                current_layer_size = current_layer_size // 2
            hidden_dim.append(current_layer_size)
        
        return hidden_dim

    def set_all_seeds(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def hyperparameter_tuning(self):
        hidden_candidates = [self.model_structure() for _ in range(20)] 
        batch_values      = [64, 128, 256]
        learning_rates    = [1e-1, 5e-2, 1e-2, 5e-3]
        iteration_counts  = [2000, 5000, 10000, 50000]

        all_combos = list(product(hidden_candidates, batch_values, learning_rates, iteration_counts))
        trial_set = random.sample(all_combos, k=50)

        device  = torch.device("cpu")
        results = {}
        best    = {"score": -1.0, "cfg": None}

        trial_idx = 0
        for hidden_struct, bs, lr, iters in trial_set:
            trial_idx += 1
            idx = 1234 + trial_idx
            self.set_all_seeds(idx)
            model = train.GeneralNN(self.input_dim, hidden_struct, self.output_dim)
            model = model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=lr)
            train_losses, val_losses, train_fs, val_fs, iterations = train.train_SGD(model, criterion, optimizer, self.X_train, self.y_train, self.X_val, self.y_val, iters, bs, self.check_every)

            if (iterations[-1] < 50000):
                train_losses.append(train_losses[-1])
                val_losses.append(val_losses[-1])
                train_fs.append(train_fs[-1])
                val_fs.append(val_fs[-1])
                iterations.append(50000)
            
            final_val_f1 = float(val_fs[-1])
            results[(tuple(hidden_struct), bs, lr, iters)] = {
                "train_losses": train_losses,
                "val_losses":   val_losses,
                "train_fs":     train_fs,
                "val_fs":       val_fs,
                "iterations":   iterations,
            }
            
            print(f"[{trial_idx}] hs={hidden_struct} bs={bs} lr={lr} iters={iters} -> val_f1={final_val_f1:.4f}")
            if final_val_f1 > best["score"]:
                best["score"] = final_val_f1
                best["cfg"]   = {"hidden": hidden_struct, "bs": bs, "lr": lr, "iters": iters}

        print("\n")
        print(f"best configuration:                 hs={best['cfg']['hidden']}  bs={best['cfg']['bs']}  lr={best['cfg']['lr']}  iters={best['cfg']['iters']}")
        print(f"best FS score on validation data:   {best['score']:.4f}")

        return results, best
            
    def best_each(self, results):
        best_bs = {}
        best_iter = {}
        best_lr = {}
        for (hidden, bs, lr, iters), metrics in results.items():
            score = metrics["val_fs"][-1]

            if bs not in best_bs or score > best_bs[bs][1]:
                best_bs[bs] = ((hidden, bs, lr, iters), score, metrics)

            if iters not in best_iter or score > best_iter[iters][1]:
                best_iter[iters] = ((hidden, bs, lr, iters), score, metrics)

            if lr not in best_lr or score > best_lr[lr][1]:
                best_lr[lr] = ((hidden, bs, lr, iters), score, metrics)
        
        self.graph_best_bs(best_bs, 1)
        self.graph_best_bs(best_iter, 2)
        self.graph_best_bs(best_lr, 3)
    
    def graph_best_bs(self, best_list, z):
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        for key, (config, score, metrics) in best_list.items():
            hidden, bs, lr, iters = config
            if (z == 1):
                label = f"bs={bs}  (lr={lr} | i={iters} | {hidden})"
            if (z == 2):
                label = f"i={iters}  with (bs={bs} | lr={lr} | {hidden})"
            if (z == 3):
                label = f"lr={lr}  (bs={bs} | i={iters} | {hidden})"
            
            axes[0,0].plot(metrics["iterations"], metrics["train_losses"], label=f"{label}", linestyle=':', marker='o')
            axes[0,1].plot(metrics["iterations"], metrics["val_losses"], label=f"{label}", linestyle='--', marker='x')
            axes[1,0].plot(metrics["iterations"], metrics["train_fs"], label=f"{label}", linestyle=':', marker='o')
            axes[1,1].plot(metrics["iterations"], metrics["val_fs"], label=f"{label}", linestyle='--', marker='x')
        
        if (z == 1):
            axes[0,0].set_title('Train Loss from Top Batch Size Results', fontsize=14, fontweight='bold')
            axes[0,1].set_title('Value Loss from Top Batch Size Results', fontsize=14, fontweight='bold')
            axes[1,0].set_title('Train FS from Top Batch Size Results', fontsize=14, fontweight='bold')
            axes[1,1].set_title('Value FS from Top Batch Size Results', fontsize=14, fontweight='bold')
        if (z == 2):
            axes[0,0].set_title('Train Loss from Top Max Iteration Results', fontsize=14, fontweight='bold')
            axes[0,1].set_title('Value Loss from Top Max Iteration Results', fontsize=14, fontweight='bold')
            axes[1,0].set_title('Train FS from Top Max Iteration Results', fontsize=14, fontweight='bold')
            axes[1,1].set_title('Value FS from Top Max Iteration Results', fontsize=14, fontweight='bold')
        if (z == 3):
            axes[0,0].set_title('Train Loss from Top Max Iteration Results', fontsize=14, fontweight='bold')
            axes[0,1].set_title('Value Loss from Top Max Iteration Results', fontsize=14, fontweight='bold')
            axes[1,0].set_title('Train FS from Top Max Iteration Results', fontsize=14, fontweight='bold')
            axes[1,1].set_title('Value FS from Top Max Iteration Results', fontsize=14, fontweight='bold')

        for ax in axes[0, :]: 
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)

        for ax in axes[1, :]: 
            ax.set_xlabel('Iterations')
            ax.set_ylabel('FS Score')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


        
def main():
    parser = ArgumentParser()
    parser.add_argument("input_path", type=str, help="Relative path to input data (csv file).")
    parser.add_argument("target_path", type=str, help="Relative path to target data (csv file).")
    args = parser.parse_args()

    # Get input and target data
    input = pd.read_csv(args.input_path)
    target = pd.read_csv(args.target_path)

    # Drop unnecessary features
    input.drop("bwd_URG_flag_count", axis=1, inplace=True)

    # Transfer input and target data to numpy arrays
    input = input.to_numpy()
    target = target.to_numpy()

    # Shuffle input and target
    shuffle_index = np.random.permutation(input.shape[0])
    input = input[shuffle_index]
    target = target[shuffle_index]

    # Split data into training, validation and testing
    X_train = input[:98000, :]
    y_train = target[:98000, :]
    X_val = input[98000:110000, :]
    y_val = target[98000:110000, :]
    X_test = input[110000:, :]
    y_test = target[110000:, :]

    # Create tensors for data
    X_train_t = torch.tensor(X_train).float()
    y_train_t = torch.tensor(y_train).float()
    X_val_t = torch.tensor(X_val).float()
    y_val_t = torch.tensor(y_val).float()
    X_test_t = torch.tensor(X_test).float()
    y_test_t = torch.tensor(y_test).float()
    
    test = Testing()
    test.input_dim  = X_train_t.shape[1]
    test.output_dim = y_train.shape[1]

    test.X_train = X_train_t
    test.y_train = y_train_t
    test.X_val = X_val_t
    test.y_val = y_val_t
    test.X_val = X_test_t
    test.y_val = y_test_t

    results, best = test.hyperparameter_tuning()
    test.best_each(results)

    device  = torch.device("cpu")
    model = train.GeneralNN(test.input_dim, best['cfg']['hidden'], test.output_dim)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=best['cfg']['lr'])
    train.train_SGD(model, criterion, optimizer, X_train_t, y_train_t, X_val_t, y_val_t, best['cfg']['iters'], best['cfg']['bs'], test.check_every)
    
    final_test_f1 = train.calculate_f_score(model, X_test_t, y_test_t)
    print(f"accuracy on testing data:           {final_test_f1:.4f}")

if __name__ == "__main__":
    main()