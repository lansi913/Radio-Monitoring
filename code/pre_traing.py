import os
import sys
import argparse
import torch
import numpy as np
import random
from bokeh.plotting import figure, show, output_notebook
from bokeh.io import output_file, save
from Module.Bone import mask_input_signal_with_no_grad
from base_model import RadarPretrainingModel
from Module.Loss import global_reconstruction_loss, local_reconstruction_loss, expert_balance_loss
from bokeh.layouts import gridplot
from Module.Dataloader import SignalDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from thop import profile
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR


def set_seed(seed):
    """Set all related random seeds to ensure reproducibility of experiments"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


RANDOM_SEED = 42
set_seed(RANDOM_SEED)
print(f">>> Global random seed set to: {RANDOM_SEED} <<<")


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


start_time = datetime.now().strftime('%Y%m%d_%H%M')
print(f"Start Time: {start_time}")

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"CUDA available, found {device_count} GPUs")
    gpu_ids = [0] if device_count >= 1 else list(range(device_count))
    if len(gpu_ids) > 1:
        device = torch.device(f"cuda:{gpu_ids[0]}")
        print(f"Using Multi-GPU Parallel Computing: GPU {gpu_ids}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    else:
        device = torch.device(f"cuda:{gpu_ids[0]}")
        print(f"Using Single GPU Computing: GPU {gpu_ids[0]}")
else:
    device = torch.device("cpu")
    print("CUDA unavailable, using CPU...")


def get_args():
    parser = argparse.ArgumentParser(
        description="Radar Pre-training with Multi-GPU")
    parser.add_argument('--dataset_path', type=str, default=r'./Data/pretraining/pretraining_set.pt',
                        help='Path to the training dataset .pt file')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save models, logs, and plots')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int,
                        default=512, help='Batch size per GPU')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()


args = get_args()
train_dataset = SignalDataset(args.dataset_path)
print('Dataset size: ', len(train_dataset))

batch_size = 512
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

mask_ratio = 0.1
lr = 1e-4
input_dim_long = 1
input_dim_short = 10
inout_seq_len_long = 1500
inout_seq_len_short = 150
hidden_dim = 128
token_dim = 512
num_heads = 16
num_tokens = 16
expert_depth = 20
top_k_long = 8
top_k_short = 5

model = RadarPretrainingModel(
    input_dim_long, input_dim_short, inout_seq_len_long, inout_seq_len_short,
    hidden_dim, token_dim, num_heads, num_tokens, expert_depth, top_k_long, top_k_short
).to(device)

if torch.cuda.is_available() and len(gpu_ids) > 1:
    model_for_training = nn.DataParallel(
        model, device_ids=gpu_ids, output_device=gpu_ids[0], dim=0)
    print(f"Model configured for Multi-GPU Parallel Mode using: {gpu_ids}")
else:
    model_for_training = model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


total_params = count_parameters(model)
print(f"Total Model Parameters: {total_params:,}")
print(f"Model Parameter Size (MB): {total_params * 4 / (1024 * 1024):.2f} MB")

try:
    dummy_input_long = torch.randn(1, input_dim_long, inout_seq_len_long)
    dummy_input_short = torch.randn(1, input_dim_short, inout_seq_len_short)
    model_cpu = model.cpu()
    flops, params = profile(model_cpu, inputs=(
        dummy_input_long, dummy_input_short, 1), verbose=False)
    print(f"Model FLOPs: {flops:,}")
    print(f"Model FLOPs (G): {flops / 1e9:.2f} G")
    model = model.to(device)
except Exception as e:
    print(f"FLOPs calculation failed: {e}")
    model = model.to(device)

model = model_for_training
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
losses = []

decoded_signals_long = {}
masked_signals_long = {}
original_signals_long = {}
decoded_signals_short = {}
masked_signals_short = {}
original_signals_short = {}

num_epochs = 50
epoch_interest = [10, 20, 30, 40, 50]

ENABLE_EARLY_STOPPING = False
early_stopping = None
if ENABLE_EARLY_STOPPING:
    early_stopping = EarlyStopping(patience=10, verbose=True, delta=1000)
    print(">>> Early Stopping Enabled <<<")
else:
    print(">>> Early Stopping Disabled <<<")

total_steps = num_epochs * len(train_loader)
scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps)
print(f">>> Strategy: [OneCycleLR] enabled, max_lr={lr} <<<")

print(f"Starting training for {num_epochs} epochs...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
        for batch_idx, batch in enumerate(pbar):
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_description(
                f"Epoch {epoch+1}/{num_epochs} [LR={current_lr:.6f}]")

            input_signal = batch.to(device)
            masked_signal, mask = mask_input_signal_with_no_grad(
                input_signal, mask_ratio=mask_ratio)
            masked_signal = masked_signal.to(device)
            mask = mask.to(device)
            batch_size_curr = input_signal.size(0)

            x_long = input_signal.reshape(
                batch_size_curr, input_dim_long, inout_seq_len_long)
            mask_long = masked_signal.reshape(
                batch_size_curr, input_dim_long, inout_seq_len_long)
            x_short = input_signal.reshape(
                batch_size_curr, input_dim_short, inout_seq_len_short)
            mask_short = masked_signal.reshape(
                batch_size_curr, input_dim_short, inout_seq_len_short)

            mask_long = mask_long.to(device)
            mask_short = mask_short.to(device)
            x_long = x_long.to(device)
            x_short = x_short.to(device)

            long_decoded, short_decoded, moe_output_long, moe_output_short = model(
                mask_long, mask_short, batch_size_curr)

            if long_decoded.shape[0] != x_long.shape[0]:
                repeat_factor = long_decoded.shape[0] // x_long.shape[0]
                x_long = x_long.repeat(repeat_factor, 1, 1)

            if short_decoded.shape[0] != x_short.shape[0]:
                repeat_factor = short_decoded.shape[0] // x_short.shape[0]
                x_short = x_short.repeat(repeat_factor, 1, 1)

            loss_long = global_reconstruction_loss(
                x_long, long_decoded, sigma_g=torch.ones_like(x_long))
            loss_short = local_reconstruction_loss(
                x_short, short_decoded, sigma_l=torch.ones_like(x_short))

            loss_expert_balance_long = expert_balance_loss(
                alpha_1=0.1,
                token_expert_map=torch.argmax(moe_output_long, dim=-1),
                expert_scores=moe_output_long,
                num_experts=moe_output_long.size(1),
                num_tokens=num_tokens,
            )
            loss_expert_balance_short = expert_balance_loss(
                alpha_1=0.1,
                token_expert_map=torch.argmax(moe_output_short, dim=-1),
                expert_scores=moe_output_short,
                num_experts=moe_output_short.size(1),
                num_tokens=num_tokens,
            )

            loss = 3 * loss_long + loss_short

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss
            pbar.set_postfix({"Batch Loss": f"{total_loss.item():.4f}"})

    losses.append(total_loss.detach().cpu().numpy())

    if epoch + 1 in epoch_interest:
        decoded_signals_long[epoch +
                             1] = long_decoded[0].detach().cpu().squeeze().numpy()
        masked_signals_long[epoch +
                            1] = mask_long[0].detach().cpu().squeeze().numpy()
        original_signals_long[epoch +
                              1] = x_long[0].detach().cpu().squeeze().numpy()

        decoded_signals_short[epoch +
                              1] = short_decoded[0].detach().cpu().squeeze().numpy()
        masked_signals_short[epoch +
                             1] = mask_short[0].detach().cpu().squeeze().numpy()
        original_signals_short[epoch +
                               1] = x_short[0].detach().cpu().squeeze().numpy()

    print(f"Epoch {epoch + 1}, Loss: {total_loss}")

    if ENABLE_EARLY_STOPPING:
        early_stopping(total_loss.item())
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

save_dir = './results'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
torch.save(model.state_dict(), os.path.join(
    save_dir, 'Mask_pre_radar_v3_'+str(start_time)+'.pth'))

output_file(os.path.join(
    save_dir, "radar_loss_curve_" + str(start_time)+".html"))
p = figure(title="Training Loss Curve", x_axis_label="Epoch",
           y_axis_label="Loss", width=800, height=400)
p.line(list(range(1, len(losses)+1)), losses,
       legend_label="Total Loss", line_width=2, color="blue")
p.legend.location = "top_right"
save(p)

plots = []
output_file(os.path.join(save_dir, "MLM_"+str(start_time)+".html"))

for epoch in decoded_signals_long.keys():
    p_original_long = figure(
        title=f"Long Original Signal (Epoch {epoch})", x_axis_label="Time", y_axis_label="Amplitude", width=400, height=300)
    p_original_long.line(list(range(1, inout_seq_len_long+1)),
                         original_signals_long[epoch], legend_label="Original", line_width=2, color="green")

    p_masked_long = figure(
        title=f"Long Masked Signal (Epoch {epoch})", x_axis_label="Time", y_axis_label="Amplitude", width=400, height=300)
    p_masked_long.line(list(range(1, inout_seq_len_long+1)),
                       masked_signals_long[epoch], legend_label="Masked", line_width=2, color="orange")

    p_decoded_long = figure(
        title=f"Long Decoded Signal (Epoch {epoch})", x_axis_label="Time", y_axis_label="Amplitude", width=400, height=300)
    p_decoded_long.line(list(range(1, inout_seq_len_long+1)),
                        decoded_signals_long[epoch], legend_label="Decoded", line_width=2, color="red")

    p_original_short = figure(
        title=f"Short Original Signal (Epoch {epoch})", x_axis_label="Time", y_axis_label="Amplitude", width=400, height=300)
    p_original_short.line(list(range(1, inout_seq_len_short+1)),
                          original_signals_short[epoch][0], legend_label="Original", line_width=2, color="green")

    p_masked_short = figure(
        title=f"Short Masked Signal (Epoch {epoch})", x_axis_label="Time", y_axis_label="Amplitude", width=400, height=300)
    p_masked_short.line(list(range(1, inout_seq_len_short+1)),
                        masked_signals_short[epoch][0], legend_label="Masked", line_width=2, color="orange")

    p_decoded_short = figure(
        title=f"Short Decoded Signal (Epoch {epoch})", x_axis_label="Time", y_axis_label="Amplitude", width=400, height=300)
    p_decoded_short.line(list(range(1, inout_seq_len_short+1)),
                         decoded_signals_short[epoch][0], legend_label="Decoded", line_width=2, color="red")

    plots.append([p_original_long, p_masked_long, p_decoded_long,
                 p_original_short, p_masked_short, p_decoded_short])

grid = gridplot([
    [p_original_long, p_masked_long, p_decoded_long],
    [p_original_short, p_masked_short, p_decoded_short]
])

save(grid)
print(
    f"Test results saved to: {os.path.join(save_dir, 'model_test_results.html')}")
