import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
from tqdm import tqdm
from base_model import RadarPretrainingModel
from torchvision.ops.focal_loss import sigmoid_focal_loss
from peft import LoraConfig, get_peft_model
from scipy.signal import resample
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from collections import Counter
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

start_time = datetime.now().strftime('%Y%m%d_%H%M')

print(f"Start Time: {start_time}")

lr = 1e-8
input_dim_long = 1
input_dim_short = 10
inout_seq_len_long = 1500
inout_seq_len_short = 150
hidden_dim = 128
token_dim = 512
num_heads = 16
num_tokens = 16
expert_depth = 10
top_k_long = 50
top_k_short = 25
num_epochs = 200


log_dir = f'LoRa_finetune_logs_{start_time}'

writer = SummaryWriter(log_dir=log_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
backbone = RadarPretrainingModel(
    input_dim_long, input_dim_short, inout_seq_len_long, inout_seq_len_short, hidden_dim, token_dim, num_heads, num_tokens, expert_depth, top_k_long, top_k_short
)


def load_filtered_state_dict(model, state_dict_path, map_location):
    checkpoint = torch.load(state_dict_path, map_location=map_location)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    filtered_state_dict = {k: v for k, v in state_dict.items()
                           if 'total_ops' not in k and 'total_params' not in k}

    model.load_state_dict(filtered_state_dict, strict=False)
    return model


backbone = load_filtered_state_dict(backbone,
                                    r'./Mask_pre_radar_v3_20250910_1947.pth', map_location=device)
backbone.to(device)
backbone.eval()


def print_model_modules(model):
    for name, module in model.named_modules():
        print(name)


print("Model modules:")
print_model_modules(backbone)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=[
        "attention_layers.0.out_proj",
        "attention_layers.1.out_proj",
        "attention_layers.2.out_proj",
        "long_moe.shared_ffn",
        "short_moe.shared_ffn",
        "long_moe.router",
        "short_moe.router",
        "output_projection",
        "liear_embeb",
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="FEATURE_EXTRACTION"
)
backbone = get_peft_model(backbone, lora_config)


class RadarFinetuneModel(nn.Module):
    def __init__(self, backbone, feature_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids):
        long_decoded, short_decoded, moe_output_long, moe_output_short = self.backbone(
            input_ids)
        long_flat = long_decoded.reshape(long_decoded.size(0), -1)
        short_flat = short_decoded.reshape(short_decoded.size(0), -1)
        moe_long_flat = moe_output_long.reshape(moe_output_long.size(0), -1)
        moe_short_flat = moe_output_short.reshape(moe_output_short.size(0), -1)
        x = torch.cat(
            [long_flat, short_flat, moe_long_flat, moe_short_flat], dim=1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


nor_pt_path = r'./Data/finetuning/Ruijin/Ruijin_Nor_Normal.pt'
nor_data = torch.load(nor_pt_path, weights_only=False)
nor_num = max(1, int(len(nor_data) * 0.065))
random.seed(42)
nor_indices = random.sample(range(len(nor_data)), nor_num)
normal_samples = [{'signal': nor_data[i], 'label': 'Normal'}
                  for i in nor_indices]

PVC_pt_path = r'./Data/finetuning/Xinhua/Xinhua_PVC.pt'

arrhythmia_samples = []
arrhythmia_paths = {
    'PVC': PVC_pt_path,
}

for condition, path in arrhythmia_paths.items():
    try:
        print(f"Loading: {condition}")
        data = torch.load(path, weights_only=False)
        arrhythmia_samples.extend(
            [{'signal': sample, 'label': condition} for sample in data])
        print(f"Successfully loaded {condition} data: {len(data)} samples")
    except:
        print(f"Failed to load {condition} data from: {path}")

all_samples = normal_samples + arrhythmia_samples

print(f"Normal samples count: {len(normal_samples)}")
print(f"Arrhythmia samples count: {len(arrhythmia_samples)}")
print(f"Total samples count: {len(all_samples)}")

labels_list = ['Normal', 'PVC']
label2id = {'Normal': 0, 'PVC': 1}
num_classes = len(label2id)

label_counts = Counter([sample['label'] for sample in all_samples])
print("\nCategory Distribution:")
for label, count in label_counts.items():
    print(f"{label}: {count} samples ({count/len(all_samples)*100:.2f}%)")


class NormalAbnormalDataset(Dataset):
    def __init__(self, samples, label2id):
        self.samples = samples
        self.label2id = label2id
        self.num_classes = len(label2id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        signal = self.samples[idx]['signal']
        if len(signal) != 1500:
            signal = torch.tensor(resample(signal, 1500), dtype=torch.float32)
        else:
            signal = torch.tensor(signal, dtype=torch.float32)
        label_idx = self.label2id[self.samples[idx]['label']]
        label_onehot = F.one_hot(torch.tensor(
            label_idx), num_classes=self.num_classes).float()
        return signal, label_onehot


full_dataset = NormalAbnormalDataset(all_samples, label2id)

total_len = len(full_dataset)
train_len = int(total_len * 0.6)
val_len = int(total_len * 0.2)
test_len = total_len - train_len - val_len
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    full_dataset, [train_len, val_len,
                   test_len], generator=torch.Generator().manual_seed(42)
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

feature_dim = 70584
model = RadarFinetuneModel(backbone, feature_dim, num_classes).to(device)

optimizer = torch.optim.Adam(
    list(model.fc.parameters()) + list(backbone.parameters()), lr=lr)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-8)

model.train()
for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0
    with tqdm(train_loader, desc=f"Finetune Epoch {epoch+1}/{num_epochs}") as pbar:
        for batch in pbar:
            input_signal, labels = batch
            input_signal = input_signal.to(device)
            labels = labels.to(device)
            batch_size = input_signal.size(0)
            mask_long = input_signal.reshape(
                batch_size, input_dim_long, inout_seq_len_long)
            mask_short = input_signal.reshape(
                batch_size, input_dim_short, inout_seq_len_short)
            optimizer.zero_grad()
            logits = model((mask_long, mask_short, batch_size))
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            labels_idx = torch.argmax(labels, dim=1)
            correct += (preds == labels_idx).sum().item()
            total += batch_size
            pbar.set_postfix(
                {"Batch Loss": f"{loss.item():.4f}", "Acc": f"{correct/total:.4f}"})
    train_acc = correct / total
    train_loss = total_loss / len(train_loader)
    scheduler.step()
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_signal, labels = batch
            input_signal = input_signal.to(device)
            labels = labels.to(device)
            batch_size = input_signal.size(0)
            mask_long = input_signal.reshape(
                batch_size, input_dim_long, inout_seq_len_long)
            mask_short = input_signal.reshape(
                batch_size, input_dim_short, inout_seq_len_short)
            logits = model((mask_long, mask_short, batch_size))
            loss = criterion(logits, labels)
            val_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            labels_idx = torch.argmax(labels, dim=1)
            val_correct += (preds == labels_idx).sum().item()
            val_total += batch_size
    val_acc = val_correct / val_total
    val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    model.train()
    writer.add_scalar(
        f'Loss/LoRa_batch{batch_size}_time_{start_time}_epoch_{num_epochs}/train', train_loss, epoch + 1)
    writer.add_scalar(
        f'Loss/LoRa_batch{batch_size}_time_{start_time}_epoch_{num_epochs}/val', val_loss, epoch + 1)
    writer.add_scalar(
        f'Acc/LoRa_batch{batch_size}_time_{start_time}_epoch_{num_epochs}/train', train_acc, epoch + 1)
    writer.add_scalar(
        f'Acc/LoRa_batch{batch_size}_time_{start_time}_epoch_{num_epochs}/val', val_acc, epoch + 1)

class_names = labels_list
model.eval()
test_loss = 0
test_correct = 0
test_total = 0
all_preds = []
all_labels = []
all_inputs = []
with torch.no_grad():
    for batch in test_loader:
        input_signal, labels = batch
        input_signal = input_signal.to(device)
        labels = labels.to(device)
        batch_size = input_signal.size(0)
        mask_long = input_signal.reshape(
            batch_size, input_dim_long, inout_seq_len_long)
        mask_short = input_signal.reshape(
            batch_size, input_dim_short, inout_seq_len_short)
        logits = model((mask_long, mask_short, batch_size))
        loss = criterion(logits, labels)
        test_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        labels_idx = torch.argmax(labels, dim=1)
        test_correct += (preds == labels_idx).sum().item()
        test_total += batch_size
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_idx.cpu().numpy())
        all_inputs.extend(input_signal.cpu().numpy())
test_acc = test_correct / test_total
test_loss = test_loss / len(test_loader)
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

os.makedirs('xinhua_tune_results/data', exist_ok=True)

np.save(f'xinhua_tune_results/data/results_{start_time}.npy', {
    'y_true': all_labels,
    'y_pred': all_preds,
    'class_names': class_names
})

cm = confusion_matrix(all_labels, all_preds)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

plt.figure(figsize=(14, 12))
sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', vmin=0, vmax=100,
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={'size': 32, 'weight': 'bold'}, cbar_kws={"fraction": 0.046, "pad": 0.04})
ax = plt.gca()
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=32)
cbar.set_ticks([0, 20, 40, 60, 80, 100])
plt.xlabel('Predicted label', fontsize=32, fontweight='bold')
plt.ylabel('True label', fontsize=32, fontweight='bold')
plt.xticks(rotation=45, fontsize=32)
plt.yticks(rotation=45, fontsize=32)
plt.tight_layout()
plt.savefig('xinhua_tune_results/LoRa_tune_xinhua_percentage_' +
            str(start_time)+'+.png', dpi=300, bbox_inches='tight')
plt.close()
