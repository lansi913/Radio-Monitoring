from torch.utils.data import Dataset
import torch
from scipy.signal import resample


class SignalDataset(Dataset):
    def __init__(self, data_path, target_length=1500):
        dataset_tensors = torch.load(data_path)
        processed_samples = [sample.unsqueeze(0) for sample in dataset_tensors]

        if target_length is not None:
            processed_samples = [self.resample(
                sample, target_length) for sample in processed_samples]
        self.data = processed_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def resample(self, signal, target_length):
        current_length = signal.shape[1]
        if current_length == target_length:
            return signal
        else:
            print(f'Resampling, current length: {current_length}')
            signal_np = signal.squeeze(0).cpu().numpy()
            resampled_np = resample(signal_np, target_length)
            resampled_signal = torch.tensor(
                resampled_np, dtype=signal.dtype, device=signal.device).unsqueeze(0)
            return resampled_signal
