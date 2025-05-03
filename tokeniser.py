import os
import json
import numpy as np
import torch
from scipy.fft import dct, idct
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)

class TemporalBPEProcessor:
    def __init__(
        self,
        scale: float = 100.0,
        min_token: int = 0,
        time_horizon: int | None = None,
        state_dim: int | None = None,
        normalization: str = "zscore",  # 'zscore' or 'minmax'
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        min_val: np.ndarray | None = None,
        max_val: np.ndarray | None = None,
    ):
        self.scale = scale
        self.min_token = min_token
        self.time_horizon = time_horizon
        self.state_dim = state_dim
        self.normalization = normalization
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val
        self.called_time_horizon = time_horizon
        self.called_state_dim = state_dim
        self.T = 16

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.normalization == "zscore":
            return (x - self.mean) / self.std
        elif self.normalization == "minmax":
            mnmx = self.max_val - self.min_val
            mnmx[mnmx<=1e-2]= 1
            return (x - self.min_val) / mnmx
        return x

    def _denormalize(self, x: np.ndarray) -> np.ndarray:
        if self.normalization == "zscore":
            return x * self.std + self.mean
        elif self.normalization == "minmax":
            mnmx = self.max_val - self.min_val
            mnmx[mnmx<=1e-2]= 1
            return x * (mnmx) + self.min_val
        return x

    def __call__(
        self,
        state_seq: np.ndarray,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
    ):
        """
        Tokenize a batch of state sequences via DCT quantization.

        Args:
            state_seq: np.ndarray of shape [B, T, D] or [T, D]
            padding: pad option (currently not used)
            truncation: truncate option (currently not used)
            max_length: length to pad or truncate to if return_tensors='pt'
            return_tensors: 'pt' for PyTorch tensors
        """
        if state_seq.ndim == 2:
            state_seq = state_seq[None, ...]

        batch_size, T, D = state_seq.shape
        self.called_time_horizon = T
        self.called_state_dim = D

        norm_seq = self.normalize(state_seq)
        print('yo2',norm_seq.shape)
        coeff = dct(norm_seq, axis=1, norm='ortho')
        q = np.around(coeff * self.scale).astype(int)

        tokens: list[list[int]] = []
        for b in range(batch_size):
            flat = (q[b].flatten() - self.min_token).clip(min=0)
            tokens.append(flat.tolist())

        if return_tensors == 'pt':
            if max_length is None:
                max_length = max(len(t) for t in tokens)
            input_ids = torch.tensor(
                [t[:max_length] + [0] * max(0, max_length - len(t)) for t in tokens],
                dtype=torch.long
            )
            attention_mask = torch.tensor(
                [[1] * min(len(t), max_length) + [0] * max(0, max_length - len(t)) for t in tokens],
                dtype=torch.long
            )
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        #print(tokens)

        return torch.tensor(tokens)+1

    def decode(
        self,
        token_ids: list[list[int]],
        state_dim: int | None = None,
    ) -> np.ndarray:
        D = state_dim or self.called_state_dim
        token_ids = (token_ids-1).clip(min = 0)
        decoded_seqs = []
        for ids in token_ids:
            arr = np.array(ids) + self.min_token
            arr = arr.reshape(-1, D)
            coeff = arr.astype(float) / self.scale
            rec = idct(coeff, axis=0, norm='ortho')
            rec = self._denormalize(rec)
            decoded_seqs.append(rec)
        return np.stack(decoded_seqs)

    @classmethod
    def fit_from_npz(
        cls,
        data_dir: str,
        num_episodes: int = -1,
        scale: float = 10.0,
        normalization: str = "zscore"
    ) -> "TemporalBPEProcessor":
        meta_path = os.path.join(data_dir, 'meta.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        episodes = meta['episodes']
        if num_episodes == -1 or num_episodes > len(episodes):
            num_episodes = len(episodes)

        seqs = []
        for ep in episodes:
            arr = np.load(os.path.join(data_dir, ep['file']))
            pose_seq = arr['pose']
            grip_seq = arr['grip'][..., None]
            # if len(seqs)==0:
            #     seqs=np.concatenate([pose_seq, grip_seq], axis=1)
            x = np.concatenate([pose_seq, grip_seq], axis=1)
            seqs.append(x)#np.concatenate([seqs, x], axis=0)
        
        all_states = np.concatenate(seqs, axis=0)
        print(len(all_states))
        mean = all_states.mean(axis=0)
        std = all_states.std(axis=0)
        std[std <= 1e-5] = 1.0
        min_val = all_states.min(axis=0)
        max_val = all_states.max(axis=0)

        if normalization == "zscore":
            normed_seqs = [(s - mean) / std for s in seqs]
        else:
            normed_seqs = [(s - min_val) / (max_val - min_val + 1e-8) for s in seqs]

        # Determine token range without BPE
        all_coeffs = [dct(s, axis=0, norm='ortho').flatten() for s in normed_seqs]
        scaled_vals = np.around(np.concatenate(all_coeffs) * scale)
        min_token = 0#int(scaled_vals.min())

        T, D = seqs[0].shape
        print('done')
        return cls(
            scale=scale,
            min_token=min_token,
            time_horizon=T,
            state_dim=D,
            normalization=normalization,
            mean=mean,
            std=std,
            min_val=min_val,
            max_val=max_val,
        )

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        np.savez(
            os.path.join(save_dir, "normalization_stats.npz"),
            mean=self.mean,
            std=self.std,
            min_val=self.min_val,
            max_val=self.max_val,
            scale=self.scale,
            min_token=self.min_token,
            time_horizon=self.called_time_horizon,
            state_dim=self.called_state_dim,
            normalization=self.normalization,
        )

    @classmethod
    def load(cls, save_dir: str) -> "TemporalBPEProcessor":
        stats = np.load(os.path.join(save_dir, "normalization_stats.npz"))
        return cls(
            scale=float(stats["scale"]),
            min_token=int(stats["min_token"]),
            time_horizon=int(stats["time_horizon"]),
            state_dim=int(stats["state_dim"]),
            normalization=str(stats["normalization"]),
            mean=stats["mean"],
            std=stats["std"],
            min_val=stats["min_val"],
            max_val=stats["max_val"],
        )

    def plot_coeff_histogram(self, data_dir: str, episode_index: int, bins: int = 50):
        meta_path = os.path.join(data_dir, 'meta.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        ep = meta['episodes'][episode_index]
        arr = np.load(os.path.join(data_dir, ep['file']))
        pose_seq = arr['pose']
        grip_seq = arr['grip'][..., None]
        seq = np.concatenate([pose_seq, grip_seq], axis=1)

        norm_seq = self.normalize(seq)
        coeffs = dct(norm_seq, axis=0, norm='ortho').flatten()
        plt.figure(figsize=(8, 4))
        plt.hist(coeffs, bins=bins)
        plt.title(f"Normalized DCT Coeffs Histogram — Episode {episode_index}")
        plt.xlabel("Normalized Coefficient")
        plt.ylabel("Count")
        plt.grid(True)
        plt.show()

def train_tokeniser(data_dir: str = 'out_dataset'):
    processor = TemporalBPEProcessor.fit_from_npz(
        data_dir, num_episodes=-1, scale=10.0, normalization="minmax"#"zscore"
    )
    processor.save("saved_processor")


def eval_tokeniser(processor: TemporalBPEProcessor, data_dir: str = 'out_dataset', num_samples: int = 8):
    meta_path = os.path.join(data_dir, 'meta.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    real_seq = []
    ep = meta['episodes'][0]
    arr = np.load(os.path.join(data_dir, ep['file']))
    pose_seq = arr['pose']
    grip_seq = arr['grip'][..., None]
    for j in range(num_samples):
        i = np.random.randint(0, len(pose_seq)-processor.T)
        seq = np.concatenate([pose_seq[i:i+processor.T], grip_seq[i:i+processor.T]], axis=1)
        real_seq.append(seq)
    real_seq = np.array(real_seq)
    print(real_seq.shape)
    tokens= processor(real_seq)
    print(tokens)
    recon_seq = processor.decode(tokens)[0]
    ss_res = ((real_seq - recon_seq) ** 2).sum()
    ss_tot = ((real_seq - real_seq.mean()) ** 2).sum()
    r2_score = 1 - (ss_res / ss_tot)
    print(f"R² Score: {r2_score:.6f}")
