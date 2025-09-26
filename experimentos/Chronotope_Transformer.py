# -*- coding: utf-8 -*-
"""
Chronotope Transformer v2 - Forced Dynamics Edition (Fixed)
Complete implementation with forced pathway dynamics and memory usage
"""

import os
import sys
import requests
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import json
import time
import pickle
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

# Check if in notebook
def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' in get_ipython().config:
            return True
    except:
        pass
    return False

# ============================================================================
# CONFIGURATION V2 - WITH FORCED DYNAMICS
# ============================================================================

@dataclass
class ChronotopeConfigV2:
    """Configuration with forced dynamics and anti-stagnation mechanisms."""

    # Dataset
    dataset: str = 'enwik8'
    data_path: Optional[str] = None

    # Training - base
    batch_size: int = 8
    block_size: int = 256
    max_iters: int = 5000
    eval_interval: int = 100
    eval_iters: int = 20

    # Learning
    learning_rate: float = 1e-5
    min_lr: float = 1e-6
    warmup_iters: int = 100

    # Regularization - ENHANCED
    dropout: float = 0.35  # Increased
    weight_decay: float = 0.15  # Increased
    gradient_clip: float = 0.5

    # Early stopping - NEW
    patience: int = 5
    min_delta: float = 0.001

    # Architecture
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    memory_size: int = 16  # Reduced to force usage

    # DYNAMICS FORCING - NEW APPROACH
    dynamics_mode: str = 'oscillatory'  # 'oscillatory', 'adaptive', 'random'
    oscillation_period: int = 100
    oscillation_amplitude: float = 0.3
    dynamics_loss_weight: float = 0.5  # Much stronger
    memory_loss_weight: float = 0.3

    # Pathway specialization - NEW
    force_specialization: bool = True
    specialization_weight: float = 0.2
    min_pathway_variance: float = 0.05  # Minimum acceptable variance

    # Anti-stagnation - NEW
    stagnation_window: int = 20
    stagnation_penalty: float = 1.0
    equilibrium_penalty: float = 0.5  # Penalty for 50/50 split

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    experiment_name: str = 'chronotope_v2_forced'
    seed: int = 42

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

# ============================================================================
# DATASET MANAGEMENT
# ============================================================================

class DatasetManager:
    """Manages dataset loading and preprocessing."""

    DATASET_URLS = {
        'enwik8': 'http://mattmahoney.net/dc/enwik8.zip',
        'text8': 'http://mattmahoney.net/dc/text8.zip',
        'shakespeare_char': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    }

    def __init__(self, config: ChronotopeConfigV2):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def download_dataset(self, dataset_name: str, data_dir: str = 'data') -> str:
        os.makedirs(data_dir, exist_ok=True)
        filepath = os.path.join(data_dir, dataset_name)

        if os.path.exists(filepath) and not os.path.isdir(filepath):
            self.logger.info(f"Dataset {dataset_name} already exists")
            return filepath

        url = self.DATASET_URLS.get(dataset_name)
        if not url:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        self.logger.info(f"Downloading {dataset_name}")

        if dataset_name in ['enwik8', 'text8']:
            r = requests.get(url, stream=True)
            with open(filepath + '.zip', 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            os.system(f'unzip -p {filepath}.zip > {filepath}')
            os.remove(f'{filepath}.zip')
        else:
            r = requests.get(url)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(r.text)

        return filepath

    def load_dataset(self) -> Tuple[torch.Tensor, Dict]:
        if self.config.data_path:
            data_path = self.config.data_path
        else:
            data_path = self.download_dataset(self.config.dataset)

        meta_path = os.path.join('data', f'{self.config.dataset}_meta.pkl')

        if os.path.exists(meta_path):
            self.logger.info(f"Loading vocabulary from cache")
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
        else:
            self.logger.info(f"Building vocabulary")

            encoding = 'latin-1' if self.config.dataset in ['enwik8', 'text8'] else 'utf-8'
            with open(data_path, 'r', encoding=encoding) as f:
                text = f.read()
                if self.config.dataset in ['enwik8', 'text8']:
                    text = text[:10000000]

            chars = sorted(list(set(text)))
            vocab_size = len(chars)
            stoi = {ch: i for i, ch in enumerate(chars)}
            itos = {i: ch for i, ch in enumerate(chars)}

            meta = {
                'vocab_size': vocab_size,
                'stoi': stoi,
                'itos': itos,
                'dataset': self.config.dataset
            }

            with open(meta_path, 'wb') as f:
                pickle.dump(meta, f)

        encoding = 'latin-1' if self.config.dataset in ['enwik8', 'text8'] else 'utf-8'
        with open(data_path, 'r', encoding=encoding) as f:
            text = f.read()
            if self.config.dataset in ['enwik8', 'text8']:
                text = text[:10000000]

        encode = lambda s: [meta['stoi'][c] for c in s]
        data = torch.tensor(encode(text), dtype=torch.long)

        return data, meta

# ============================================================================
# DYNAMICS ENFORCEMENT LOSSES
# ============================================================================

class DynamicsEnforcementLoss(nn.Module):
    """Forces dynamic behavior instead of static equilibrium."""

    def __init__(self, config: ChronotopeConfigV2):
        super().__init__()
        self.config = config
        self.pathway_history = deque(maxlen=config.stagnation_window)
        self.step_counter = 0

    def forward(self, fast_weights: torch.Tensor, slow_weights: torch.Tensor,
                step: int) -> Tuple[torch.Tensor, Dict]:
        """
        Calculates losses that force dynamic behavior.
        """
        losses = {}
        device = fast_weights.device

        # Current balance
        current_balance = fast_weights.mean()
        self.pathway_history.append(current_balance.item())

        # 1. ANTI-STAGNATION LOSS
        if len(self.pathway_history) >= self.config.stagnation_window:
            variance = np.var(list(self.pathway_history))
            if variance < self.config.min_pathway_variance:
                # Heavily penalize low variance
                stagnation_penalty = self.config.stagnation_penalty * \
                                   (self.config.min_pathway_variance - variance) / \
                                   self.config.min_pathway_variance
                losses['stagnation'] = torch.tensor(stagnation_penalty, device=device)

        # 2. OSCILLATION ENCOURAGEMENT
        if self.config.dynamics_mode == 'oscillatory':
            # Target oscillates between favoring fast and slow
            phase = 2 * math.pi * step / self.config.oscillation_period
            target = 0.5 + self.config.oscillation_amplitude * math.sin(phase)
            oscillation_loss = (current_balance - target) ** 2
            losses['oscillation'] = self.config.dynamics_loss_weight * oscillation_loss

        elif self.config.dynamics_mode == 'random':
            # Random target to force exploration
            if step % 20 == 0:
                self.random_target = 0.2 + 0.6 * np.random.random()
            target = getattr(self, 'random_target', 0.5)
            random_loss = (current_balance - target) ** 2
            losses['random'] = self.config.dynamics_loss_weight * random_loss

        # 3. EQUILIBRIUM PENALTY
        # Strongly penalize perfect 50/50 balance
        equilibrium_distance = abs(current_balance.item() - 0.5)
        if equilibrium_distance < 0.05:  # Too close to equilibrium
            equilibrium_penalty = self.config.equilibrium_penalty * \
                                (0.05 - equilibrium_distance) / 0.05
            losses['equilibrium'] = torch.tensor(equilibrium_penalty, device=device)

        # 4. SPECIALIZATION ENFORCEMENT
        if self.config.force_specialization:
            # Reward when pathways have different activations
            difference = torch.abs(fast_weights - slow_weights).mean()
            if difference < 0.2:  # Not different enough
                specialization_loss = self.config.specialization_weight * (0.2 - difference)
                losses['specialization'] = specialization_loss

        # Sum all losses
        total_loss = sum(losses.values()) if losses else torch.tensor(0.0, device=device)

        return total_loss, losses


class MemoryUsageEnforcer(nn.Module):
    """Forces model to actively use memory through auxiliary tasks."""

    def __init__(self, config: ChronotopeConfigV2):
        super().__init__()
        self.config = config
        self.memory_predictor = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, memory_output: torch.Tensor,
                original_features: torch.Tensor,
                memory_weights: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Creates auxiliary tasks that require memory usage.
        """
        losses = {}
        device = memory_output.device

        # 1. RECONSTRUCTION TASK
        # Memory should help reconstruct input features
        reconstructed = self.memory_predictor(memory_output)
        reconstruction_loss = F.mse_loss(reconstructed, original_features.detach())
        losses['memory_reconstruction'] = self.config.memory_loss_weight * reconstruction_loss

        # 2. UTILIZATION ENFORCEMENT
        max_weight = memory_weights.max(dim=-1)[0].mean()
        if max_weight < 0.1:  # Less than 10% utilization
            utilization_penalty = (0.1 - max_weight) ** 2
            losses['memory_underuse'] = 0.3 * utilization_penalty

        # 3. DIVERSITY ENFORCEMENT
        # Penalize always accessing the same memory slots
        entropy = -torch.sum(memory_weights * torch.log(memory_weights + 1e-8), dim=-1)
        avg_entropy = entropy.mean()
        if avg_entropy < 1.0:  # Low entropy = not diverse
            diversity_loss = (1.0 - avg_entropy)
            losses['memory_diversity'] = 0.1 * diversity_loss

        total_loss = sum(losses.values()) if losses else torch.tensor(0.0, device=device)

        return total_loss, losses

# ============================================================================
# ENHANCED MODEL COMPONENTS
# ============================================================================

class ForcedDualPathway(nn.Module):
    """Dual pathways with forced specialization."""

    def __init__(self, config: ChronotopeConfigV2):
        super().__init__()
        self.config = config

        # FAST pathway - high frequency processing
        self.fast_path = nn.Sequential(
            nn.LayerNorm(config.n_embd),
            nn.Linear(config.n_embd, config.n_embd),
            nn.GELU(),  # Sharp nonlinearity
            nn.Dropout(config.dropout),
        )

        # SLOW pathway - low frequency processing
        self.slow_path = nn.Sequential(
            nn.LayerNorm(config.n_embd),
            nn.Linear(config.n_embd, config.n_embd),
            nn.Tanh(),  # Smooth nonlinearity
            nn.Dropout(config.dropout),
        )

        # Additional slow processing
        self.slow_conv = nn.Conv1d(config.n_embd, config.n_embd,
                                   kernel_size=3, padding=1, groups=1)

        # Gating mechanism with initial bias
        self.gate = nn.Sequential(
            nn.Linear(config.n_embd * 2, config.n_embd),
            nn.LayerNorm(config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, 2)
        )

        # Initialize with bias away from equilibrium
        with torch.no_grad():
            self.gate[-1].bias.data = torch.tensor([1.0, -1.0])

    def forward(self, x: torch.Tensor, force_pathway: Optional[str] = None) -> Tuple[torch.Tensor, Dict]:
        B, T, C = x.shape

        # Fast pathway
        fast_out = self.fast_path(x)

        # Slow pathway with convolution
        slow_base = self.slow_path(x)
        x_conv = x.transpose(1, 2)  # [B, C, T]
        slow_conv = self.slow_conv(x_conv).transpose(1, 2)  # [B, T, C]
        slow_out = slow_base + 0.5 * slow_conv  # Combine

        # Gating
        if force_pathway == 'fast':
            fast_weight = torch.ones(B, 1, 1, device=x.device)
            slow_weight = torch.zeros(B, 1, 1, device=x.device)
        elif force_pathway == 'slow':
            fast_weight = torch.zeros(B, 1, 1, device=x.device)
            slow_weight = torch.ones(B, 1, 1, device=x.device)
        else:
            # Dynamic gating based on content
            gate_input = torch.cat([
                fast_out.mean(dim=1),
                slow_out.mean(dim=1)
            ], dim=-1)
            gate_logits = self.gate(gate_input)
            gate_weights = F.softmax(gate_logits, dim=-1)

            fast_weight = gate_weights[:, 0:1].unsqueeze(1)
            slow_weight = gate_weights[:, 1:2].unsqueeze(1)

        # Weighted combination
        output = fast_weight * fast_out + slow_weight * slow_out

        stats = {
            'fast_weight': fast_weight.mean().item(),
            'slow_weight': slow_weight.mean().item(),
            'fast_out': fast_out,
            'slow_out': slow_out
        }

        return output, fast_weight.squeeze(), slow_weight.squeeze(), stats


class EnhancedMemory(nn.Module):
    """Memory module with forced usage mechanisms - FIXED."""

    def __init__(self, config: ChronotopeConfigV2):
        super().__init__()
        self.config = config

        # Memory bank as parameter (not buffer) to avoid in-place issues
        self.memory_bank = nn.Parameter(torch.randn(config.memory_size, config.n_embd) * 0.1)

        # Memory operations
        self.query_proj = nn.Linear(config.n_embd, config.n_embd)
        self.key_proj = nn.Linear(config.n_embd, config.n_embd)
        self.value_proj = nn.Linear(config.n_embd, config.n_embd)

        # Output gating
        self.memory_gate = nn.Sequential(
            nn.Linear(config.n_embd * 2, config.n_embd),
            nn.LayerNorm(config.n_embd),
            nn.Sigmoid()
        )

        # Memory update network (for training)
        self.memory_updater = nn.Linear(config.n_embd, config.n_embd)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, T, C = x.shape

        # Query memory with attention mechanism
        queries = self.query_proj(x.mean(dim=1))  # [B, C]
        keys = self.key_proj(self.memory_bank)  # [M, C]
        values = self.value_proj(self.memory_bank)  # [M, C]

        # Attention scores
        scores = torch.matmul(queries, keys.T) / math.sqrt(C)
        weights = F.softmax(scores, dim=-1)  # [B, M]

        # Retrieve from memory
        retrieved = torch.matmul(weights, values)  # [B, C]
        retrieved_expanded = retrieved.unsqueeze(1).expand(-1, T, -1)

        # Gate and combine
        gate_input = torch.cat([x, retrieved_expanded], dim=-1)
        gate = self.memory_gate(gate_input)
        output = x + gate * retrieved_expanded

        # NO IN-PLACE UPDATES - let gradient flow naturally
        # Memory will be updated through backprop on the Parameter

        output = self.dropout(output)

        stats = {
            'memory_weights': weights,
            'memory_output': output,
            'original_features': x,
            'memory_utilization': weights.max(dim=-1)[0].mean().item()
        }

        return output, stats


class StabilizedAttention(nn.Module):
    """Multi-head attention with stability features."""

    def __init__(self, config: ChronotopeConfigV2):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)

        return out


class ChronotopeBlockV2(nn.Module):
    """Enhanced Chronotope block with forced dynamics."""

    def __init__(self, config: ChronotopeConfigV2):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        self.attention = StabilizedAttention(config)
        self.dual_pathway = ForcedDualPathway(config)
        self.memory = EnhancedMemory(config)

        self.ffwd = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x, collect_stats=True):
        # Attention
        x = x + self.attention(self.ln1(x))

        # Dual pathways with forced dynamics
        pathway_out, fast_w, slow_w, pathway_stats = self.dual_pathway(x)
        x = x + pathway_out

        # Memory with forced usage
        memory_out, memory_stats = self.memory(x)
        x = memory_out

        # Feed-forward
        x = x + self.ffwd(self.ln2(x))

        stats = {
            **pathway_stats,
            **memory_stats,
            'fast_weight_tensor': fast_w,
            'slow_weight_tensor': slow_w
        }

        return x, stats


class ChronotopeTransformerV2(nn.Module):
    """Chronotope Transformer with forced dynamics."""

    def __init__(self, config: ChronotopeConfigV2, vocab_size: int):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            ChronotopeBlockV2(config) for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size)

        # Loss modules
        self.dynamics_loss = DynamicsEnforcementLoss(config)
        self.memory_loss = MemoryUsageEnforcer(config)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None, step=0):
        B, T = idx.shape
        device = idx.device

        # Embeddings
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = self.dropout(tok_emb + pos_emb)

        # Process through blocks
        all_stats = []
        all_fast_weights = []
        all_slow_weights = []

        for block in self.blocks:
            x, stats = block(x)
            all_stats.append(stats)
            if 'fast_weight_tensor' in stats:
                all_fast_weights.append(stats['fast_weight_tensor'])
                all_slow_weights.append(stats['slow_weight_tensor'])

        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Calculate losses
        total_loss = None
        loss_breakdown = {}

        if targets is not None:
            # Language modeling loss
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            lm_loss = F.cross_entropy(logits_flat, targets_flat)
            loss_breakdown['lm_loss'] = lm_loss.item()
            total_loss = lm_loss

            # Dynamics enforcement loss
            if all_fast_weights:
                fast_avg = torch.stack([w.mean() for w in all_fast_weights])
                slow_avg = torch.stack([w.mean() for w in all_slow_weights])
                dynamics_loss, dynamics_breakdown = self.dynamics_loss(fast_avg, slow_avg, step)
                total_loss = total_loss + dynamics_loss
                for k, v in dynamics_breakdown.items():
                    loss_breakdown[f'd_{k}'] = v.item() if hasattr(v, 'item') else v

            # Memory enforcement loss
            if all_stats and 'memory_weights' in all_stats[0]:
                # Use first layer's memory stats
                memory_loss, memory_breakdown = self.memory_loss(
                    all_stats[0]['memory_output'],
                    all_stats[0]['original_features'],
                    all_stats[0]['memory_weights']
                )
                total_loss = total_loss + memory_loss
                for k, v in memory_breakdown.items():
                    loss_breakdown[f'm_{k}'] = v.item() if hasattr(v, 'item') else v

        return logits, total_loss, all_stats, loss_breakdown

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate text."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            logits, _, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# ============================================================================
# EARLY STOPPING AND EVALUATION
# ============================================================================

class EarlyStoppingCallback:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True  # Improvement
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


class EnhancedEvaluator:
    """Comprehensive evaluation with pathway analysis."""

    def __init__(self, model, config, train_data, val_data, meta):
        self.model = model
        self.config = config
        self.train_data = train_data
        self.val_data = val_data
        self.meta = meta
        self.device = config.device

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.pathway_variance_history = []
        self.memory_util_history = []
        self.loss_components = defaultdict(list)
        self.pathway_stats = defaultdict(list)
        self.memory_stats = defaultdict(list)

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.config.block_size, (self.config.batch_size,))
        x = torch.stack([data[i:i+self.config.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.config.block_size+1] for i in ix])
        return x.to(self.device), y.to(self.device)

    @torch.no_grad()
    def estimate_loss(self, num_batches=None):
        if num_batches is None:
            num_batches = self.config.eval_iters

        out = {}
        self.model.eval()

        for split in ['train', 'val']:
            losses = torch.zeros(num_batches)
            for k in range(num_batches):
                X, Y = self.get_batch(split)
                logits, loss, _, _ = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()

        self.model.train()
        return out

    def comprehensive_evaluate(self, step):
        """Full evaluation with all metrics."""
        losses = self.estimate_loss()
        self.train_losses.append(losses['train'])
        self.val_losses.append(losses['val'])

        # Get detailed stats
        self.model.eval()
        X, Y = self.get_batch('val')
        _, _, stats, loss_breakdown = self.model(X, Y, step=step)

        # Track components
        for key, value in loss_breakdown.items():
            self.loss_components[key].append(value)

        # Calculate pathway variance
        variance = 0
        if stats and 'fast_weight' in stats[0]:
            fast_weights = [s['fast_weight'] for s in stats]
            slow_weights = [s['slow_weight'] for s in stats]
            variance = np.var(fast_weights)
            self.pathway_variance_history.append(variance)

            # Track pathway stats
            self.pathway_stats['fast_weight'].append(np.mean(fast_weights))
            self.pathway_stats['slow_weight'].append(np.mean(slow_weights))
            self.pathway_stats['balance'].append(abs(np.mean(fast_weights) - np.mean(slow_weights)))

        # Memory utilization
        util = 0
        if stats and 'memory_utilization' in stats[0]:
            util = np.mean([s['memory_utilization'] for s in stats])
            self.memory_util_history.append(util)
            self.memory_stats['utilization'].append(util)

        self.model.train()

        return {
            'train_loss': losses['train'],
            'val_loss': losses['val'],
            'pathway_variance': variance,
            'memory_utilization': util,
            'loss_breakdown': loss_breakdown
        }

    def generate_samples(self, num_samples=3, max_tokens=200):
        """Generate text samples at different temperatures."""
        self.model.eval()
        decode = lambda l: ''.join([self.meta['itos'][i] for i in l])

        samples = []
        temperatures = [0.5, 0.8, 1.0]

        for temp in temperatures[:num_samples]:
            context = torch.zeros((1, 1), dtype=torch.long, device=self.device)
            generated = self.model.generate(context, max_new_tokens=max_tokens,
                                          temperature=temp, top_k=40)
            text = decode(generated[0].tolist())
            samples.append({'temperature': temp, 'text': text})

        self.model.train()
        return samples

    def plot_training_curves(self, save_path='training_curves.png'):
        """Plot comprehensive training curves."""
        if len(self.train_losses) < 2:
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train', alpha=0.8)
        axes[0, 0].plot(self.val_losses, label='Validation', alpha=0.8)
        axes[0, 0].set_xlabel('Evaluation Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Progress')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Pathway weights
        if self.pathway_stats['fast_weight']:
            axes[0, 1].plot(self.pathway_stats['fast_weight'], label='Fast', color='red', alpha=0.8)
            axes[0, 1].plot(self.pathway_stats['slow_weight'], label='Slow', color='blue', alpha=0.8)
            axes[0, 1].set_xlabel('Evaluation Steps')
            axes[0, 1].set_ylabel('Weight')
            axes[0, 1].set_title('Pathway Dynamics')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Memory utilization
        if self.memory_stats['utilization']:
            axes[0, 2].plot(self.memory_stats['utilization'], color='green', alpha=0.8)
            axes[0, 2].set_xlabel('Evaluation Steps')
            axes[0, 2].set_ylabel('Utilization')
            axes[0, 2].set_title('Memory Utilization')
            axes[0, 2].grid(True, alpha=0.3)

        # Validation loss detail
        axes[1, 0].plot(self.val_losses, color='orange', alpha=0.8)
        axes[1, 0].set_xlabel('Evaluation Steps')
        axes[1, 0].set_ylabel('Validation Loss')
        axes[1, 0].set_title('Validation Loss Detail')
        axes[1, 0].grid(True, alpha=0.3)

        # Pathway balance
        if self.pathway_stats['balance']:
            axes[1, 1].plot(self.pathway_stats['balance'], color='purple', alpha=0.8)
            axes[1, 1].set_xlabel('Evaluation Steps')
            axes[1, 1].set_ylabel('|Fast - Slow|')
            axes[1, 1].set_title('Pathway Balance')
            axes[1, 1].grid(True, alpha=0.3)

        # Pathway variance
        if self.pathway_variance_history:
            axes[1, 2].plot(self.pathway_variance_history, color='brown', alpha=0.8)
            axes[1, 2].set_xlabel('Evaluation Steps')
            axes[1, 2].set_ylabel('Variance')
            axes[1, 2].set_title('Pathway Variance')
            axes[1, 2].axhline(y=self.config.min_pathway_variance, color='red',
                              linestyle='--', alpha=0.5, label='Target')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)

        plt.suptitle(f'Chronotope Transformer Training - {self.config.experiment_name}')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
        return fig

    def save_metrics(self, filepath='metrics.json'):
        """Save all metrics to JSON."""
        metrics = {
            'train_losses': [float(x) for x in self.train_losses],
            'val_losses': [float(x) for x in self.val_losses],
            'pathway_stats': {k: [float(x) for x in v] for k, v in self.pathway_stats.items()},
            'memory_stats': {k: [float(x) for x in v] for k, v in self.memory_stats.items()},
            'pathway_variance': [float(x) for x in self.pathway_variance_history],
            'memory_utilization': [float(x) for x in self.memory_util_history],
            'loss_components': {k: [float(x) for x in v] for k, v in self.loss_components.items()},
            'final_metrics': {
                'best_val_loss': float(min(self.val_losses)) if self.val_losses else float('inf'),
                'final_train_loss': float(self.train_losses[-1]) if self.train_losses else float('inf'),
                'final_val_loss': float(self.val_losses[-1]) if self.val_losses else float('inf'),
                'final_pathway_variance': float(self.pathway_variance_history[-1]) if self.pathway_variance_history else 0,
                'final_memory_util': float(self.memory_util_history[-1]) if self.memory_util_history else 0,
            },
            'config': self.config.to_dict()
        }

        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {filepath}")

    def print_summary(self):
        """Print evaluation summary."""
        if not self.val_losses:
            return

        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Best Validation Loss: {min(self.val_losses):.4f}")
        print(f"Final Train Loss: {self.train_losses[-1]:.4f}")
        print(f"Final Validation Loss: {self.val_losses[-1]:.4f}")

        if self.pathway_stats['fast_weight']:
            print(f"\nPathway Statistics:")
            print(f"  Final Fast Weight: {self.pathway_stats['fast_weight'][-1]:.3f}")
            print(f"  Final Slow Weight: {self.pathway_stats['slow_weight'][-1]:.3f}")
            print(f"  Mean Balance: {np.mean(self.pathway_stats['balance']):.3f}")
            print(f"  Final Variance: {self.pathway_variance_history[-1]:.4f}")
            print(f"  Target Variance: >{self.config.min_pathway_variance:.4f}")

        if self.memory_stats['utilization']:
            print(f"\nMemory Statistics:")
            print(f"  Mean Utilization: {np.mean(self.memory_stats['utilization']):.3f}")
            print(f"  Final Utilization: {self.memory_stats['utilization'][-1]:.3f}")

        # Success/Warning messages
        print("\n" + "-"*60)
        if self.pathway_variance_history and self.pathway_variance_history[-1] > self.config.min_pathway_variance:
            print("SUCCESS: Achieved dynamic pathway behavior!")
        else:
            print("WARNING: Pathways remained relatively static")

        if self.memory_util_history and self.memory_util_history[-1] > 0.10:
            print("SUCCESS: Memory is being actively used!")
        else:
            print("WARNING: Memory underutilized")

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_v2(config: Optional[ChronotopeConfigV2] = None):
    """Training with forced dynamics and comprehensive evaluation."""

    if config is None:
        config = ChronotopeConfigV2()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{config.experiment_name}.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting Chronotope V2 training with FORCED DYNAMICS")

    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Load data
    dataset_manager = DatasetManager(config)
    data, meta = dataset_manager.load_dataset()

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    logger.info(f"Dataset: {config.dataset}")
    logger.info(f"Train: {len(train_data)/1e6:.2f}M, Val: {len(val_data)/1e6:.2f}M tokens")
    logger.info(f"Vocabulary size: {meta['vocab_size']}")

    # Initialize model
    model = ChronotopeTransformerV2(config, meta['vocab_size'])
    model = model.to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {total_params/1e6:.2f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.99)
    )

    # Initialize helpers
    evaluator = EnhancedEvaluator(model, config, train_data, val_data, meta)
    early_stopping = EarlyStoppingCallback(config.patience, config.min_delta)

    # Learning rate schedule
    def get_lr(step):
        if step < config.warmup_iters:
            return config.learning_rate * step / config.warmup_iters
        if step > config.max_iters:
            return config.min_lr
        decay_ratio = (step - config.warmup_iters) / (config.max_iters - config.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return config.min_lr + coeff * (config.learning_rate - config.min_lr)

    # Training loop
    best_val_loss = float('inf')
    train_start = time.time()

    for step in range(config.max_iters):
        t0 = time.time()

        # Update learning rate
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Training step
        X, Y = evaluator.get_batch('train')
        logits, loss, stats, loss_breakdown = model(X, Y, step=step)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()

        dt = (time.time() - t0) * 1000

        # Logging
        if step % 10 == 0:
            logger.info(f"step {step}: loss={loss.item():.4f}, lr={lr:.2e}, time={dt:.2f}ms")

        # Evaluation
        if step % config.eval_interval == 0 and step > 0:
            metrics = evaluator.comprehensive_evaluate(step)

            logger.info(f"eval {step}: train={metrics['train_loss']:.4f}, "
                       f"val={metrics['val_loss']:.4f}, "
                       f"var={metrics['pathway_variance']:.4f}, "
                       f"mem={metrics['memory_utilization']:.3f}")

            # Early stopping check
            if early_stopping(metrics['val_loss']):
                best_val_loss = metrics['val_loss']
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config.to_dict(),
                    'val_loss': best_val_loss
                }, f'{config.experiment_name}_best.pt')
                logger.info(f"  -> Saved best model (val={best_val_loss:.4f})")

            if early_stopping.should_stop:
                logger.info("Early stopping triggered!")
                break

            # Generate samples periodically
            if step % 500 == 0:
                logger.info("\n--- Sample Generation ---")
                samples = evaluator.generate_samples(num_samples=1, max_tokens=200)
                for sample in samples:
                    logger.info(f"Temperature {sample['temperature']}:")
                    logger.info(sample['text'][:200])
                logger.info("--- End Sample ---\n")

    # Training complete
    train_time = (time.time() - train_start) / 60
    logger.info(f"\nTraining complete in {train_time:.2f} minutes")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")

    # Final evaluation and visualization
    evaluator.print_summary()

    # Generate final samples
    logger.info("\n" + "="*60)
    logger.info("FINAL TEXT GENERATION SAMPLES")
    logger.info("="*60)

    samples = evaluator.generate_samples(num_samples=3, max_tokens=300)
    for sample in samples:
        logger.info(f"\n--- Temperature {sample['temperature']} ---")
        logger.info(sample['text'])

    # Save metrics and plot curves
    evaluator.save_metrics(f'{config.experiment_name}_metrics.json')
    evaluator.plot_training_curves(f'{config.experiment_name}_curves.png')

    logger.info(f"\nAll results saved with prefix: {config.experiment_name}")

    return model, evaluator

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__" or in_notebook():
    config = ChronotopeConfigV2(
        dataset='shakespeare_char',
        max_iters=5000,
        eval_interval=100,

        # Force dynamics
        dynamics_mode='oscillatory',
        oscillation_period=100,
        oscillation_amplitude=0.3,
        dynamics_loss_weight=0.5,

        # Force memory usage
        memory_size=16,
        memory_loss_weight=0.3,

        # Anti-stagnation
        min_pathway_variance=0.05,
        stagnation_penalty=1.0,
        equilibrium_penalty=0.5,

        # Early stopping
        patience=5,
        min_delta=0.001,

        experiment_name='chronotope_v2_forced'
    )

    print("="*60)
    print("CHRONOTOPE V2 - FORCED DYNAMICS EDITION")
    print("="*60)
    print(f"Configuration:")
    print(f"  Dataset: {config.dataset}")
    print(f"  Max iterations: {config.max_iters}")
    print(f"  Dynamics mode: {config.dynamics_mode}")
    print(f"  Target variance: >{config.min_pathway_variance}")
    print("="*60)

    model, evaluator = train_v2(config)
    print("\nTraining complete. Model and results saved.")