"""The [R3-8] LSTM, its training draws, fitting, scoring and the worker pool.

One model class for every arm: a one-layer left-to-right LSTM over the [R3-1] symbols that
predicts every interval symbol and EOS. Arms differ only in their training data, start symbol and
initialisation.

Training draws: an epoch is as many draws as there are training codas. Each draw picks clan c
with probability proportional to n_c ** alpha (n_c = the clan's training codas), then one of the
clan's training codas uniformly, independently and with replacement. A single-clan model uses the
same sampler with one clan. The draws come from a generator seeded with the fit's seed and are
used by nothing else, so two fits with the same seed and the same training codas see the same
codas in the same order whatever their start symbol.

Determinism: every fit runs on one torch thread with deterministic algorithms (as residual.py).
Embeddings and the loss are one-hot products, elementwise only, so they are deterministic on
every device.
"""
import concurrent.futures
import contextlib
import copy
import hashlib
import math
import multiprocessing
import os
import time

import numpy as np
import torch
from torch import nn

from .data import N_OUTPUTS

LOG2E = 1.0 / math.log(2.0)
# nn.LSTM's cuDNN determinism note allows :16:8 and :4096:2; the reproducibility page allows :16:8 and
# :4096:8 (the same tuple as residual.py's CUBLAS_SETTINGS).
CUBLAS_SETTINGS = (':16:8', ':4096:2', ':4096:8')

_DATA = {}


class CodaLSTM(nn.Module):
    """Embedding (one-hot product), dropout, one LSTM layer, dropout, linear output over 21 symbols."""

    def __init__(self, n_inputs, embedding_dim, hidden, dropout):
        super().__init__()
        self.n_inputs = n_inputs
        self.embedding = nn.Parameter(torch.empty(n_inputs, embedding_dim))
        nn.init.normal_(self.embedding)  # nn.Embedding's default initialisation
        self.lstm = nn.LSTM(embedding_dim, hidden, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden, N_OUTPUTS)

    def forward(self, x):
        e = nn.functional.one_hot(x, self.n_inputs).to(self.embedding.dtype) @ self.embedding
        h, _ = self.lstm(self.dropout(e))
        return self.output(self.dropout(h))


def position_bits(logits, y):
    """log2 p(y_t | prefix) at every position, through a one-hot product."""
    log_probabilities = nn.functional.log_softmax(logits, dim=-1)
    return (log_probabilities * nn.functional.one_hot(y, N_OUTPUTS).to(logits.dtype)).sum(dim=-1) * LOG2E


def state_sha256(state_dict):
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode() + str(tuple(tensor.shape)).encode() + str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def sampling_probabilities(sizes, alpha):
    """q_c proportional to n_c ** alpha over the clans with training codas."""
    sizes = np.asarray(sizes, dtype=np.float64)
    if np.any(sizes <= 0):
        raise ValueError('Every sampled clan needs training codas')
    weights = sizes ** alpha
    return weights / weights.sum()


class Draws:
    """The training draws of one fit: clan by q, coda uniformly within the clan, with replacement."""

    def __init__(self, train_idx, clan_index, alpha, seed):
        train_idx = np.asarray(train_idx, dtype=np.int64)
        clans = np.unique(clan_index[train_idx])
        self.per_clan = [train_idx[clan_index[train_idx] == c] for c in clans]
        self.sizes = np.array([len(p) for p in self.per_clan], dtype=np.int64)
        self.q = sampling_probabilities(self.sizes, alpha)
        self.flat = torch.from_numpy(np.concatenate(self.per_clan))
        self.offsets = torch.from_numpy(np.concatenate([[0], np.cumsum(self.sizes)[:-1]]).astype(np.int64))
        self.sizes_t = torch.from_numpy(self.sizes.astype(np.float64))
        self.q_t = torch.from_numpy(self.q)
        self.n = int(self.sizes.sum())
        self.generator = torch.Generator().manual_seed(seed)

    def epoch(self):
        clan = torch.multinomial(self.q_t, self.n, replacement=True, generator=self.generator)
        u = torch.rand(self.n, generator=self.generator, dtype=torch.float64)
        within = torch.minimum((u * self.sizes_t[clan]).floor().long(), self.sizes_t[clan].long() - 1)
        return self.flat[self.offsets[clan] + within]


def _tensors(device):
    key = ('tensors', device)
    if key not in _DATA:
        data = _DATA['arrays']
        _DATA[key] = {name: torch.from_numpy(data[name]).to(device)
                      for name in ('x_untagged', 'x_tagged', 'y', 'lengths', 'clan_index')}
    return _DATA[key]


def score_codas(model, index, tagged, device, batch_size):
    """Per-coda log2 probability, log2 p(EOS | intervals) (float64) and number of predicted symbols
    scored, in the order of index."""
    t = _tensors(device)
    x_all = t['x_tagged'] if tagged else t['x_untagged']
    index = torch.as_tensor(np.asarray(index, dtype=np.int64)).to(device)
    lls, eos, counts = [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(index), batch_size):
            b = index[start:start + batch_size]
            lengths = t['lengths'][b]
            width = int(lengths.max())
            x, y = x_all[b, :width], t['y'][b, :width]
            bits = position_bits(model(x), y).double()
            mask = (torch.arange(width, device=device)[None, :] < lengths[:, None]).double()
            lls.append((bits * mask).sum(dim=1).cpu())
            eos.append(bits[torch.arange(len(b), device=device), lengths - 1].cpu())
            counts.append(mask.sum(dim=1).long().cpu())
    if not lls:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int64)
    return torch.cat(lls).numpy(), torch.cat(eos).numpy(), torch.cat(counts).numpy()


def validation_sums(model, index, tagged, device, batch_size, clan_index):
    ll, _, _ = score_codas(model, index, tagged, device, batch_size)
    n = _DATA['arrays']['lengths'][index]
    clans = clan_index[index]
    by_clan = {int(c): (float(ll[clans == c].sum()), int(n[clans == c].sum())) for c in np.unique(clans)}
    return float(ll.sum()), int(n.sum()), by_clan


def fit(task):
    """Train one model with one seed.

    With ``epochs`` None the fit stops early on inner-validation bits per symbol (evaluated before
    any step and after every epoch; the best epoch is restored); otherwise it trains exactly
    ``epochs`` epochs (a refit) and keeps the final weights. Returns the training record and, for
    ``score_idx``, the per-coda scores."""
    started = time.time()
    device = task['device']
    arrays = _DATA['arrays']
    clan_index = arrays['clan_index']
    t = _tensors(device)
    tagged = task['tagged']
    x_all = t['x_tagged'] if tagged else t['x_untagged']
    torch.manual_seed(task['seed'])
    model = CodaLSTM(arrays['n_inputs'], task['embedding_dim'], task['hidden'], task['dropout'])
    init_sha256 = state_sha256(model.state_dict())
    if task.get('init_checkpoint'):
        state = torch.load(task['init_checkpoint'], map_location='cpu', weights_only=True)
        if state_sha256(state) != task['init_checkpoint_sha256']:
            raise ValueError(f"Checkpoint {task['init_checkpoint']} differs from the one recorded")
        model.load_state_dict(state)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=task['learning_rate'])
    draws = Draws(task['train_idx'], clan_index, task['alpha'], task['seed'])
    val_idx, epochs = task.get('val_idx'), task.get('epochs')
    early_stopping = epochs is None
    if early_stopping and (val_idx is None or len(val_idx) == 0):
        raise ValueError('Early stopping needs inner-validation codas')
    eval_batch = task['eval_batch_size']
    curve, draw_hashes, epoch_times = [], [], []

    def validate():
        ll, n, by_clan = validation_sums(model, val_idx, tagged, device, eval_batch, clan_index)
        return dict(bits_per_symbol=ll / n, ll_bits=ll, n_symbols=n, by_clan=by_clan)

    best = best_state = None
    best_epoch, since_best, epoch = 0, 0, 0
    if early_stopping:
        best = validate()
        best_state = copy.deepcopy(model.state_dict())
        curve.append(best['bits_per_symbol'])
    max_epochs = task['max_epochs'] if early_stopping else epochs
    lengths_all, y_all = t['lengths'], t['y']
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = draws.epoch()
        if task.get('record_draws', True):
            draw_hashes.append(hashlib.sha256(order.numpy().tobytes()).hexdigest()[:16])
        order = order.to(device)
        for start in range(0, draws.n, task['batch_size']):
            b = order[start:start + task['batch_size']]
            lengths = lengths_all[b]
            width = int(lengths.max())
            mask = (torch.arange(width, device=device)[None, :] < lengths[:, None]).to(torch.float32)
            bits = position_bits(model(x_all[b, :width]), y_all[b, :width])
            loss = -(bits * mask).sum() / mask.sum() / LOG2E  # mean cross-entropy per symbol, nats
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_times.append(time.time() - started)
        if early_stopping:
            current = validate()
            curve.append(current['bits_per_symbol'])
            if current['bits_per_symbol'] > best['bits_per_symbol']:
                best, best_epoch, since_best = current, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                since_best += 1
                if since_best >= task['patience']:
                    break
    if early_stopping:
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    result = dict(key=task['key'], best_epoch=best_epoch, epochs_trained=epoch,
                  early_stopped=early_stopping and since_best >= task['patience'],
                  curve=curve, draw_hashes=draw_hashes, init_sha256=init_sha256,
                  state_sha256=state_sha256(final_state), n_train=draws.n,
                  sampling_q=draws.q.tolist(), device=device, pid=os.getpid())
    if early_stopping:
        result['inner_validation'] = best
    if task.get('score_idx') is not None:
        ll, eos, counts = score_codas(model, task['score_idx'], tagged, device, eval_batch)
        result['scores'] = dict(ll_bits=ll, ll_eos_bits=eos, n_symbols=counts)
    if task.get('save_checkpoint'):
        torch.save(final_state, task['save_checkpoint'])
        result['checkpoint'] = task['save_checkpoint']
    result['seconds'] = time.time() - started
    result['epoch_times'] = epoch_times
    result['draws'] = draws.n * len(epoch_times)
    return result


# ---------------------------------------------------------------- running fits

@contextlib.contextmanager
def deterministic_single_thread():
    threads, deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)


def require_cuda_determinism():
    if not torch.cuda.is_available():
        raise ValueError('CUDA requested but not available')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in CUBLAS_SETTINGS:
        raise ValueError(f'Set CUBLAS_WORKSPACE_CONFIG to one of {CUBLAS_SETTINGS} before starting the workers')


def _init_worker(arrays, device):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if device == 'cuda':
        require_cuda_determinism()
    _DATA.clear()
    _DATA['arrays'] = arrays


def _run(task):
    return fit(task)


def _warm(device):
    _tensors(device)
    time.sleep(0.5)
    return os.getpid()


def warm_pool(pool):
    """Start every worker (spawn, initializer, data on its device) before anything is timed."""
    if pool.pool is not None:
        for future in [pool.pool.submit(_warm, pool.device) for _ in range(pool.workers)]:
            future.result()
    else:
        _tensors(pool.device)


class _Deferred:
    def __init__(self, task):
        self.task = task

    def result(self):
        with deterministic_single_thread():
            return fit(self.task)


class FitPool:
    """Fits in this process (workers=0) or in a spawn-context process pool with one torch thread
    per worker and deterministic algorithms, so results don't depend on the worker count.
    Results come back in submission order."""

    def __init__(self, arrays, workers, device='cpu'):
        self.arrays, self.workers, self.device, self.pool = arrays, workers, device, None

    def __enter__(self):
        if self.workers > 0:
            self.pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.workers, mp_context=multiprocessing.get_context('spawn'),
                initializer=_init_worker, initargs=(self.arrays, self.device))
        else:
            if self.device == 'cuda':
                require_cuda_determinism()
            _DATA.clear()
            _DATA['arrays'] = self.arrays
        return self

    def __exit__(self, *exc):
        if self.pool is not None:
            self.pool.shutdown(cancel_futures=True)

    def submit(self, tasks):
        for task in tasks:
            if task['device'] != self.device:
                raise ValueError(f"Task for {task['device']} submitted to a {self.device} pool")
        if self.pool is None:
            return [_Deferred(t) for t in tasks]
        return [self.pool.submit(_run, t) for t in tasks]

    def run(self, tasks, order_by_cost=True):
        """Run tasks, longest first when order_by_cost (results still in submission order)."""
        if not tasks:
            return []
        order = sorted(range(len(tasks)), key=lambda i: -tasks[i].get('cost', 0)) if order_by_cost else range(len(tasks))
        futures = dict(zip(order, self.submit([tasks[i] for i in order])))
        return [futures[i].result() for i in range(len(tasks))]
