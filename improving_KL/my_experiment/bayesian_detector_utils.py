import abc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


def pad_g_values(g_values, length):
    pad_size = length - len(g_values)
    g_dtype = g_values.dtype
    return np.concatenate([g_values, np.zeros((pad_size, 30), dtype=g_dtype)])


def process_data_for_training(data, truncation_length, torch_device, validate_size=0.3):

    # Step 1: filter and truncate
    filtered_data = []
    for i in range(len(data)):
        if len(data[i]['output_ids'][0]) >= truncation_length:
            data[i]['g_values'] = data[i]['g_values'][:truncation_length]
            data[i]['mc_g_values'] = data[i]['mc_g_values'][:truncation_length]
            data[i]['r_values'] = data[i]['r_values'][:truncation_length]
            data[i]['context_repetition_mask'] = data[i]['context_repetition_mask'][:truncation_length]
            filtered_data.append(data[i])
        else:
            pad_size = truncation_length - len(data[i]['output_ids'][0])
            data[i]['g_values'] = pad_g_values(data[i]['g_values'], truncation_length)
            data[i]['mc_g_values'] = pad_g_values(data[i]['mc_g_values'], truncation_length)
            data[i]['r_values'] = np.concatenate([data[i]['r_values'], np.zeros(pad_size)])
            data[i]['context_repetition_mask'] = np.concatenate([data[i]['context_repetition_mask'], np.ones(pad_size)])
            filtered_data.append(data[i])
    print(f'Filtered training data size: {len(filtered_data)}')

    wm_g_values = np.array([np.stack([filtered_data[i]['g_values'], filtered_data[i]['mc_g_values']], axis=1) for i in range(len(filtered_data))])
    wm_r_values = np.array([filtered_data[i]['r_values'] for i in range(len(filtered_data))])
    wm_skipped = np.array([1-filtered_data[i]['context_repetition_mask'] for i in range(len(filtered_data))])
    wm_labels = np.ones(len(filtered_data), dtype=np.float32)
    # Use simulation data for unwatermarked data
    uwm_g_values = np.random.randint(0, 2, size=wm_g_values.shape)
    uwm_r_values = np.random.rand(*wm_r_values.shape)
    uwm_skipped = np.ones(wm_skipped.shape)
    uwm_labels = np.zeros(len(filtered_data), dtype=np.float32)

    # Step 2: Train/test split
    # We'll use the same indices for all related arrays
    wm_idx = np.arange(len(wm_labels))
    uwm_idx = np.arange(len(uwm_labels))

    wm_train_idx, wm_cv_idx = train_test_split(wm_idx, test_size=validate_size)
    uwm_train_idx, uwm_cv_idx = train_test_split(uwm_idx, test_size=validate_size)

    # Step 3: Split the arrays
    def split_data(arr, train_idx, test_idx):
        return arr[train_idx], arr[test_idx]

    wm_g_train, wm_g_cv = split_data(wm_g_values, wm_train_idx, wm_cv_idx)
    wm_r_train, wm_r_cv = split_data(wm_r_values, wm_train_idx, wm_cv_idx)
    wm_s_train, wm_s_cv = split_data(wm_skipped, wm_train_idx, wm_cv_idx)
    wm_l_train, wm_l_cv = split_data(wm_labels, wm_train_idx, wm_cv_idx)

    uwm_g_train, uwm_g_cv = split_data(uwm_g_values, uwm_train_idx, uwm_cv_idx)
    uwm_r_train, uwm_r_cv = split_data(uwm_r_values, uwm_train_idx, uwm_cv_idx)
    uwm_s_train, uwm_s_cv = split_data(uwm_skipped, uwm_train_idx, uwm_cv_idx)
    uwm_l_train, uwm_l_cv = split_data(uwm_labels, uwm_train_idx, uwm_cv_idx)

    # Step 4: Concatenate wm and uwm data
    g_train = np.concatenate([wm_g_train, uwm_g_train], axis=0)
    r_train = np.concatenate([wm_r_train, uwm_r_train], axis=0)
    s_train = np.concatenate([wm_s_train, uwm_s_train], axis=0)
    l_train = np.concatenate([wm_l_train, uwm_l_train], axis=0)

    g_cv = np.concatenate([wm_g_cv, uwm_g_cv], axis=0)
    r_cv = np.concatenate([wm_r_cv, uwm_r_cv], axis=0)
    s_cv = np.concatenate([wm_s_cv, uwm_s_cv], axis=0)
    l_cv = np.concatenate([wm_l_cv, uwm_l_cv], axis=0)

    # Step 5: Shuffle train and test sets
    def shuffle_arrays(*arrays):
          assert all(len(arr) == len(arrays[0]) for arr in arrays), "All arrays must be the same length"
          p = np.random.permutation(len(arrays[0]))
          return [arr[p] for arr in arrays]

    g_train, r_train, s_train, l_train = shuffle_arrays(g_train, r_train, s_train, l_train)
    g_cv, r_cv, s_cv, l_cv = shuffle_arrays(g_cv, r_cv, s_cv, l_cv)


    train_g_values = torch.from_numpy(g_train).to(torch_device)
    train_r_values = torch.from_numpy(r_train).to(torch_device)
    train_masks = torch.from_numpy(s_train).to(torch_device)
    train_labels = torch.from_numpy(l_train).to(torch_device)
    cv_g_values = torch.from_numpy(g_cv).to(torch_device)
    cv_r_values = torch.from_numpy(r_cv).to(torch_device)
    cv_masks = torch.from_numpy(s_cv).to(torch_device)
    cv_labels = torch.from_numpy(l_cv).to(torch_device)

    return train_g_values, train_r_values, train_masks, train_labels, cv_g_values, cv_r_values, cv_masks, cv_labels


def process_data_for_testing(data, truncation_length, torch_device):

    # Step 1: filter and truncate
    filtered_data = []
    for i in range(len(data)):
        if len(data[i]['output_ids'][0]) >= truncation_length:
            data[i]['g_values'] = data[i]['g_values'][:truncation_length]
            data[i]['mc_g_values'] = data[i]['mc_g_values'][:truncation_length]
            data[i]['r_values'] = data[i]['r_values'][:truncation_length]
            data[i]['context_repetition_mask'] = data[i]['context_repetition_mask'][:truncation_length]
            filtered_data.append(data[i])
        else:
            pad_size = truncation_length - len(data[i]['output_ids'][0])
            data[i]['g_values'] = pad_g_values(data[i]['g_values'], truncation_length)
            data[i]['mc_g_values'] = pad_g_values(data[i]['mc_g_values'], truncation_length)
            data[i]['r_values'] = np.concatenate([data[i]['r_values'], np.zeros(pad_size)])
            data[i]['context_repetition_mask'] = np.concatenate([data[i]['context_repetition_mask'], np.ones(pad_size)])
            filtered_data.append(data[i])
    print(f'Filtered testing data size: {len(filtered_data)}')

    wm_g_values = np.array([np.stack([filtered_data[i]['g_values'], filtered_data[i]['mc_g_values']], axis=1) for i in range(len(filtered_data))])
    wm_r_values = np.array([filtered_data[i]['r_values'] for i in range(len(filtered_data))])
    wm_skipped = np.array([1-filtered_data[i]['context_repetition_mask'] for i in range(len(filtered_data))])
    wm_labels = np.ones(len(filtered_data), dtype=np.float32)
    # Use simulation data for unwatermarked data
    uwm_g_values = np.random.randint(0, 2, size=wm_g_values.shape)
    uwm_r_values = np.random.rand(*wm_r_values.shape)
    uwm_skipped = np.ones(wm_skipped.shape)
    uwm_labels = np.zeros(len(filtered_data), dtype=np.float32)

    # Step 2: Concatenate wm and uwm data
    g = np.concatenate([wm_g_values, uwm_g_values], axis=0)
    r = np.concatenate([wm_r_values, uwm_r_values], axis=0)
    s = np.concatenate([wm_skipped, uwm_skipped], axis=0)
    l = np.concatenate([wm_labels, uwm_labels], axis=0)

    # Step 3: Shuffle train and test sets
    def shuffle_arrays(*arrays):
          assert all(len(arr) == len(arrays[0]) for arr in arrays), "All arrays must be the same length"
          p = np.random.permutation(len(arrays[0]))
          return [arr[p] for arr in arrays]

    g, r, s, l = shuffle_arrays(g, r, s, l)

    test_g_values = torch.from_numpy(g).to(torch_device)
    test_r_values = torch.from_numpy(r).to(torch_device)
    test_masks = torch.from_numpy(s).to(torch_device)
    test_labels = torch.from_numpy(l).to(torch_device)
    return test_g_values, test_r_values, test_masks, test_labels


def tpr_at_fpr(pred_scores: torch.Tensor, true_labels: torch.Tensor, target_fpr: float = 0.01) -> float:
    """
    Compute TPR at a specified FPR threshold (default 1%) for binary classification.

    Args:
        pred_scores (torch.Tensor): Predicted probabilities or scores (shape [N]).
        true_labels (torch.Tensor): True binary labels (0 or 1) (shape [N]).
        target_fpr (float): Target false positive rate (default 0.01 for 1%).

    Returns:
        float: TPR at the closest FPR to target_fpr.
    """
    # Convert to NumPy
    w_pred = pred_scores.detach().cpu().numpy()
    w_true = true_labels.detach().cpu().numpy()

    positive_idxs = w_true == 1
    negative_idxs = w_true == 0

    positive_scores = w_pred[positive_idxs]
    negative_scores = w_pred[negative_idxs]
    fpr_threshold = np.percentile(negative_scores, 100 - target_fpr * 100)

    return np.mean(positive_scores >= fpr_threshold)


def batched_predict(model, g_values, r_values, masks, device, batch_size=128):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(g_values), batch_size):
            g_batch = g_values[i:i+batch_size].to(device)
            r_batch = r_values[i:i+batch_size].to(device)
            mask_batch = masks[i:i+batch_size].to(device)

            preds = model(g_batch, r_batch, mask_batch, train_mode=False)
            all_preds.append(preds.cpu())  # move back to CPU to save memory

    return torch.cat(all_preds, dim=0)


class AcceptNet(nn.Module):
    def __init__(self, input_dim=60, hidden1=30, hidden2=10, alpha=3):
        super().__init__()
        self.alpha = alpha
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, 1)

    def forward(self, gD, gT, r, train_mode=True):
        # gD, gT: shape (..., 30)
        # r: shape (..., 1)
        x = torch.cat([gD, gT], dim=-1)  # shape (..., 60)
        # small MLP to get threshold T
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        T = self.out(x)  # shape (..., 1)
        if train_mode:
            # smooth version for training
            prob = torch.sigmoid(self.alpha * (T - r))
        else:
            # hard threshold for inference
            prob = (r <= T).float()

        return prob


class LikelihoodModel(nn.Module, abc.ABC):
    """Base class for likelihood models."""

    @abc.abstractmethod
    def forward(self, g_values: torch.Tensor) -> torch.Tensor:
        """Compute likelihoods given g-values."""
        pass


class LikelihoodModelWatermarked(LikelihoodModel):
    """Model for P(g_values|watermarked)."""

    def __init__(self, watermarking_depth: int):
        super().__init__()
        self.watermarking_depth = watermarking_depth

        # Initialize parameters
        self.beta = nn.Parameter(
            -2.5 + 0.001 * torch.randn(1, 1, watermarking_depth)
        )
        self.delta = nn.Parameter(
            0.001 * torch.randn(1, 1, watermarking_depth, watermarking_depth)
        )

    def l2_loss(self) -> torch.Tensor:
        return torch.sum(self.delta ** 2)

    def _compute_latents(
        self, g_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute latent probabilities."""
        x = g_values.unsqueeze(-2).repeat(
            1, 1, self.watermarking_depth, 1
        )
        x = torch.tril(x, diagonal=-1)

        logits = torch.sum(self.delta * x, dim=-1) + self.beta

        p_two_unique_tokens = torch.sigmoid(logits)
        p_one_unique_token = 1 - p_two_unique_tokens

        return p_one_unique_token, p_two_unique_tokens

    def forward(self, g_values: torch.Tensor) -> torch.Tensor:
        """Compute P(g_values|watermarked)."""
        p_one_unique_token, p_two_unique_tokens = self._compute_latents(g_values)
        return 0.25 * ((g_values + 0.5) * p_two_unique_tokens + p_one_unique_token)



class LikelihoodModelUnwatermarked(LikelihoodModel):
    """Model for P(g_values|unwatermarked)."""

    def forward(self, g_values: torch.Tensor) -> torch.Tensor:
        """Compute P(g_values|unwatermarked)."""
        return 0.25 * torch.ones_like(g_values)


class LikelihoodModelAccept(LikelihoodModel):
    """Model for P(watermark key|r)."""

    def __init__(self):
        # Initialize parameters
        super().__init__()
        self.beta = nn.Parameter(
            0.5 + 0.01 * torch.randn(1)
        )
        self.delta = nn.Parameter(
            10 + torch.randn(1)
        )

    def forward(self, r_values: torch.Tensor) -> torch.Tensor:
        """Compute P(watermark key|r)."""

        return torch.sigmoid(self.delta*(self.beta - r_values))

class BayesianDetectorModule(nn.Module):
    """Bayesian detector model."""

    def __init__(
        self,
        watermarking_depth: int,
        baserate: float = 0.5,
        thresholdnet: bool = False,
        prior_accept: bool = False,
        prior_accept_rate: float = 0.5,
    ):
        super().__init__()
        self.watermarking_depth = watermarking_depth
        self.baserate = baserate
        self.prior_accept = prior_accept
        self.prior_accept_rate = prior_accept_rate
        self.thresholdnet = thresholdnet

        self.likelihood_model_watermarked_D = LikelihoodModelWatermarked(
            watermarking_depth=watermarking_depth
        )
        self.likelihood_model_watermarked_T = LikelihoodModelWatermarked(
            watermarking_depth=watermarking_depth
        )
        self.likelihood_model_unwatermarked = LikelihoodModelUnwatermarked()
        if not prior_accept and thresholdnet:
              # use a network to estimate threshold
            self.likelihood_model_accept = AcceptNet()
        else: # use a simple sigmoid
            self.likelihood_model_accept = LikelihoodModelAccept()

        self.prior = nn.Parameter(torch.tensor([baserate]))

    def l2_loss(self) -> torch.Tensor:
        return self.likelihood_model_watermarked_D.l2_loss() + self.likelihood_model_watermarked_T.l2_loss()

    def forward(
        self,
        g_values: torch.Tensor, 
        r_values: torch.Tensor,
        mask: torch.Tensor,
        train_mode: bool = True,
    ) -> torch.Tensor:
        """
        Compute P(watermarked|g_values).
        g_values: shape (batch_size, sequence_len, 2, depth)
        r_values: shape (batch_size, sequence_len)
        mask: shape (batch_size, sequence_len)
        train_mode: bool
        """
        likelihoods_watermarked_D = self.likelihood_model_watermarked_D(g_values[:,:,0]) # shape: (batch_size, sequence_len, depth)
        likelihoods_watermarked_T = self.likelihood_model_watermarked_T(g_values[:,:,1])
        if not self.prior_accept and not self.thresholdnet:
            likelihoods_accept = self.likelihood_model_accept(r_values) # the likelihood to use watermark key D, shape: (batch_size, sequence_len)
        elif not self.prior_accept and self.thresholdnet:
            batch_size, n = g_values.shape[:2]
            gD_flat = g_values[:,:,0].view(batch_size * n, 30)
            gT_flat = g_values[:,:,1].view(batch_size * n, 30)
            r_flat = r_values.view(batch_size * n, 1)
            likelihoods_accept_flat = self.likelihood_model_accept(gD_flat.float(), gT_flat.float(), r_flat, train_mode=train_mode)
            likelihoods_accept = likelihoods_accept_flat.view(batch_size, n) # the likelihood to use watermark key D, shape: (batch_size, sequence_len)
        else:
            likelihoods_accept = torch.full(
                (r_values.shape[0], r_values.shape[1]),
                fill_value=self.prior_accept_rate,
                dtype=r_values.dtype,
                device=r_values.device
            )   # shape: (batch_size, sequence_len)
        likelihoods_unwatermarked = self.likelihood_model_unwatermarked(g_values[:,:,0])  # shape: (batch_size, sequence_len, depth)
        likelihoods_watermarked = likelihoods_watermarked_D * likelihoods_accept.unsqueeze(-1) + likelihoods_watermarked_T * (1 - likelihoods_accept.unsqueeze(-1))  # shape: (batch_size, sequence_len, depth)

        mask = mask.unsqueeze(-1)
        prior = torch.clamp(self.prior, 1e-5, 1 - 1e-5)

        log_likelihoods_watermarked = torch.log(
            torch.clamp(likelihoods_watermarked, min=1e-30)
        )
        log_likelihoods_unwatermarked = torch.log(
            torch.clamp(likelihoods_unwatermarked, min=1e-30)
        )

        log_odds = log_likelihoods_watermarked - log_likelihoods_unwatermarked

        relative_surprisal_likelihood = torch.sum(
            log_odds * mask, dim=(1, 2)
        ) # sum over sequence length and layers

        relative_surprisal_prior = torch.log(prior) - torch.log(1 - prior)
        relative_surprisal = relative_surprisal_prior + relative_surprisal_likelihood

        return torch.sigmoid(relative_surprisal)


def train_model(
    detector_module: BayesianDetectorModule,
    g_values: torch.Tensor,
    r_values: torch.Tensor,
    mask: torch.Tensor,
    watermarked: torch.Tensor, # labels
    epochs: int = 250,
    learning_rate: float = 1e-3,
    minibatch_size: int = 64,
    l2_weight: float = 0.0,
    g_values_val: torch.Tensor = None,
    r_values_val: torch.Tensor = None,
    mask_val: torch.Tensor = None,
    watermarked_val: torch.Tensor = None,
    verbose: bool = False,
    validation: str = 'tpr',
) -> tuple[dict, float]:
    """Train the detector model."""

    optimizer = torch.optim.Adam(detector_module.parameters(), lr=learning_rate)

    history = {}
    highest_tpr = 0
    best_state = None

    for epoch in range(epochs):
        detector_module.train()

        # Training
        train_losses = []
        for i in range(0, len(g_values), minibatch_size):
            batch_g = g_values[i:i+minibatch_size]
            batch_r = r_values[i:i+minibatch_size]
            batch_m = mask[i:i+minibatch_size]
            batch_w = watermarked[i:i+minibatch_size]

            optimizer.zero_grad()

            pred = detector_module(batch_g, batch_r, batch_m).to(torch.float32)
            loss = F.binary_cross_entropy(
                pred, batch_w.float()
            ) + l2_weight * detector_module.l2_loss()

            train_losses.append(loss.item())

            loss.backward()
            optimizer.step()

        avg_train_loss = sum(train_losses) / len(train_losses)

        # Validation
        detector_module.eval()
        with torch.no_grad():
            if g_values_val is not None:
                val_pred = detector_module(g_values_val, r_values_val, mask_val, train_mode=False).to(torch.float32)
                if validation == 'tpr':
                    val_loss = tpr_at_fpr(val_pred, watermarked_val)
                    val_entro_loss = F.binary_cross_entropy(
                    val_pred, watermarked_val.float()
                )
                elif validation == 'entropy':
                    val_loss = F.binary_cross_entropy(
                    val_pred, watermarked_val.float()
                )

                if val_loss > highest_tpr:
                    highest_tpr = val_loss
                    best_state = detector_module.state_dict()

                if verbose:
                    print(f"Epoch {epoch}: train_loss {avg_train_loss:.6f}, val_loss {val_loss:.6f}, val_entro_loss {val_entro_loss}")

            history[epoch] = {
                'train_loss': avg_train_loss,
                'val_loss': val_loss if g_values_val is not None else None
            }

    if best_state is not None:
        detector_module.load_state_dict(best_state)
    return history, highest_tpr