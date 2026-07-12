import torch
import torch.nn.functional as f
import numpy as np

import torch


class Dataset:
    def __init__(self, batch_size, v_size, S, rf, n_samples, pad):
        self.batch_size = batch_size
        self.v_size = v_size
        self.S = S
        self.resolution_factor = rf
        self.n_samples = n_samples

        # Tensori a bassa risoluzione

        self.v_mask = torch.zeros(batch_size, S, S)
        if pad > 0:
            # alto
            self.v_mask[:, :pad, :] = 1
            # basso
            self.v_mask[:, -pad:, :] = 1
            # sinistra
            self.v_mask[:, :, :pad] = 1
            # destra
            self.v_mask[:, :, -pad:] = 1
        # Tensori ad alta risoluzione per il campionamento

        self.v_mask_full_res = torch.zeros(batch_size, 1, S * rf, S * rf)
        # Stati nascosti
        self.hidden_states = torch.zeros(batch_size, v_size, S+1, S+1)

    def ask(self):
        grid_offsets = []
        sample_v_mask = []
        self.hidden_states = torch.zeros((self.batch_size, self.v_size, self.S+1, self.S+1))
        for _ in range(self.n_samples):
            offset = torch.rand(3)  # [x_off_norm, y_off_norm, t_off_norm]
            grid_offsets.append(offset)

            # calcolo degli offset discreti in [0 .. rf-1]
            x_off = min(int(self.resolution_factor * offset[0]), self.resolution_factor - 1)
            y_off = min(int(self.resolution_factor * offset[1]), self.resolution_factor - 1)

            # sottocampionamento strided su tutta la batch

            sample_v_mask.append(
                self.v_mask_full_res[:, :, x_off::self.resolution_factor, y_off::self.resolution_factor]
            )

        return self.v_mask

    def tell(self, hidden_state):
        
        # sovrascrivo tutti gli stati nascosti con il nuovo tensor
        self.hidden_states = hidden_state.detach()
