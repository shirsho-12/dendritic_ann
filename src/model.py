import torch
from torch import nn
import torch.nn.functional as F
from receptive_fields import receptive_fields, connectivity, random_connectivity


def make_masks(
    dends,
    soma,
    synapses,
    num_layers,
    img_width,
    img_height,
    num_classes=10,
    channels=1,
    conventional=False,
    sparse=False,
    rfs=True,
    rfs_type="somatic",
    rfs_mode="random",
    input_sample=None,
    seed=None,
):
    """
    Returns a list of masks (weights + biases) for dendritic and somatic layers.
    Masks returned follow the same convention as receptive_fields():
      mask_s_d shape == (input_dim, soma*dends)  # i.e. (in_features, out_features)
    We'll transpose before using in MaskedLinear (which expects [out, in]).
    """
    masks = []
    for i in range(num_layers):
        if i == 0:
            matrix = torch.zeros((img_width, img_height))
        else:
            divisors = [j for j in range(1, soma[i - 1] + 1) if soma[i - 1] % j == 0]
            ix = len(divisors) // 2
            if len(divisors) % 2 == 0:
                matrix = torch.zeros((divisors[ix], divisors[ix - 1]))
            else:
                matrix = torch.zeros((divisors[ix], divisors[ix]))

        # RF-based structured or random connectivity
        if rfs:
            mask_s_d, centers = receptive_fields(
                matrix,
                somata=soma[i],
                dendrites=dends[i],
                num_of_synapses=synapses,
                opt=rfs_mode,
                rfs_type=rfs_type,
                prob=0.7,
                num_channels=channels if i == 0 else 1,
                seed=seed,
            )
            # mask_s_d is a torch tensor already, shape (matrix.size * channels, soma*dends)
        else:
            factor = channels if i == 0 else 1
            inputs_size = (
                matrix.numel() * factor
            )  # <--- fix: compute product, not a generator
            mask_s_d = random_connectivity(
                inputs=int(inputs_size),
                outputs=soma[i] * dends[i],
                conns=synapses * soma[i] * dends[i],
                seed=seed,
            )

        # keep mask_s_d as-is (shape = in x out). we'll transpose where needed.
        masks.append(mask_s_d.to(torch.int32))
        masks.append(
            torch.ones((mask_s_d.shape[1],), dtype=torch.int32)
        )  # bias mask length = out_features

        # Dendrite → soma mask (this has shape (dends*soma, soma) i.e. inputs x outputs)
        if not sparse:
            mask_d_s = connectivity(inputs=dends[i] * soma[i], outputs=soma[i])
        else:
            mask_d_s = random_connectivity(
                inputs=dends[i] * soma[i],
                outputs=soma[i],
                conns=dends[i] * soma[i],
                seed=seed,
            )

        masks.append(mask_d_s.to(torch.int32))
        masks.append(torch.ones((mask_d_s.shape[1],), dtype=torch.int32))

    # Output layer masks (weights: prev_out_features x num_classes in the same in x out convention)
    # masks[-2] is mask_d_s (the last connectivity), its shape = (in_features_last, out_features_last)
    prev_out = masks[-2].shape[1]
    masks.append(torch.ones((prev_out, num_classes), dtype=torch.int32))
    masks.append(torch.ones((num_classes,), dtype=torch.int32))
    return masks


class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("mask", mask)  # [out, in]
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        masked_weight = self.weight * self.mask  # type: ignore
        return F.linear(x, masked_weight, self.bias)


class DANN(nn.Module):
    def __init__(
        self,
        input_dim,
        dends,
        soma,
        num_classes,
        relu_slope=0.1,
        dropout=False,
        rate=0.5,
        seed=None,
        conventional=False,
        sparse=False,
        rfs=True,
        rfs_type="somatic",
        rfs_mode="random",
        input_sample=None,
    ):
        super(DANN, self).__init__()

        self.num_layers = len(dends)
        self.relu_slope = relu_slope
        self.dropout = dropout
        self.rate = rate
        self.layers = nn.ModuleDict()
        masks = make_masks(
            dends=dends,
            soma=soma,
            synapses=num_classes,
            num_layers=self.num_layers,
            img_width=input_dim[0],
            img_height=input_dim[1],
            num_classes=num_classes,
            channels=1,
            conventional=conventional,
            sparse=sparse,
            rfs=rfs,
            rfs_type=rfs_type,
            rfs_mode=rfs_mode,
            input_sample=input_sample,
            seed=seed,
        )

        # Each layer uses 4 masks: dend_w, dend_b, soma_w, soma_b
        mask_idx = 0

        # Layer 1
        self.layers["dend_1"] = MaskedLinear(
            input_dim[0] * input_dim[1],
            dends[0] * soma[0],
            mask=masks[mask_idx].T.to(torch.float32),  # <<< transpose here
        )
        mask_idx += 2  # skip bias mask

        self.layers["soma_1"] = MaskedLinear(
            dends[0] * soma[0],
            soma[0],
            mask=masks[mask_idx].T.to(torch.float32),  # <<< transpose here
        )
        mask_idx += 2

        # Remaining dendro-somatic layers
        for j in range(1, self.num_layers):
            self.layers[f"dend_{j+1}"] = MaskedLinear(
                soma[j - 1],
                dends[j] * soma[j],
                mask=masks[mask_idx].T.to(torch.float32),  # <<< transpose here
            )
            mask_idx += 2
            self.layers[f"soma_{j+1}"] = MaskedLinear(
                dends[j] * soma[j],
                soma[j],
                mask=masks[mask_idx].T.to(torch.float32),  # <<< transpose here
            )
            mask_idx += 2

        # Output layer (masks[-2] is in x out, so transpose to out x in)
        self.output = MaskedLinear(
            soma[-1], num_classes, mask=masks[-2].T.to(torch.float32)
        )

    def forward(self, x):
        for j in range(1, self.num_layers + 1):
            dend = self.layers[f"dend_{j}"](x)
            dend = F.leaky_relu(dend, negative_slope=self.relu_slope)
            if self.dropout:
                dend = F.dropout(dend, p=self.rate, training=self.training)

            soma = self.layers[f"soma_{j}"](dend)
            soma = F.leaky_relu(soma, negative_slope=self.relu_slope)
            if self.dropout:
                soma = F.dropout(soma, p=self.rate, training=self.training)

            x = soma  # pass to next layer

        out = self.output(x)
        return F.softmax(out, dim=-1)


if __name__ == "__main__":
    model = DANN(
        input_dim=(28, 28),
        dends=[128, 64],
        soma=[64, 32],
        num_classes=10,
        seed=42,
    )
    print(model)
    sample_input = torch.randn((1, 28 * 28))
    output = model(sample_input)
    print(output)
