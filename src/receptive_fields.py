import torch


# ==========================================================
# Helper: Chessboard distance transform (torch version)
# ==========================================================
def chessboard_distance(tensor):
    """
    Compute the chessboard distance transform of a binary tensor.
    0s in tensor become centers (distance 0).
    """
    device = tensor.device
    inf = torch.full_like(tensor, float("inf"), dtype=torch.float32)
    dist = torch.where(tensor == 0, torch.zeros_like(tensor, dtype=torch.float32), inf)

    # Propagate minimum distance from neighbors (like scipy.ndimage.distance_transform_cdt)
    for _ in range(max(tensor.shape)):  # guaranteed to converge
        dist_padded = torch.nn.functional.pad(dist, (1, 1, 1, 1), value=float("inf"))
        neighbor_min = torch.stack(
            [
                dist_padded[0:-2, 0:-2],
                dist_padded[0:-2, 1:-1],
                dist_padded[0:-2, 2:],
                dist_padded[1:-1, 0:-2],
                dist_padded[1:-1, 2:],
                dist_padded[2:, 0:-2],
                dist_padded[2:, 1:-1],
                dist_padded[2:, 2:],
            ],
            dim=0,
        ).min(0)[0]
        new_dist = torch.minimum(dist, neighbor_min + 1)
        if torch.equal(new_dist, dist):  # converged
            break
        dist = new_dist
    return dist


# ==========================================================
# Neighborhood extraction
# ==========================================================
def nb_vals(matrix, indices, size=1, opt=None):
    """
    Get indices of neighbors around a center pixel using chessboard metric.
    """
    dist = torch.ones_like(matrix)
    dist[indices[0], indices[1]] = 0
    dist = chessboard_distance(dist)

    if opt == "extended":
        nb_indices = torch.nonzero(dist == size, as_tuple=False)
    else:
        nb_indices = torch.nonzero(dist <= size, as_tuple=False)
    return nb_indices


# ==========================================================
# Random connectivity matrix
# ==========================================================
def random_connectivity(
    inputs, outputs, opt="random", conns=None, seed=None, device="cpu"
):
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)

    mask = torch.zeros((inputs, outputs), dtype=torch.int32, device=device)

    if opt == "one_to_one":
        idxs = torch.randint(0, inputs, (outputs,), generator=g)
        mask[idxs, torch.arange(outputs, device=device)] = 1

    elif opt == "random":
        if conns is None or conns <= 0 or not isinstance(conns, int):
            raise ValueError("Specify `conns` as positive integer.")
        if conns > mask.numel():
            raise ValueError("`conns` exceeds total connections.")
        flat_idxs = torch.randperm(mask.numel(), generator=g, device=device)[:conns]
        mask.view(-1)[flat_idxs] = 1

    elif opt == "constant":
        if conns is None or conns <= 0 or not isinstance(conns, int):
            raise ValueError("Specify `conns` as positive integer.")
        if conns > inputs:
            raise ValueError("`conns` cannot exceed number of input nodes.")
        for j in range(outputs):
            idx = torch.randperm(inputs, generator=g, device=device)[:conns]
            mask[idx, j] = 1

    else:
        raise ValueError("`opt` must be one of ['one_to_one', 'random', 'constant']")
    return mask


# ==========================================================
# Center selection and synapse allocation
# ==========================================================
def choose_centers(possible_values, nodes, seed=None, device="cpu"):
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    possible_values = torch.tensor(possible_values, device=device)
    idx = torch.randint(0, len(possible_values), (nodes,), generator=g)
    return possible_values[idx]


def allocate_synapses(nb, matrix, num_of_synapses, num_channels=1, seed=None):
    g = torch.Generator(device=matrix.device)
    if seed is not None:
        g.manual_seed(seed)

    M, N = matrix.shape
    mask = torch.zeros((M, N), dtype=torch.int32, device=matrix.device)

    syn_indices = nb_vals(matrix, nb)
    if syn_indices.shape[0] < num_of_synapses:
        diff = num_of_synapses - syn_indices.shape[0]
        extra = nb_vals(matrix, nb, size=2, opt="extended")

        cnt = 3
        while diff > extra.shape[0]:
            extra_more = nb_vals(matrix, nb, size=cnt, opt="extended")
            extra = torch.cat((extra, extra_more), dim=0)
            cnt += 1
            if diff <= extra.shape[0]:
                break
        added = extra[torch.randperm(extra.shape[0], generator=g)[:diff]]
        syn_indices = torch.cat((syn_indices, added), dim=0)
    elif syn_indices.shape[0] > num_of_synapses:
        idx = torch.randperm(syn_indices.shape[0], generator=g)[:num_of_synapses]
        syn_indices = syn_indices[idx]

    if syn_indices.shape[0] != num_of_synapses:
        raise ValueError("Synapse allocation mismatch!")

    mask[syn_indices[:, 0], syn_indices[:, 1]] = 1
    if num_channels > 1:
        mask = mask.unsqueeze(2).repeat(1, 1, num_channels)

    return mask.reshape(-1)


# ==========================================================
# Make receptive field mask
# ==========================================================
def make_mask_matrix(
    centers_ids,
    matrix,
    dendrites,
    somata,
    num_of_synapses,
    num_channels=1,
    rfs_type="somatic",
    seed=None,
):
    g = torch.Generator(device=matrix.device)
    if seed is not None:
        g.manual_seed(seed)

    M, N = matrix.shape
    mask_final = torch.zeros(
        (dendrites * somata, M * N * num_channels),
        dtype=torch.int32,
        device=matrix.device,
    )
    counter = 0

    if rfs_type == "somatic":
        for center in centers_ids:
            nb_indices = nb_vals(matrix, center)
            if dendrites < nb_indices.shape[0]:
                idx = torch.randperm(nb_indices.shape[0], generator=g)[:dendrites]
                nb_indices = nb_indices[idx]
            elif dendrites > nb_indices.shape[0]:
                diff = dendrites - nb_indices.shape[0]
                extra_centers = nb_vals(matrix, center, size=2, opt="extended")
                cnt = 3
                while diff > extra_centers.shape[0]:
                    extra_more = nb_vals(matrix, center, size=cnt, opt="extended")
                    extra_centers = torch.cat((extra_centers, extra_more), dim=0)
                    cnt += 1
                    if diff <= extra_centers.shape[0]:
                        break
                added = extra_centers[
                    torch.randperm(extra_centers.shape[0], generator=g)[:diff]
                ]
                nb_indices = torch.cat((nb_indices, added), dim=0)

            for nb in nb_indices:
                mask_final[counter, :] = allocate_synapses(
                    nb, matrix, num_of_synapses, num_channels=num_channels, seed=seed
                )
                counter += 1

    elif rfs_type == "dendritic":
        for center in centers_ids:
            mask_final[counter, :] = allocate_synapses(
                center, matrix, num_of_synapses, num_channels=num_channels, seed=seed
            )
            counter += 1

    return mask_final


# ==========================================================
# Top-level RF constructor
# ==========================================================
def receptive_fields(
    matrix,
    somata,
    dendrites,
    num_of_synapses,
    opt="random",
    rfs_type="somatic",
    step=None,
    prob=None,
    num_channels=1,
    size_rfs=None,
    centers_ids=None,
    seed=None,
):
    g = torch.Generator(device=matrix.device)
    if seed is not None:
        g.manual_seed(seed)

    M, N = matrix.shape
    if rfs_type == "somatic":
        nodes = somata
    elif rfs_type == "dendritic":
        nodes = dendrites * somata

    # Select centers
    if centers_ids is None:
        if opt == "random":
            centers_w = choose_centers(range(M), nodes, seed, device=matrix.device)
            centers_h = choose_centers(range(N), nodes, seed, device=matrix.device)
            centers_ids = [
                (int(x.item()), int(y.item())) for x, y in zip(centers_w, centers_h)
            ]

        elif opt == "serial":
            if step is None:
                raise ValueError("`step` required for serial mode")
            xv, yv = torch.meshgrid(torch.arange(M), torch.arange(N), indexing="ij")
            coords = torch.stack([xv.flatten(), yv.flatten()], dim=1)
            list_of_indices = torch.arange(0, coords.shape[0], step)
            centers_ids = [tuple(coords[i].tolist()) for i in list_of_indices]

        else:
            raise NotImplementedError(f"RF opt {opt} not yet supported in torch port.")

    mask_final = make_mask_matrix(
        centers_ids,
        matrix,
        dendrites,
        somata,
        num_of_synapses,
        num_channels,
        rfs_type,
        seed,
    )
    return mask_final.T.int(), centers_ids


# ==========================================================
# Structured connectivity (non-random)
# ==========================================================
def connectivity(inputs, outputs, device="cpu"):
    if outputs <= 0 or inputs <= 0:
        raise ValueError("Inputs/outputs must be positive.")
    if inputs % outputs != 0:
        raise ValueError("Inputs must divide evenly by outputs.")
    mat = torch.zeros((inputs, outputs), dtype=torch.int32, device=device)
    in_per_out = inputs // outputs
    for j in range(outputs):
        mat[in_per_out * j : in_per_out * (j + 1), j] = 1
    return mat
