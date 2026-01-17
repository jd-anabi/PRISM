import torch

from core.Simulator import dim_simulator, nd_simulator, nadrowski_simulator

def gen_obs(sim: str, params: list, t: torch.Tensor, inits: list, n_segs: int, steady_idx: int, dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cpu")):
    """
    Function in the SBI pipeline that generates observational data to test the posterior extracted by the neural network.
    :param sim: name of the simulator; currently only valid ones are "Dimensional", "Non-dimensional", and "Nadrowski"
    :param params: the parameters to use for the simulation
    :param t: the time to run the simulation for
    :param inits: the initial values to use for the state variables of the simulation
    :param n_segs: the number of segments to divide the time series into for memory efficient simulations
    :param steady_idx: the index that corresponds to the steady state
    :param dtype: the data type to use
    :param device: the device to run the simulator on
    :return: tensor of shape (number of state variables, t[steady_idx:].shape[0])
    :raise: ValueError if sim is not a valid one
    """
    valid_sims = ["Dimensional", "Non-dimensional", "Nadrowski"]
    if sim not in valid_sims:
        raise ValueError(f"Invalid simulator: {sim}")

    # move to device
    t = t.to(dtype=dtype, device=device)

    # construct parameters and initial conditions tensor
    params_tensor = torch.tensor(params, dtype=dtype, device=device).unsqueeze(0) # shape: (1, number of parameters)
    inits_tensor = torch.tensor(inits, dtype=dtype, device=device).unsqueeze(0) # shape: (1, number of state variables)

    # construct a tensor that corresponds to no force
    force = torch.zeros((1, t.shape[0]), dtype=dtype, device=device)

    simulator = None
    match sim:
        case "Dimensional":
            simulator = dim_simulator.DimSimulator(params_tensor, force, inits_tensor, t, segs=n_segs)
        case "Non-dimensional":
            simulator = nd_simulator.NDSimulator(params_tensor, force, inits_tensor, t, segs=n_segs)
        case "Nadrowski":
            simulator = nadrowski_simulator.NadrowskiSimulator(params_tensor, force, inits_tensor, t, segs=n_segs)
        case _:
            raise ValueError(f"Invalid simulator: {sim}")

    obs = simulator.simulate()[:, 0, :, steady_idx:]
    return obs