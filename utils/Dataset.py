# @markdown ### **Dataset**
# @markdown
# @markdown Defines `PushTStateDataset` and helper functions
# @markdown
# @markdown The dataset class
# @markdown - Load data (obs, action) from a zarr storage
# @markdown - Normalizes each dimension of obs and action to [-1,1]
# @markdown - Returns
# @markdown  - All possible segments with length `pred_horizon`
# @markdown  - Pads the beginning and the end of each episode with repetition
# @markdown  - key `obs`: shape (obs_horizon, obs_dim)
# @markdown  - key `action`: shape (pred_horizon, action_dim)

from utils.imports import *
from utils.dataset_generator import generate_dataset

# parameters
pred_horizon = 16
obs_horizon = 2
action_horizon = 8


# |o|o|                             observations: 2
# | |a|a|a|a|a|a|a|a|               actions executed: 8
# |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16


def create_sample_indices(
        episode_ends: np.ndarray, sequence_length: int,
        pad_before: int = 0, pad_after: int = 0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


# normalize data
def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats


def normalize_data(data, stats):
    # normalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data


# dataset
class PushTStateDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path,
                 pred_horizon, obs_horizon, action_horizon, refresh=None):

        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

        # read from zarr dataset
        dataset_root = zarr.open(dataset_path, 'r')
        # All demonstration episodes are concatinated in the first dimension N
        train_data = {
            # (N, action_dim)
            'action': dataset_root['data']['action'][:],
            # (N, obs_dim)
            'obs': dataset_root['data']['state'][:]
        }
        self.train_data = train_data

        # Marks one-past the last index for each episode
        episode_ends = dataset_root['meta']['episode_ends'][:]

        self.episode_ends = episode_ends

        if refresh is None:
            self.__refresh__()

    def __refresh__(self):
        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=self.episode_ends,
            sequence_length=self.pred_horizon,
            # add padding such that each timestep in the dataset are seen
            pad_before=self.obs_horizon - 1,
            pad_after=self.action_horizon - 1)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in self.train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data

    def __len__(self):
        # all possible segments of the dataset
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        nsample['obs'] = nsample['obs'][:self.obs_horizon, :]
        return nsample


####################################################################################

def __load_dataset_push_t__():
    # download demonstration data from Google Drive
    dataset_path = "datasets/pusht_cchi_v7_replay.zarr.zip"
    os.makedirs('datasets', exist_ok=True)

    if not os.path.isfile(dataset_path):
        id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
        gdown.download(id=id, output=dataset_path, quiet=False)

    # create dataset from file
    dataset = PushTStateDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon
    )
    return dataset


def load_dataset(system_name: str, dtype, device):
    print(f'[Dataset] Loading Dataset {system_name} from memory')
    if system_name == '2d':
        fn_dataset = __load_dataset_lqr2d__

    elif system_name == '3d':
        fn_dataset = __load_dataset_lqr3d__

    elif system_name == 'drone':
        fn_dataset = __load_dataset_drone__

    else:
        raise f'Dataset {system_name} not known'

    dataset, obs_dim, action_dim, name, fn_distance, fn_speed = fn_dataset(dtype, device)

    return dataset, obs_dim, action_dim, name, fn_distance, fn_speed


""" Our dataset are build on top or PushTDataset class"""


def __load__(system_name, dtype, device):
    dataset = __load_dataset_push_t__()
    if not os.path.isfile(f'datasets/dataset_{system_name}.pkl'):
        generate_dataset(system_name=system_name)

    f = open(f'datasets/dataset_{system_name}.pkl', 'rb')
    synthetic_ours = pickle.load(f)
    f.close()

    finish = synthetic_ours['finish_token'].astype(np.int8)
    episode_ends = np.array([idx for idx, a in enumerate(finish) if a == 1])
    del synthetic_ours['finish_token']
    dataset.train_data = synthetic_ours
    dataset.episode_ends = episode_ends
    dataset.__refresh__()

    return dataset


def __load_dataset_lqr2d__(dtype, device):
    dataset = __load__(system_name='2d',
                       dtype=dtype,
                       device=device)

    # observation and action dimensions
    obs_dim, action_dim = 4, 2

    # this depends on how carlo have generated the data
    fn_distance = lambda _obs: np.linalg.norm([_obs[0], _obs[2]])  # x, xdot, y, ydot
    fn_speed = lambda _obs: np.linalg.norm([_obs[1], _obs[3]])

    print(f'[2d][@carlo change] Obs: x, x_dot, y, y_dot')
    print('[@carlo change] Action: x_acc, y_ acc (?)')
    print('Observation Dim: ', obs_dim, 'Action Dim: ', action_dim)
    return dataset, obs_dim, action_dim, '2d', fn_distance, fn_speed


def __load_dataset_lqr3d__(dtype, device):
    dataset = __load__(system_name='3d',
                       dtype=dtype,
                       device=device)

    # observation and action dimensions
    obs_dim = 6
    action_dim = 3

    fn_distance = lambda _obs: np.linalg.norm([_obs[0], _obs[2], _obs[4]])  # x, xdot, y, ydot, z, zdot
    fn_speed = lambda _obs: np.linalg.norm([_obs[1], _obs[3], _obs[5]])  # x, xdot, y, ydot, z, zdot

    print(f'[3d][@carlo change] Obs: x, x_dot, y, y_dot')
    print('[@carlo change] Action: x_acc, y_ acc (?)')
    print('Observation Dim: ', obs_dim, 'Action Dim: ', action_dim)
    return dataset, obs_dim, action_dim, '3d', fn_distance, fn_speed


def __load_dataset_drone__(dtype, device):
    dataset = __load__(system_name='drone',
                       dtype=dtype,
                       device=device)

    # observation and action dimensions
    obs_dim = 12
    action_dim = 4

    fn_distance = lambda _obs: np.linalg.norm([_obs[0], _obs[1], _obs[2]])  # x, y, z, xdot, ydot, zdot
    fn_speed = lambda _obs: np.linalg.norm([_obs[3], _obs[4], _obs[5]])  # x, y, z, xdot, ydot, zdot

    print(f'[drone][@carlo change] Obs: x, x_dot, y, y_dot')
    print('[@carlo change] Action: x_acc, y_ acc (?)')
    print('Observation Dim: ', obs_dim, 'Action Dim: ', action_dim)

    return dataset, obs_dim, action_dim, 'drone', fn_distance, fn_speed


def load_dataset_lqr2d_observation():
    dataset = __load_dataset_push_t__()
    f = open('datasets/dataset_lqr.pkl', 'rb')
    synthetic_ours = pickle.load(f)
    f.close()
    synthetic_ours['obs'] = synthetic_ours['observations']

    # x, y only, no speed, translated by one, action is next observation
    synthetic_ours['action'] = synthetic_ours['observations'][1:, (0, 2)]
    del synthetic_ours['actions'], synthetic_ours['observations']
    finish = synthetic_ours['finish_token'].astype(np.int8)
    episode_ends = np.array([idx for idx, a in enumerate(finish) if a == 1])
    del synthetic_ours['finish_token']
    dataset.train_data = synthetic_ours
    dataset.episode_ends = episode_ends
    dataset.__refresh__()
    # observation and action dimensions

    obs_dim = 4
    action_dim = 2
    name = 'LQR2D_obs'

    fn_distance = lambda _obs: np.linalg.norm([_obs[0], _obs[2]])  # x, xdot, y, ydot
    fn_speed = lambda _obs: np.linalg.norm([_obs[1], _obs[3]])

    print(f'[{name}][@carlo change] Obs: x, x_dot, y, y_dot')
    print('[@carlo change] Action: x_acc, y_ acc (?)')
    print('Observation Dim: ', obs_dim, 'Action Dim: ', action_dim)
    return dataset, obs_dim, action_dim, name, fn_distance, fn_speed


def show_statistics(dataset_ours):
    dataset_paper = __load_dataset_push_t__()

    # Stat of trajectories
    ll1 = []
    for i in range(len(dataset_paper.episode_ends) - 1):
        ll1.append(
            len(dataset_paper.train_data['obs'][dataset_paper.episode_ends[i]:dataset_paper.episode_ends[i + 1]]))

    ll2 = []
    for i in range(len(dataset_ours.episode_ends) - 1):
        ll2.append(len(dataset_ours.train_data['obs'][dataset_ours.episode_ends[i]:dataset_ours.episode_ends[i + 1]]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(ll1)
    ax2.hist(ll2)
    ax1.set_xlabel('Trajectory Length')
    ax2.set_xlabel('Trajectory Length')
    ax1.set_title('PushT Dataset')
    ax2.set_title('Our Synthetic Dataset')
