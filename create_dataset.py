import os
import math
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from subprocess import check_call
from scipy.spatial import distance_matrix
from torch.utils.data import DataLoader, Dataset
from k_means_constrained import KMeansConstrained

from utils.data_utils import set_seed, str2bool, save_dataset
from problems.op.op_baseline import read_oplib, write_oplib, calc_op_total


def generate_data(dataset_size, graph_size, prize_type, num_agents=1, num_depots=1, oracle=False, max_length=0, i=0):
    """Generate a problem instance."""

    # Coordinates
    d = {'depot': np.random.uniform(size=(dataset_size, 2)).squeeze(),
         'loc': np.random.uniform(size=(dataset_size, graph_size, 2)).squeeze()}

    # Reward values (some methods are taken from Fischetti et al. 1998)
    # Constant: All prizes with same value (value=1)
    if prize_type == 'const':
        d['prize'] = np.ones((dataset_size, graph_size)).squeeze()

    # Uniform: normalized random values
    elif prize_type == 'unif':
        d['prize'] = ((1 + np.random.randint(0, 100, size=(dataset_size, graph_size))) / 100.).squeeze()

    # Cooperative or non-cooperative multi-agent mission planning
    elif prize_type == 'coop' or prize_type == 'nocoop':
        clf = KMeansConstrained(n_clusters=num_agents, size_min=graph_size // num_agents,
                                size_max=graph_size // num_agents + 1, random_state=0, max_iter=1)
        if dataset_size == 1:
            labels = clf.fit_predict(d['loc'])
        else:
            labels = np.zeros((dataset_size, graph_size))
            for j in tqdm(range(dataset_size)):
                labels[j] = clf.fit_predict(d['loc'][i])
        d['prize'] = np.ones(labels.shape)
        d['prize'][labels != np.random.randint(low=0, high=num_agents, size=1)[0]] = 0.5 if prize_type == 'coop' else 0

    # Distance: based on distance to depot
    else:
        assert prize_type == 'dist'
        prize = np.linalg.norm(d['depot'][None, :] - d['loc'], axis=-1)
        d['prize'] = (1 + (prize / prize.max(axis=-1, keepdims=True) * 99).astype(int)) / 100.

    # Max length is approximately half of optimal TSP tour, such that half (a bit more) of the nodes can be visited
    # which is maximally difficult as this has the largest number of possibilities
    if max_length <= 0:
        MAX_LENGTHS = {
            20: 2.,
            50: 3.,
            100: 4.,
            200: 5.
        }
        if dataset_size == 1:
            d['max_length'] = MAX_LENGTHS[graph_size]
        else:
            d['max_length'] = np.full(dataset_size, MAX_LENGTHS[graph_size])
    else:
        if dataset_size == 1:
            d['max_length'] = np.random.uniform(max_length / 2, max_length)
        else:
            d['max_length'] = np.random.uniform(max_length / 2, max_length, size=dataset_size)

    # Output depot
    if num_depots > 1:
        d['depot2'] = np.random.uniform(size=(dataset_size, 2)).squeeze()

    # Oracle
    if oracle and dataset_size == 1:
        executable = os.path.abspath(os.path.join('problems', 'op', 'compass', 'compass'))
        problem_filename = os.path.join(path_graph, 'compass{}.oplib'.format(i))
        log_filename = os.path.join(path_graph, 'compass{}.log'.format(i))
        tour_filename = os.path.join(path_graph, 'compass{}.tour'.format(i))
        write_oplib(problem_filename, d['depot'], d['loc'], d['prize'], d['max_length'], name='compass')
        with open(log_filename, 'w') as f:
            check_call([executable, '--op', '--op-ea4op', problem_filename, '-o', tour_filename], stdout=f, stderr=f)
        tour = read_oplib(tour_filename, n=len(d['prize']))
        d['oracle'] = calc_op_total(d['prize'], tour)
        os.remove(problem_filename)
        os.remove(log_filename)
        os.remove(tour_filename)

    if dataset_size > 1:
        return list(zip(*[v.tolist() for _, v in d.items()]))
    return d


def estimate_max_length(n):
    """Estimate the maximum length allowed for the OP using the lower bound of the TSP:
       http://lcm.csa.iisc.ernet.in/dsa/node187.html"""
    loc = np.random.uniform(size=(500, n, 2))
    c = 0
    for b in range(loc.shape[0]):
        d = distance_matrix(loc[b], loc[b])
        min_edges = [np.sum(sorted(d[:, i])[:3]) for i in range(loc.shape[1])]
        c += np.sum(min_edges) / 2
    c = c / loc.shape[0]
    return c


class ScenarioGenerator(Dataset):

    def __init__(self, dataset_size, graph_size, data_distribution, num_agents, num_depots, oracle, max_length,
                 path_graph, padding):
        super(ScenarioGenerator, self).__init__()
        self.dataset_size = dataset_size
        self.graph_size = graph_size
        self.data_distribution = data_distribution
        self.num_agents = num_agents
        self.num_depots = num_depots
        self.oracle = oracle
        self.max_length = max_length
        self.path_graph = path_graph
        self.padding = padding

    def __getitem__(self, item):

        # Generate OP instance
        scenario = generate_data(1, self.graph_size, self.data_distribution, num_agents=self.num_agents,
                                 num_depots=self.num_depots, oracle=self.oracle, max_length=self.max_length, i=item)

        # Save instance in .pkl file
        with open(os.path.join(self.path_graph, str(item).zfill(self.padding) + '.pkl'), 'wb') as file:
            pickle.dump(scenario, file)
        return []

    def __len__(self):
        return self.dataset_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")

    # Options
    parser.add_argument("--data_dir", default='datasets', help="Create datasets in data_dir/problem")
    parser.add_argument("--compass_oracle", type=str2bool, default=False,
                        help="Use Compass as oracle to normalize cost function (recommended if len(graph_sizes) > 1)")
    parser.add_argument("--estimate_max_length", type=str2bool, default=False,
                        help="Estimate max_length instead of using default values")
    parser.add_argument('--num_workers', type=int, default=16, help='Num of parallel workers loading batches of data')

    # Problem
    parser.add_argument('--data_distribution', type=str, default='coop',
                        help="Distributions to generate for OP: const, dist, unif, coop or nocoop")
    parser.add_argument('--num_agents', type=int, default=4, help="Number of agents (for OP-coop or OP-nocoop)")
    parser.add_argument('--num_depots', type=int, default=1, help="Number of depots. Options are 1 or 2. num_depots=1"
                        "means that the start and end depot are the same. num_depots=2 means that they are different")

    # Sizes
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for the DataLoader")
    parser.add_argument("--train_sizes", type=int, nargs='+', default=[1280000],
                        help="Sizes of the train dataset (1 train_size per graph_size). Set 0 to avoid it")
    parser.add_argument("--test_sizes", type=int, nargs='+', default=[10000],
                        help="Sizes of the test dataset (1 test_size per graph_size). Set 0 to avoid it")
    parser.add_argument("--val_sizes", type=int, nargs='+', default=[10000],
                        help="Sizes of the validation dataset (1 val_size per graph_size). Set 0 to avoid it")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[20],
                        help="Sizes of the problem instances")

    # Set seed and check that everything is ok
    opts = parser.parse_args()
    set_seed(opts.seed)
    assert opts.num_depots in [1, 2], 'num_depots can only be 1 or 2'
    assert len(opts.train_sizes) == len(opts.graph_sizes) or opts.train_sizes == [0], \
        'train_sizes and graph_sizes must have the same length'
    assert len(opts.test_sizes) == len(opts.graph_sizes) or opts.test_sizes == [0], \
        'test_sizes and graph_sizes must have the same length'
    assert len(opts.val_sizes) == len(opts.graph_sizes) or opts.val_sizes == [0], \
        'val_sizes and graph_sizes must have the same length'

    # Create main directory
    path = os.path.join(opts.data_dir, 'op' + '/' + opts.data_distribution + '/' + str(opts.num_agents) +
                        'agents/' + str(opts.num_depots) + 'depots')
    path = os.path.join(path, '_'.join([str(g) for g in opts.graph_sizes]))
    path = path + '_seed{}'.format(opts.seed)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    # Create data for train, test, validation and rollout baseline
    subsets = {'test': opts.test_sizes, 'train': opts.train_sizes, 'val': opts.val_sizes}
    for subset, subset_sizes in subsets.items():
        if not subset_sizes == [0]:
            print('\n' + subset.upper() + ':')
            path_subset = os.path.join(path, subset)
            padding = math.floor(math.log(np.max(subset_sizes), 10)) + 1

            # Create data for each graph size
            for idx, graph_size in enumerate(opts.graph_sizes):

                # Estimate max_length or use default values
                max_length_est = estimate_max_length(graph_size) if opts.estimate_max_length else 0

                # Different folder for each graph size
                print('Graph size = {}'.format(graph_size))
                path_graph = os.path.join(path_subset, str(graph_size))
                if not os.path.exists(path_graph):
                    os.makedirs(path_graph, exist_ok=True)

                if subset == 'test':
                    dataset = generate_data(subset_sizes[idx], graph_size, opts.data_distribution, opts.num_agents,
                                            opts.num_depots, False, max_length_est)
                    save_dataset(dataset, os.path.join(path_graph, 'data.pkl'))
                else:
                    # DataLoader
                    datacreator = ScenarioGenerator(subset_sizes[idx], graph_size, opts.data_distribution,
                                                    opts.num_agents, opts.num_depots, opts.compass_oracle,
                                                    max_length_est, path_graph, padding)
                    dataloader = DataLoader(datacreator, batch_size=opts.batch_size, num_workers=opts.num_workers)

                    # Create batches of data
                    for batch_id, batch in enumerate(tqdm(dataloader)):
                        continue
    print('Finished')
