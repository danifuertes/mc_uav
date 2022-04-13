import os
import torch
import random
import argparse
import numpy as np
from matplotlib import pyplot as plt

from utils import load_model
from utils.functions import load_problem
from utils.data_utils import set_seed, str2bool
from nets.attention_model import AttentionModel
from nets.pointer_network import PointerNetwork
from nets.gpn import GPN


def arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=5, help='Random seed to use')

    # Model
    parser.add_argument('--num_agents', type=int, default=4, help="Number of agents")
    parser.add_argument('--num_depots', type=int, default=1, help="Number of depots. Options are 1 or 2. num_depots=1"
                        "means that the start and end depot are the same. num_depots=2 means that they are different")  # 2 depots is only supported for Attention on OP
    parser.add_argument('--load_path', help='Path to load model. Just indicate the directory where epochs are saved or'
                                            'the directory + the specific epoch you want to load')
    parser.add_argument('--baseline', default=None, help="If not None, it will execute the given baseline for the"
                                                         "specified problem instead of the loaded model")
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')

    # Problem
    parser.add_argument('--problem', default='op', help="The problem to solve, default 'tsp'")
    parser.add_argument('--graph_size', type=int, default=20, help="The size of the problem graph")
    parser.add_argument('--data_distribution', type=str, default='coop',
                        help='Data distribution to use during training. Options: coop, nocoop, const, dist, unif')
    parser.add_argument('--test_coop', type=str2bool, default=True,
                        help="For the OP with coop/nocoop distribution, set test_coop=True to see the multi-agent plot")

    opts = parser.parse_args(args)
    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda

    # Check problem is correct
    assert opts.problem in ('tsp', 'op'), 'Supported problems are TSP and OP'
    # TODO: add VRP and PCTSP
    assert opts.num_agents > 0, 'num_agents must be greater than 0'

    # Check baseline is correct for the given problem
    if opts.baseline is not None:
        if opts.problem == 'op':  # TODO: add TSP, VRP and PCTSP baselines
            assert opts.baseline in ('tsili', 'tsiligreedy', 'gurobi', 'gurobigap', 'gurobit', 'opga', 'ortools',
                                     'compass'),\
                'Supported baselines for OP are tsili, gurobi, opga, ortools and compass'
    return opts


def assign_colors(n):
    color = {k: [] for k in 'rgb'}
    for i in range(n):
        temp = {k: random.randint(0, 230) for k in 'rgb'}
        for k in temp:
            while 1:
                c = temp[k]
                t = set(j for j in range(c - 25, c + 25) if 0 <= j <= 230)
                if t.intersection(color[k]):
                    temp[k] = random.randint(0, 230)
                else:
                    break
            color[k].append(temp[k])
    return [(color['r'][i] / 256, color['g'][i] / 256, color['b'][i] / 256) for i in range(n)]


def baselines(baseline, problem, dataset, device):

    # Prepare inputs
    inputs = dataset.data[0]
    if not (baseline == 'tsili' or baseline == 'tsiligreedy'):
        if problem.NAME == 'tsp':
            inputs = inputs.detach().numpy().tolist()
        else:
            for k, v in inputs.items():
                inputs[k] = v.detach().numpy().tolist()

    # OR-TOOLS
    if baseline == 'ortools':
        from problems.op.op_ortools import solve_op_ortools
        model_name = 'OR-Tools'
        _, tour = solve_op_ortools(inputs['depot'], inputs['loc'], inputs['prize'], inputs['max_length'])

    # Genetic Algorithm (GA)
    elif baseline == 'opga':
        from problems.op.opga.opevo import run_alg as run_opga_alg
        model_name = 'GA'
        _, tour, _ = run_opga_alg(
            [(*pos, p) for p, pos in zip([0, 0] + inputs['prize'], [inputs['depot'], inputs['depot']] + inputs['loc'])],
            inputs['max_length'], return_sol=True, verbose=False)
        tour = np.array(tour, dtype=int)[:-1, 3]
        tour[1:] = tour[1:] - 1

    # Compass
    elif baseline == 'compass':
        from subprocess import check_call
        from problems.op.op_baseline import write_oplib, read_oplib
        model_name = 'Compass'
        name = 'temp'
        executable = os.path.abspath(os.path.join('problems', 'op', 'compass', 'compass'))
        problem_filename = os.path.abspath("{}.oplib".format(name))
        tour_filename = os.path.abspath("{}.tour".format(name))
        log_filename = os.path.abspath("{}.log".format(name))

        write_oplib(problem_filename, inputs['depot'], inputs['loc'], inputs['prize'], inputs['max_length'], name=name)
        with open(log_filename, 'w') as f:
            check_call([executable, '--op', '--op-ea4op', problem_filename, '-o', tour_filename], stdout=f, stderr=f)
        tour = read_oplib(tour_filename, n=len(inputs['prize']))
        tour = np.insert(tour, 0, 0)

        os.remove(problem_filename)
        os.remove(tour_filename)
        os.remove(log_filename)

    # Gurobi
    elif baseline == 'gurobi' or baseline == 'gurobigap' or baseline == 'gurobit':
        import re
        from problems.op.op_gurobi import solve_euclidian_op as solve_euclidian_op_gurobi
        model_name = 'Gurobi'
        match = re.match(r'^([a-z]+)(\d*)$', baseline)
        assert match
        method = match[1]
        runs = 1 if match[2] == '' else int(match[2])
        cost, tour = solve_euclidian_op_gurobi(
            inputs['depot'], inputs['loc'], inputs['prize'], inputs['max_length'], threads=1,
            timeout=runs if method[6:] == "t" else None,
            gap=float(runs) if method[6:] == "gap" else None
        )

    # Tsiligirides
    else:
        import re
        from tqdm import tqdm
        from torch.utils.data import DataLoader
        from utils import move_to, sample_many
        from problems.op.tsiligirides import op_tsiligirides

        if baseline == 'tsili':
            model_name = 'Tsili'
            sample = False
            num_samples = 1
        else:
            model_name = 'Tsili (greedy)'
            sample = True
            match = re.match(r'^([a-z]+)(\d*)$', baseline)
            assert match
            runs = 1 if match[2] == '' else int(match[2])
            num_samples = runs

        max_calc_batch_size = 1000
        eval_batch_size = max(1, max_calc_batch_size // num_samples)

        dataloader = DataLoader(dataset)
        tour = []
        for batch in tqdm(dataloader, mininterval=0.1):
            batch = move_to(batch, device)

            with torch.no_grad():
                if num_samples * eval_batch_size > max_calc_batch_size:
                    assert eval_batch_size == 1
                    assert num_samples % max_calc_batch_size == 0
                    batch_rep = max_calc_batch_size
                    iter_rep = num_samples // max_calc_batch_size
                else:
                    batch_rep = num_samples
                    iter_rep = 1
                sequences, costs = sample_many(
                    lambda inp: (None, op_tsiligirides(inp, sample)),
                    problem.get_costs,
                    batch, batch_rep=batch_rep, iter_rep=iter_rep)
                tour.append([np.insert(np.trim_zeros(pi.cpu().numpy()), 0, 0) for cost, pi in zip(costs, sequences)])

        if problem.NAME == 'tsp':
            inputs = inputs.detach().numpy().tolist()
        else:
            for k, v in inputs.items():
                inputs[k] = v.detach().numpy().tolist()

    # Lists to numpy arrays
    if problem.NAME == 'tsp':
        inputs = np.array(inputs)
    else:
        for k, v in inputs.items():
            inputs[k] = np.array(v)

    return np.array(tour).squeeze(), inputs, model_name


def plot_tour(tour, inputs, problem, model_name, data_dist='', num_depots=1):
    """
    Plot a given tour.
    # Arguments
        tour (numpy array): ordered list of nodes.
        inputs (dict or numpy array): if TSP, inputs is an array containing the coordinates of the nodes. Otherwise, it
        is a dict with the coordinates of the nodes (loc) and the depot (depot), and other possible features.
        problem (str): name of the problem.
        model_name (str): name of the model.
        data_dist (str): type of prizes for the OP. For any other problem, just set this to ''.
        num_depots: number of depots. Options are 1 or 2.
    """

    # Initialize plot
    fig, ax = plt.subplots()
    ax.set_xlim([-.05, .05])
    ax.set_ylim([-.05, .05])

    # Data
    depot = inputs[tour[0]] if problem == 'tsp' else inputs['depot']
    if num_depots > 1:
        depot2 = inputs['depot2']
        plt.scatter(depot2[0], depot2[1], c='r')
    loc = np.delete(inputs, tour[0], axis=0) if problem == 'tsp' else inputs['loc']

    # Plot nodes (black circles) and depot (red circle)
    plt.scatter(depot[0], depot[1], c='b')
    plt.scatter(loc[..., 0], loc[..., 1], c='k')

    # If tour starts and ends in depot
    if len(tour.shape) == 0:
        # Set title
        title = problem.upper()
        title += ' (' + data_dist.lower() + ')' if len(data_dist) > 0 else title
        title += ': Length = 0'
        if problem == 'op':
            # Add OP rewards to the title (if problem is OP)
            prize = inputs['prize']
            title += ' / {:.4g} | Prize = {:.4g} / {:.4g}'.format(inputs['max_length'], 0, np.sum(prize))
        ax.set_title(title)
        plt.show()
        return

    # Calculate the length of the tour
    loc = np.insert(loc, tour[0], depot, axis=0) if problem == 'tsp' else np.concatenate(([depot], loc), axis=0)
    if num_depots > 1:
        loc = np.concatenate((loc, [depot2]), axis=0)
    nodes = np.take(loc, tour, axis=0)
    d = np.sum(np.linalg.norm(nodes[1:] - nodes[:-1], axis=1)) + np.linalg.norm(nodes[0] - depot)

    # Set title
    title = problem.upper()
    title += ' (' + data_dist.lower() + ')' if len(data_dist) > 0 else ''
    title += ' - {:s}: Length = {:.4g}'.format(model_name, d)
    if problem == 'op':
        # Add OP prize to the title (if problem is OP)
        prize = inputs['prize']
        reward = np.sum(np.take(prize, tour[:-1] - 1))
        title += ' / {:.4g} | Prize = {:.4g} / {:.4g}'.format(inputs['max_length'], reward, np.sum(prize))
    ax.set_title(title)

    # Add the start depot at the start of the tour
    tour = np.insert(tour, 0, 0, axis=0)

    # Draw arrows
    for i in range(1, tour.shape[0]):
        dx = loc[tour[i], 0] - loc[tour[i - 1], 0]
        dy = loc[tour[i], 1] - loc[tour[i - 1], 1]
        plt.arrow(loc[tour[i - 1], 0], loc[tour[i - 1], 1], dx, dy, head_width=.025, fc='g', ec=None,
                  length_includes_head=True)
    plt.show()


def plot_multitour(num_agents, tours, inputs, problem, model_name, data_dist='', num_depots=1):
    """
    Plot the tours of all the agents.
    # Arguments
        num_agents: number of agents.
        tours (numpy array): ordered list of nodes.
        inputs (dict or numpy array): if TSP, inputs is an array containing the coordinates of the nodes. Otherwise, it
        is a dict with the coordinates of the nodes (loc) and the depot (depot), and other possible features.
        problem (str): name of the problem.
        model_name (str): name of the model.
        data_dist (str): type of prizes for the OP. For any other problem, just set this to ''.
    """

    data_dist_dir = '' if data_dist == '' else '_{}'.format(data_dist)
    image_dir = 'images/{}_{}'.format(model_name.lower(), problem.lower()) + data_dist_dir
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # Initialize global plots
    fig1 = plt.figure(num_agents)
    plt.xlim([-.05, .05])
    plt.ylim([-.05, .05])
    fig2 = plt.figure(num_agents + 1)
    plt.xlim([-.05, .05])
    plt.ylim([-.05, .05])

    # Assign a color to each agent
    colors = assign_colors(num_agents + 2)
    color_shared = np.array(colors[-2]).reshape(1, -1)
    color_depot = np.array(colors[-1]).reshape(1, -1)

    length_sum, prize_sum, prize_max = 0, 0, 0
    for agent in range(num_agents):
        tour = tours[agent]
        tour = tour if tour.size > 1 else np.array([tour])
        color = np.array(colors[agent]).reshape(1, -1)

        # Initialize individual plots
        fig = plt.figure(agent)
        plt.xlim([-.05, .05])
        plt.ylim([-.05, .05])

        # Data
        loc = inputs[agent]['loc']
        prize = inputs[agent]['prize']
        depot = inputs[agent]['depot']
        if num_depots > 1:
            depot2 = inputs[agent]['depot2']
        max_length = inputs[agent]['max_length']

        # Plot region assignment
        plt.figure(num_agents + 1)
        plt.scatter(loc[prize == 1][..., 0], loc[prize == 1][..., 1], c=color, label='Agent {}'.format(agent))
        plt.scatter(depot[0], depot[1], s=200, c=color_depot, marker='^', label='Depot' if agent == 0 else '')
        if num_depots > 1:
            plt.scatter(depot2[0], depot2[1], s=200, c=color_depot, marker='v', label='Depot' if agent == 0 else '')

        # Plot regions
        plt.figure(num_agents)
        plt.scatter(loc[prize == 1][..., 0], loc[prize == 1][..., 1], c=color, label='Agent {}'.format(agent))
        plt.scatter(depot[0], depot[1], s=200, c=color_depot, marker='^', label='Depot' if agent == 0 else '')
        if num_depots > 1:
            plt.scatter(depot2[0], depot2[1], s=200, c=color_depot, marker='v', label='Depot' if agent == 0 else '')
        if agent == 0:
            for l in range(len(loc)):
                plt.text(loc[l, 0] + .005, loc[l, 1] + .005, str(l + 1))
        plt.figure(agent)
        plt.scatter(depot[0], depot[1], s=200, c=color_depot, marker='^', label='Depot')
        if num_depots > 1:
            plt.scatter(depot2[0], depot2[1], s=200, c=color_depot, marker='v', label='Depot')
        plt.scatter(loc[prize == 1][..., 0], loc[prize == 1][..., 1], c=color, label='Initial')
        plt.scatter(loc[prize != 1][..., 0], loc[prize != 1][..., 1], c=color_shared, label='Shared')
        for l in range(len(loc)):
            plt.text(loc[l, 0] + .005, loc[l, 1] + .005, str(l + 1))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.9))
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Calculate the length of the tour
        loc = np.concatenate(([depot], loc), axis=0)
        if num_depots > 1:
            loc = np.concatenate((loc, [depot2]), axis=0)
        nodes = np.take(loc, tour, axis=0)
        d = np.sum(np.linalg.norm(nodes[1:] - nodes[:-1], axis=1)) + np.linalg.norm(nodes[-1] - nodes[0])
        length_sum += d

        # Agent information
        info = problem.upper()
        info += ' (' + data_dist.lower() + ')' if len(data_dist) > 0 else info
        info += ' - {:s} {}: Length = {:.4g}'.format(model_name, agent, d)

        # Add OP prize to the title
        reward = np.sum(np.take(prize, tour[:-1] - 1))
        prize_sum += reward
        prize_max += np.sum(prize)
        info += ' / {:.4g} | Prize = {:.4g} / {:.4g}'.format(max_length, reward, np.sum(prize))
        plt.title(info)

        # Add the start depot and the end depot to the tour
        if tour[0] != 0:
            tour = np.insert(tour, 0, 0, axis=0)
        elif tour[-1] != loc.shape[0] - 1:
            tour = np.insert(tour, len(tour), 0, axis=0)
        print('Agent {}: '.format(agent), tour)

        # Draw arrows
        for i in range(1, tour.shape[0]):
            dx = loc[tour[i], 0] - loc[tour[i - 1], 0]
            dy = loc[tour[i], 1] - loc[tour[i - 1], 1]
            plt.figure(agent)
            plt.arrow(loc[tour[i - 1], 0], loc[tour[i - 1], 1], dx, dy, head_width=.025, fc=color, ec=color,
                      length_includes_head=True, alpha=0.5)
            plt.figure(num_agents)
            plt.arrow(loc[tour[i - 1], 0], loc[tour[i - 1], 1], dx, dy, head_width=.025, fc=color, ec=color,
                      length_includes_head=True, alpha=0.5)
        fig.savefig(image_dir + '/agent_{}.png'.format(agent), dpi=150)

    # Plot region assignment
    plt.figure(num_agents + 1)
    plt.title('Region assignment')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.85))
    plt.tight_layout(rect=[0.05, 0, 1, 1])
    fig2.savefig(image_dir + '/assignment.png', dpi=150)

    # Agents information
    info = problem.upper()
    info += ' (' + data_dist.lower() + ')' if len(data_dist) > 0 else ''
    info += ' - {:s}: Av. Length = {:.3g}'.format(model_name, length_sum / num_agents)
    info += ' / {:.3g} | Total Prize = {:.3g} / {:.3g}'.format(inputs[0]['max_length'], prize_sum, np.sum(prize_max))
    plt.figure(num_agents)
    plt.title(info)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.85))
    plt.tight_layout(rect=[0.05, 0, 1, 1])
    fig1.savefig(image_dir + '/solution.png', dpi=150)
    plt.show()


def main(opts):

    # Set seed for reproducibility
    set_seed(opts.seed)

    # Set the device
    device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Load problem
    problem = load_problem(opts.problem)
    dataset = problem.make_dataset(size=opts.graph_size, num_samples=1, distribution=opts.data_distribution,
                                   test_coop=opts.test_coop, num_agents=opts.num_agents, num_depots=opts.num_depots)
    inputs = dataset.data[0]

    # Apply baseline (OR-Tools, Compass, GA, Tsiligirides, Gurobi) instead of a trained model (Transformer, PN, GPN)
    if opts.baseline is not None:
        if problem.NAME == 'op' and opts.data_distribution == 'coop' and opts.test_coop:
            tours, inputs_dict = [], {}
            for agent in range(opts.num_agents):
                ds = dataset
                ds.data[0] = inputs[agent]
                tour, inp, model_name = baselines(opts.baseline, problem, ds, device)
                tours.append(tour)
                inputs_dict[agent] = inp
            plot_multitour(opts.num_agents, tours, inputs_dict, problem.NAME, model_name,
                           data_dist=opts.data_distribution)
        else:
            tour, inputs, model_name = baselines(opts.baseline, problem, dataset, device)

            # Print/Plot results
            print(tour)
            plot_tour(tour, inputs, problem.NAME, model_name)
        return

    # Load model (Transformer, PN, GPN) for evaluation on the chosen device
    model, _ = load_model(opts.load_path)
    model.set_decode_type('greedy')
    model.num_depots = opts.num_depots
    model.eval()  # Put in evaluation mode to not track gradients
    model.to(device)
    if isinstance(model, AttentionModel):
        model_name = 'Transformer'
    elif isinstance(model, PointerNetwork):
        model_name = 'Pointer'
    else:
        assert isinstance(model, GPN), 'Model should be an instance of AttentionModel, PointerNetwork or GPN'
        model_name = 'GPN'

    # OP (coop)
    if problem.NAME == 'op' and (opts.data_distribution == 'coop' or opts.data_distribution == 'nocoop') and opts.test_coop:
        tours = []
        for i in range(opts.num_agents):
            for k, v in inputs[i].items():
                inputs[i][k] = v.unsqueeze(0).to(device)
            _, _, tour = model(inputs[i], return_pi=True)
            tours.append(tour.cpu().detach().numpy().squeeze())
            for k, v in inputs[i].items():
                inputs[i][k] = v.cpu().detach().numpy().squeeze()
        plot_multitour(opts.num_agents, tours, inputs, problem.NAME, model_name, data_dist=opts.data_distribution,
                       num_depots=opts.num_depots)
        return

    # TSP
    elif problem.NAME == 'tsp':
        inputs = inputs.unsqueeze(0).to(device)

    # VRP, PCTSP and OP (const, dist, unif)
    else:
        for k, v in inputs.items():
            inputs[k] = v.unsqueeze(0).to(device)

    # Calculate tour
    _, _, tour = model(inputs, return_pi=True)

    # Torch tensors to numpy
    tour = tour.cpu().detach().numpy().squeeze()
    if problem.NAME == 'tsp':
        inputs = inputs.cpu().detach().numpy().squeeze()
    else:
        for k, v in inputs.items():
            inputs[k] = v.cpu().detach().numpy().squeeze()

    # Print/Plot results
    print(np.insert(tour, 0, 0, axis=0))
    plot_tour(tour, inputs, problem.NAME, model_name, data_dist=opts.data_distribution, num_depots=opts.num_depots)


if __name__ == "__main__":
    main(arguments())
