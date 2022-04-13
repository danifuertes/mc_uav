import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
from utils.functions import sample_many


class Encoder(nn.Module):
    """Maps a graph represented as an input sequence to a hidden vector"""

    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.init_hx, self.init_cx = self.init_hidden(hidden_dim)

    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        return output, hidden

    def init_hidden(self, hidden_dim):
        """Trainable initial hidden state"""
        std = 1. / math.sqrt(hidden_dim)
        enc_init_hx = nn.Parameter(torch.FloatTensor(hidden_dim))
        enc_init_hx.data.uniform_(-std, std)

        enc_init_cx = nn.Parameter(torch.FloatTensor(hidden_dim))
        enc_init_cx.data.uniform_(-std, std)
        return enc_init_hx, enc_init_cx


class Attention(nn.Module):
    """A generic attention module for a decoder in seq2seq"""

    def __init__(self, dim, use_tanh=False, C=10.):
        super(Attention, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(dim, dim)
        self.project_ref = nn.Conv1d(dim, dim, 1, 1)
        self.C = C  # tanh exploration
        self.tanh = nn.Tanh()

        self.v = nn.Parameter(torch.FloatTensor(dim))
        self.v.data.uniform_(-(1. / math.sqrt(dim)), 1. / math.sqrt(dim))

    def forward(self, query, ref):
        """
        Args:
            query: is the hidden state of the decoder at the current
                time step. batch x dim
            ref: the set of hidden states from the encoder.
                sourceL x batch x hidden_dim
        """
        # ref is now [batch_size x hidden_dim x sourceL]
        ref = ref.permute(1, 2, 0)
        q = self.project_query(query).unsqueeze(2)  # batch x dim x 1
        e = self.project_ref(ref)  # batch_size x hidden_dim x sourceL
        # expand the query by sourceL
        # batch x dim x sourceL
        expanded_q = q.repeat(1, 1, e.size(2))
        # batch x 1 x hidden_dim
        v_view = self.v.unsqueeze(0).expand(
            expanded_q.size(0), len(self.v)).unsqueeze(1)
        # [batch_size x 1 x hidden_dim] * [batch_size x hidden_dim x sourceL]
        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)
        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u
        return e, logits


class CriticNetworkLSTM(nn.Module):
    """Useful as a baseline in REINFORCE updates"""

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 n_process_block_iters,
                 tanh_exploration,
                 use_tanh):
        super(CriticNetworkLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_process_block_iters = n_process_block_iters

        self.encoder = Encoder(embedding_dim, hidden_dim)

        self.process_block = Attention(hidden_dim, use_tanh=use_tanh, C=tanh_exploration)
        self.sm = nn.Softmax(dim=1)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs):
        """
        Args:
            inputs: [embedding_dim x batch_size x sourceL] of embedded inputs
        """
        inputs = inputs.transpose(0, 1).contiguous()

        encoder_hx = self.encoder.init_hx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)
        encoder_cx = self.encoder.init_cx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)

        # encoder forward pass
        enc_outputs, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_hx, encoder_cx))

        # grab the hidden state and process it via the process block
        process_block_state = enc_h_t[-1]
        for i in range(self.n_process_block_iters):
            ref, logits = self.process_block(process_block_state, enc_outputs)
            process_block_state = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
        # produce the final scalar output
        out = self.decoder(process_block_state)
        return out


class GPN(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 **kwargs):
        super(GPN, self).__init__()

        # Problem
        self.problem = problem
        self.is_tsp = problem.NAME == "tsp"
        self.is_op = problem.NAME == "op"
        self.is_vrp = problem.NAME == 'cvrp' or problem.NAME == 'sdvrp'
        self.is_pctsp = problem.NAME == 'pctsp'
        assert self.is_tsp or self.is_op, "GPN only supported for TSP and OP yet"     # TODO: remove in the future

        # Dimensions
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Input dimension
        if self.is_tsp:
            self.input_dim = 2  # x, y (TSP)
        else:
            self.input_dim = 4  # x, y, prize, remaining length (OP)

        # Initial embedding projection
        std = 1. / math.sqrt(embedding_dim)
        self.node_embed = nn.Parameter(torch.FloatTensor(self.input_dim, embedding_dim))
        self.node_embed.data.uniform_(-std, std)

        # Placeholder for the starting node
        self.init_node_placeholder = nn.Parameter(torch.FloatTensor(embedding_dim))
        self.init_node_placeholder.data.uniform_(-std, std)

        # Weights for the GNN
        self.W1 = nn.Linear(embedding_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.W3 = nn.Linear(hidden_dim, hidden_dim)

        # Aggregation function for the GNN
        self.agg_1 = nn.Linear(embedding_dim, hidden_dim)
        self.agg_2 = nn.Linear(hidden_dim, hidden_dim)
        self.agg_3 = nn.Linear(hidden_dim, hidden_dim)

        # Parameters to regularize the GNN
        r1 = torch.ones(1).cuda()
        r2 = torch.ones(1).cuda()
        r3 = torch.ones(1).cuda()
        self.r1 = nn.Parameter(r1)
        self.r2 = nn.Parameter(r2)
        self.r3 = nn.Parameter(r3)

        # LSTM
        self.lstm = nn.LSTMCell(embedding_dim, hidden_dim)

        # Attention
        self.pointer = Attention(hidden_dim, use_tanh=tanh_clipping > 0, C=tanh_clipping)
        self.glimpse = Attention(hidden_dim, use_tanh=False)
        self.softmax = nn.Softmax(dim=1)
        self.n_glimpses = 1
        self.tanh_exploration = tanh_clipping

        # Mask
        self.mask_glimpses = mask_inner
        self.mask_logits = mask_logits
        self.decode_type = None  # Needs to be set explicitly before use

    def forward(self, inputs, eval_tours=None, return_pi=False):

        # Make predictions
        _log_p, pi = self._inner(inputs, eval_tours=eval_tours)

        # Calculate cost
        cost, mask = self.problem.get_costs(inputs, pi)

        # Log likelihood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = self._calc_log_likelihood(_log_p, pi, mask)
        if return_pi:
            return cost, ll, pi
        return cost, ll

    def _inner(self, inputs, eval_tours=None):

        # State
        state = self.problem.make_state(inputs)

        # TSP: Prepare for node embedding
        if self.is_tsp:
            batch_size, graph_size, _ = inputs.size()
            data = inputs

        # OP: Prepare for node embedding by combining all input data (depot, loc, prize, max_length)
        else:

            # Dimensions
            batch_size, graph_size, _ = inputs['loc'].size()
            graph_size += 1

            # Concatenate depot and loc
            loc = torch.cat((inputs['depot'][:, None, :], inputs['loc']), dim=1)

            # Get (and normalize) future remaining length between current node and each of the next candidate nodes
            max_length = (
                             state.get_remaining_length() -
                             (
                                 inputs['depot'].tile(graph_size).reshape(-1, graph_size, 2) - loc
                             ).norm(p=2, dim=-1)
                         )[:, :, None] / inputs['max_length'].tile(graph_size).reshape(-1, graph_size, 1)

            # Concat depot prize (which is 0) and loc prizes
            prize = torch.cat((torch.zeros_like(state.get_remaining_length()), inputs['prize']), dim=-1)[:, :, None]

            # Concatenate spatial info (loc), prize info (prize) and temporal info (max_length)
            data = torch.cat((loc, prize, max_length), dim=-1)

        # Apply node embedding
        node_embedding = torch.mm(
            data.transpose(0, 1).contiguous().view(-1, self.input_dim),
            self.node_embed
        ).view(graph_size, batch_size, -1)

        # GNN -> context.shape = (batch_size x sourceL x embedding_dim)
        context = self.gnn(node_embedding, graph_size, batch_size)

        # Initialize hidden state of LSTM
        h0 = c0 = Variable(
            torch.zeros(batch_size, self.hidden_dim, out=node_embedding.data.new()), requires_grad=False
        )  # Variable is deprecated, use Tensor
        hidden = (h0, c0)

        # Initialize current node
        x = self.init_node_placeholder.unsqueeze(0).repeat(batch_size, 1)

        # Output lists
        outputs = []
        selections = []

        # Iterate
        i = 0
        while not state.all_finished():

            # Get mask that avoids revisiting nodes
            mask = state.get_mask().squeeze(1)

            # LSTM
            h, c = self.lstm(x, hidden)
            hidden = (h, c)

            # Attention model
            for i in range(self.n_glimpses):
                ref, logits = self.glimpse(h, context)
                # For the glimpses, only mask before softmax, so we have always an L1 norm 1 readout vector
                if self.mask_glimpses:
                    logits[mask] = -np.inf
                # [batch_size x h_dim x sourceL] * [batch_size x sourceL x 1] = [batch_size x h_dim x 1]
                h = torch.bmm(ref, self.softmax(logits).unsqueeze(2)).squeeze(2)
            _, logits = self.pointer(h, context)  # query = h, ref = context

            # Masking before softmax makes probs sum to one
            if self.mask_logits:
                logits[mask] = -np.inf

            # Calculate log_softmax for better numerical stability
            log_p = torch.log_softmax(logits, dim=1)
            probs = log_p.exp()

            if not self.mask_logits:
                # If self.mask_logits, this would be redundant, otherwise we must mask to make sure we don't resample
                # Note that as a result the vector of probs may not sum to one (this is OK for .multinomial sampling)
                # But practically by not masking the logits, a model is learned over all sequences (also infeasible)
                # while only during sampling feasibility is enforced (a.k.a. by setting to 0. here)
                probs[mask] = 0.
                # For consistency, we should also mask out in log_p, but the values set to 0 will not be sampled and
                # Therefore not be used by the reinforce estimator

            # Select nodes from probabilities (ArgMax or Sample)
            idxs = self.decode(probs, mask) if eval_tours is None else eval_tours[:, i]

            # Update state
            state = state.update(idxs)

            # OP: context and node_embedding have to be recalculated
            if self.is_op:

                # Get current position
                current_coords = loc[[b for b in range(batch_size)], state.get_current_node().squeeze()]

                # Calculate again max_length for the current position
                max_length = (
                                 state.get_remaining_length() -
                                 (
                                     current_coords.tile(graph_size).reshape(-1, graph_size, 2) - loc
                                 ).norm(p=2, dim=-1)
                             )[:, :, None]

                # Concatenate everything again
                data = torch.cat((loc, prize, max_length), dim=-1)

                # Apply node embedding again
                node_embedding = torch.mm(
                    data.transpose(0, 1).contiguous().view(-1, self.input_dim),
                    self.node_embed
                ).view(graph_size, batch_size, -1)

                # Calculate the context (with the GNN) again
                context = self.gnn(node_embedding, graph_size, batch_size)

            # Gather node embedding of selected nodes
            x = torch.gather(
                node_embedding,
                0,
                idxs.contiguous().view(1, batch_size, 1).expand(1, batch_size, *node_embedding.size()[2:])
            ).squeeze(0)

            # Use outs to point to next object
            outputs.append(log_p)
            selections.append(idxs)
            i += 1

        return torch.stack(outputs, 1), torch.stack(selections, 1)

    def gnn(self, node_embedding, graph_size, batch_size):
        """Graph Neural Network (GNN) to get the context information for the Attention model."""
        node_embedding = node_embedding.view(-1, self.embedding_dim)
        context = self.r1 * self.W1(node_embedding) + (1 - self.r1) * F.relu(self.agg_1(node_embedding))
        context = self.r2 * self.W2(context) + (1 - self.r2) * F.relu(self.agg_2(context))
        context = self.r3 * self.W3(context) + (1 - self.r3) * F.relu(self.agg_3(context))
        return context.view(graph_size, batch_size, -1)  # output: (sourceL x batch_size x embedding_dim)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type

    def decode(self, probs, mask):
        if self.decode_type == "greedy":
            _, idxs = probs.max(1)
            assert not mask.gather(1, idxs.unsqueeze(-1)).data.any(), \
                "Decode greedy: infeasible action has maximum probability"
        elif self.decode_type == "sampling":
            idxs = probs.multinomial(1).squeeze(1)
            # Check if sampling went OK, can go wrong due to bug on GPU
            while mask.gather(1, idxs.unsqueeze(-1)).data.any():
                print(' [!] resampling due to race condition')
                idxs = probs.multinomial().squeeze(1)
        else:
            assert False, "Unknown decode type"

        return idxs

    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        return sample_many(
            lambda input: self._inner(input),  # Need to unpack tuple into arguments
            lambda input, pi: self.problem.get_costs(input, pi),  # Don't need embeddings as input to get_costs
            input, batch_rep, iter_rep
        )
