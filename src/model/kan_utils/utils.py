import torch
import numpy as np
from tqdm import tqdm


def plot(kan, beta=3, norm_alpha=False, scale=1.0, tick=False, in_vars=None, out_vars=None, title=None, ax=None):
    """
    plot KAN. Before plot, kan(x) should be run on a typical input to collect statistics on activations functions.

    Args:
    -----
        beta : float
            positive number. control the transparency of each activation. transparency = tanh(beta*l1).
        norm_alpha: bool, default False
            If True, normalize transparency within layer such that higher alpha is set to 1
        scale : float
            control the size of the insert plot of the activation functions
        in_vars: None or list of str
            the name(s) of input variables
        out_vars: None or list of str
            the name(s) of output variables
        title: None or str
            title
        tick: bool, default False
            draw ticks on insert plot
        ax: Axes, default None
            If None, create a new figure

    Returns:
    --------
        Figure

    Example
    -------
    >>> # see more interactive examples in demos
    >>> plot(model)
    """

    import matplotlib.pyplot as plt
    import networkx as nx

    depth = len(kan.layers)

    # Add nodes to graph, choose position at the same time
    pos = {}
    G = nx.Graph()
    for n, l in enumerate(kan.width):
        for m in range(l):
            G.add_node((n, m))
            pos[(n, m)] = [(1 / (2 * l) + m / l) * (1 - 0.1 * (n % 2)), n]

    # Add network edges
    for la in range(depth):
        for i in range(kan.width[la]):
            for j in range(kan.width[la + 1]):
                G.add_edge((la, i), (la + 1, j))

    if ax is None:
        _, ax = plt.subplots()

    nx.draw_networkx_nodes(G, pos, ax=ax)
    # Plot in and out vars if available
    offset = 0  # 0.15  # Find offset as size of a node ??
    mask_in = kan.layers[0].mask.cpu().detach().numpy()
    if in_vars is not None:
        name_attrs = {(0, m): in_vars[m] for m in range(kan.width[0])}
        nx.draw_networkx_labels(G, {n: (x, y - offset) for n, (x, y) in pos.items()}, labels=name_attrs, font_color="red", ax=ax)
    elif mask_in.shape[0] != mask_in.shape[1]:  # If there is some permutation invariants inputs, lets labels them appropeially
        groups = np.argmax(mask_in, axis=0)  # Group to which belong each input
        name_attrs = {(0, m): groups[m] for m in range(kan.width[0])}
        nx.draw_networkx_labels(G, {n: (x, y - offset) for n, (x, y) in pos.items()}, labels=name_attrs, font_color="yellow", ax=ax)
    if out_vars is not None:
        name_attrs = {}
        for m in range(kan.width[-1]):
            name_attrs[(len(kan.width) - 1, m)] = out_vars[m]
        nx.draw_networkx_labels(G, {n: (x, y + offset) for n, (x, y) in pos.items()}, labels=name_attrs, font_color="red", ax=ax)

    def score2alpha(score):
        return np.tanh(beta * score)

    # Add insert plot of each activation functions
    inserts_axes = []
    act_lines = []

    for la in range(depth):
        inserts_axes.append([[None for _ in range(kan.width[la + 1])] for _ in range(kan.width[la])])
        act_lines.append([[None for _ in range(kan.width[la + 1])] for _ in range(kan.width[la])])
        if hasattr(kan.layers[la], "l1_norm"):
            alpha = score2alpha(kan.layers[la].l1_norm.cpu().detach().numpy())
            alpha = alpha / (alpha.max() if norm_alpha else 1.0)
            # Take for ranges, either the extremal of the centers or the min/max of the data
            ranges = [torch.linspace(kan.layers[la].min_vals[d].item(), kan.layers[la].max_vals[d].item(), 150) for d in range(kan.width[la])]
            x_in = torch.stack(ranges, dim=1)
            acts_vals = kan.layers[la].activations_eval(x_in).cpu().detach().numpy()
            x_ranges = x_in.cpu().detach().numpy()
            # Take mask into account
            mask_la = kan.layers[la].mask.cpu().detach().numpy()  # L'idée c'est d'avoir mask [i][j] = True/False pour savoir si on plot
            mask = np.zeros(mask_la.shape[1], dtype=bool)
            mask[np.argmax(mask_la, axis=1)] = True  # Only plot first graph for each group

            for i in range(kan.width[la]):
                for j in range(kan.width[la + 1]):
                    u, v = (la, i), (la + 1, j)
                    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], alpha=alpha[j, i], ax=ax)
                    if mask[i]:
                        # Compute central position of the edge
                        x = (pos[u][0] + pos[v][0]) / 2
                        y = (pos[u][1] + pos[v][1]) / 2

                        width = scale * 0.1
                        height = scale * 0.1

                        # Créer un axe en insert
                        inset_ax = ax.inset_axes([x - 0.5 * width, y - 0.5 * height, width, height], transform=ax.transData, box_aspect=1.0)
                        if tick is False:
                            inset_ax.set_xticks([])
                            inset_ax.set_yticks([])
                        else:
                            inset_ax.tick_params(axis="both", which="both", length=0)  # Rendre les ticks invisibles

                            for label in inset_ax.get_xticklabels() + inset_ax.get_yticklabels():
                                label.set_alpha(alpha[j, i])

                        act_lines[la][i][j] = inset_ax.plot(x_ranges[:, i], acts_vals[:, j, i], "-", color="red", alpha=alpha[j, i])[0]
                        for spine in inset_ax.spines.values():
                            spine.set_alpha(alpha[j, i])
                        inset_ax.patch.set_alpha(alpha[j, i])
                        inserts_axes[la][i][j] = inset_ax
        else:
            for i in range(kan.width[la]):
                for j in range(kan.width[la + 1]):
                    u, v = (la, i), (la + 1, j)
                    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax)

    if title is not None:
        ax.set_title(title)
    return inserts_axes, act_lines


def update_plot(kan, inserts_axes, act_lines, beta=3, norm_alpha=False, tick=False):
    """
    plot KAN. Before plot, kan(x) should be run on a typical input to collect statistics on activations functions.

    Args:
    -----
        beta : float
            positive number. control the transparency of each activation. transparency = tanh(beta*l1).
        mask : bool
            If True, plot with mask (need to run prune() first to obtain mask). If False (by default), plot all activation functions.
        mode : bool
            "supervised" or "unsupervised". If "supervised", l1 is measured by absolution value (not subtracting mean); if "unsupervised", l1 is measured by standard deviation (subtracting mean).
        scale : float
            control the size of the insert plot of the activation functions
        in_vars: None or list of str
            the name(s) of input variables
        out_vars: None or list of str
            the name(s) of output variables
        title: None or str
            title

    Returns:
    --------
        Figure

    Example
    -------
    >>> # see more interactive examples in demos
    >>> model = KAN(width=[2,3,1], grid=3, k=3, noise_scale=1.0)
    >>> x = torch.normal(0,1,size=(100,2))
    >>> model(x) # do a forward pass to obtain model.acts
    >>> model.plot()
    """

    depth = len(kan.layers)

    def score2alpha(score):
        return np.tanh(beta * score)

    for la in range(depth):
        if hasattr(kan.layers[la], "l1_norm"):
            alpha = score2alpha(kan.layers[la].l1_norm.cpu().detach().numpy())
            alpha = alpha / (alpha.max() if norm_alpha else 1.0)
            # Take for ranges, either the extremal of the centers or the min/max of the data
            ranges = [torch.linspace(kan.layers[la].min_vals[d], kan.layers[la].max_vals[d], 150) for d in range(kan.width[la])]
            x_in = torch.stack(ranges, dim=1)
            acts_vals = kan.layers[la].activations_eval(x_in).cpu().detach().numpy()
            x_ranges = x_in.cpu().detach().numpy()
            for i in range(kan.width[la]):
                for j in range(kan.width[la + 1]):
                    inset_ax = inserts_axes[la][i][j]
                    if tick is False:
                        inset_ax.set_xticks([])
                        inset_ax.set_yticks([])
                    else:
                        inset_ax.tick_params(axis="both", which="both", length=0)  # Rendre les ticks invisibles

                        for label in inset_ax.get_xticklabels() + inset_ax.get_yticklabels():
                            label.set_alpha(alpha[j, i])
                    act_lines[la][i][j].set_xdata(x_ranges[:, i])
                    act_lines[la][i][j].set_ydata(acts_vals[:, j, i])
                    act_lines[la][i][j].set_alpha(alpha[j, i])

                    inset_ax.set_xlim(x_ranges[0, i], x_ranges[-1, i])
                    inset_ax.set_ylim(acts_vals[:, j, i].min(), acts_vals[:, j, i].max())
                    # act_lines[i][j] = inset_ax.plot(x_ranges[:, i], acts_vals[:, j, i], "-", color="red", alpha=alpha[j, i])
                    for spine in inset_ax.spines.values():
                        spine.set_alpha(alpha[j, i])
                    inset_ax.patch.set_alpha(alpha[j, i])
