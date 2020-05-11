__author__ = 'Saeid SOHILY-KHAH'
"""
Machine learning algorithms: Reinforcement Q-Learning [using networkx]
"""
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Define graph
def load_graph(axes):
    '''
    Define a graph using networkx
    :param: axes:
    :return:
    '''
    graph = nx.Graph()

    # Nodes definition
    nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    dict_nodes = {0: 'START', 1: '1', 2: '2', 3: 'PARTY', 4: 'CASINO', 5: 'SLEEP',
                  6: '6', 7: '7', 8: '8', 9: 'STUDY', 10: 'END'}
    start = 0
    positive_coef = [9]
    negative_coef = [3, 4, 5]
    end = 10
    graph.add_nodes_from(nodes) # add nodes to graph

    # Edges definition
    edges_list = [(0, 8), (1, 2), (1, 8), (2, 3), (2, 9), (3, 4), (3, 5), (6, 8), (7, 9), (9, 10)]
    graph.add_edges_from(edges_list) # add edges to graph

    # Plot graph
    n_size = {0: 2, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 1, 7: 1, 8: 1, 9: 2, 10: 2}
    n_color = {0: 'y', 1: 'b', 2: 'b', 3: 'r', 4: 'r', 5: 'r', 6: 'b', 7: 'b', 8: 'b', 9: 'g', 10: 'y'}
    pos = {0: (310, 40), 6: (200, 30), 8: (260, 100), 1: (260, 200), 2: (260,300),
           3: (130,370), 5: (30,340), 4: (60,460), 9: (330,370), 7: (440,400), 10: (390,460)}

    nx.draw(graph,
            nodelist=n_size.keys(),
            node_size=[v*1100 for v in n_size.values()],  # nodes' size
            node_color=[v for v in n_color.values()],    # nodes' color
            pos=pos, # nodes' position
            with_labels=False,
            ax=axes)

    nx.draw_networkx_labels(graph, pos=pos, labels=dict_nodes, ax=axes) # set nodes' label

    # Graph's legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='START / SUCCESS(END)', markerfacecolor='y', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='NORMAL', markerfacecolor='b', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='POSITIVE COEFFICIENT', markerfacecolor='g', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='NEGATIVE COEFFICIENT', markerfacecolor='r', markersize=15)
                       ]
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    axes.set_title(r'GRAPH PATH: START  $\rightarrow$  SUCCESS')
    axes.legend(handles=legend_elements)
    return graph, start, positive_coef, negative_coef, end


# Define accessible actions in graph according to the edges
def get_accessible_actions(reward, state):
    '''
    Define accessible actions
    :param reward: reward matrix
    :param state:
    :return: acc_actions
    '''
    current_state_row = reward[state,]
    acc_actions = np.where(current_state_row >= 0)[1]
    return acc_actions


# Define accessible actions in graph according to the edges and environments
def get_accessible_actions_wrt_environment(reward, environment, state):
    '''
    Define accessible actions by considering environmental's impact
    :param reward: reward matrix
    :param environment: environment matrix
    :param state:
    :return: acc_actions
    '''
    current_state_row = reward[state,]
    acc_actions = np.where(current_state_row >= 0)[1]

    # Remove negative directions in multiple routes (keep direction in single route)
    environment_row = environment[state, acc_actions]
    if np.sum(environment_row < 0):
        _acc_actions = acc_actions[np.array(environment_row)[0]>=0]
        if len(_acc_actions) > 0:
            acc_actions = _acc_actions
    return acc_actions


# Define a random next accessible action
def sample_next_action(acc_actions):
    '''
    Define a random action between all accessible actions
    :param acc_actions: list of accessible actions
    :return:
    '''
    return int(np.random.choice(acc_actions, 1))


# Collect environmental information
def collect_environmental_data(action, positive_coef, negative_coef):
    '''
    Collect environmental positive and negative information
    :param action:
    :param positive_coef: positive coefficients (e.g. study)
    :param negative_coef: negative coefficients (e.g. party)
    :return:
    '''
    env_data = []
    if action in positive_coef:
        env_data.append('pos')

    if action in negative_coef:
        env_data.append('neg')
    return env_data


# Update Q-learning matrix
def update_Q(Q, reward, current_state, action, lr):
    '''
    Update Q
    :param Q: Q-learning matrix
    :param reward: reward matrix
    :param current_state:
    :param action:
    :param lr: learning rate
    :return:
    '''
    max_index_lst = np.where(Q[action,] == np.max(Q[action,]))[1]
    max_index = int(np.random.choice(max_index_lst, size=1)) # random choice in max_index list
    max_value = Q[action, max_index]
    Q[current_state, action] = reward[current_state, action] + lr * max_value
    return Q


# Plot historical scores
def plot_history_scores(scores_history, scores_history_withenv):
    '''
    Plot Q-learning's historical scores
    :param scores_history: Q-learning scores without considering environment coefficients
    :param scores_history_withenv: Q-learning scores with  considering all environment coefficients
    :return:
    '''
    plt.figure(figsize=(17, 9))
    plt.plot(scores_history, 'r', label='Q-learning scores without considering environment coefficients')
    plt.plot(scores_history_withenv, 'b', label='Q-learning scores with  considering all environment coefficients')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title("Q-LEARNING'S HISTORICAL SCORES", fontsize=12)
    plt.legend(loc='lower right')
    plt.savefig('q_learning_history.png', bbox_inches='tight') # save the plot locally
    return


# ------------------------------------------------ MAIN ----------------------------------------------------
if __name__ == '__main__':
    # Plot graph
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(17, 9))  # set figure shape and size
    fig.tight_layout(pad=3.0)  # set the spacing between subplots in Matplotlib

    # Load graph
    graph, start, positive_coef, negative_coef, end = load_graph(axes[0])

    # Reward initialization (matrix -1)
    matrix_size = len(graph.nodes) # number of nodes in the graph
    reward = np.matrix(np.ones(shape=(matrix_size, matrix_size))) * -1

    # Update reward (assign 0s to edges and 100 to goal-reaching point)
    for edge in graph.edges:
        if edge[1] == end:
            reward[edge] = 100 # goal-reaching point
        else:
            reward[edge] = 0

        if edge[0] == end:
            reward[edge[::-1]] = 100 # goal-reaching point
        else:
            reward[edge[::-1]] = 0 # reverse of point
    reward[end, end] = 100 # add goal point round trip

    # Q_learning initialization
    Q = np.matrix(np.zeros([matrix_size, matrix_size]))
    environment_pos = np.matrix(np.zeros([matrix_size, matrix_size])) # positive environments
    environment_neg = np.matrix(np.zeros([matrix_size, matrix_size])) # negative environments
    iter_number = 1000
    learning_rate = 0.8 #[0,1) - closer to 0->tend to consider only immediate rewards, closer to 1->consider future rewards with greater weight

    # Q_learning initialization: Collect environments' information
    scores_history = []
    for i in range(iter_number):
        if scores_history:
            current_state = np.random.randint(0, int(Q.shape[0]))
        else:
            current_state = start
        accessible_actions = get_accessible_actions(reward, current_state) # possible actions from current state
        action = sample_next_action(accessible_actions) # choose one action between all the possible actions

        Q = update_Q(Q, reward, current_state, action, learning_rate) # update matrix Q

        environment_data = collect_environmental_data(action, positive_coef, negative_coef)
        if 'pos' in environment_data:
            environment_pos[current_state, action] += 1

        if 'neg' in environment_data:
            environment_neg[current_state, action] += 1

        score = np.sum(Q / np.max(Q) * 100) if np.max(Q) > 0 else 0 # score of matrix Q
        scores_history.append(score)

    # Training Q-learning
    Q = np.matrix(np.zeros([matrix_size, matrix_size]))
    environments = environment_pos - environment_neg # combine positive and negative environments to one environment

    scores_history_withenv = []
    for i in range(iter_number):
        if scores_history_withenv:
            current_state = np.random.randint(0, int(Q.shape[0]))
            accessible_actions = get_accessible_actions_wrt_environment(reward, environments, current_state)
        else:
            current_state = start
            accessible_actions = get_accessible_actions(reward, current_state)

        action = sample_next_action(accessible_actions) # choose one action between all the possible actions

        Q = update_Q(Q, reward, current_state, action, learning_rate) # update matrix Q

        environment_data = collect_environmental_data(action, positive_coef, negative_coef)
        if 'pos' in environment_data:
            environments[current_state, action] += 1

        if 'neg' in environment_data:
            environments[current_state, action] += 1

        score = np.sum(Q / np.max(Q) * 100) if np.max(Q) > 0 else 0  # score of matrix Q
        scores_history_withenv.append(score)

    # Plot trained Q matrix (heatmap)
    inner_subplot = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=axes[1])
    ax = plt.Subplot(fig, inner_subplot[0, 0])
    plt.axis('off')
    xticks = yticks = ['START', '1', '2', 'PARTY', 'CASINO', 'SLEEP', '6', '7', '8', 'STUDY', 'END']
    ax.set_title('Q-MATRIX (HEATMAP)', fontsize=12)
    fig.add_subplot(ax)
    sns.set(font_scale=1.2)  # for label size
    color_map = sns.cubehelix_palette(dark=0, light=0.95, as_cmap=True)  # color_map for seaborn plot
    hm = sns.heatmap(Q,
                     cmap=color_map,
                     annot=True,
                     annot_kws={"size": 12},
                     fmt=".1f",
                     xticklabels=xticks,
                     yticklabels=yticks)  # plot Q heatmap
    hm.set_yticklabels(hm.get_yticklabels(), rotation=0)

    # To save the plot locally
    plt.savefig('q_learning.png', bbox_inches='tight')

    # Plot Q-learning's historical scores (uncomment below line to plot)
    #plot_history_scores(scores_history, scores_history_withenv)
    plt.show()