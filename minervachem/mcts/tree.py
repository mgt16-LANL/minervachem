#!/usr/bin/env python
import random
import math
import logging
import numpy as np
# from state import *
from minerva_state import *

# MCTS scalar.  Larger scalar will increase exploitation, smaller will increase exploration.
# SCALAR=1/(2*math.sqrt(2.0))

logname = "mcts.log"
logging.basicConfig(
    filename=logname,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
)
logger = logging.getLogger("mcts")


def sigmoid(x, alpha=1):
    return 1 / (1 + np.exp(-alpha * x))


def boltzmann(x, alpha=1):
    if len(x) == 0:
        raise ValueError
    q = np.exp(alpha * np.asarray(x))
    z = np.sum(q)
    return q / z

scalar = 0.35

class Node:
    """Node class"""

    def __init__(self, state, parent=None):
        """The units of the search tree that are being grown/expanded via MCTS.

        Args:
                state (State class): corresponding state of the node; includes information such as chosen moves, reward, and turn counter
                parent (Node class, optional): parent node of the current node. Defaults to None.
        """
        self.visits = 1
        self.reward = 0.0
        self.state = state
        self.children = []
        self.parent = parent

    def add_child(self, child_state):
        """Add a child node to (aka expand) the current node that has the state child_state.

        Args:
                child_state (State class): state of the child node to be added
        """
        child = Node(child_state, self)
        self.children.append(child)

    def fully_expanded(self):
        """Function to check if the node is fully expanded.
        If the number of a moves already selected is equal to the number children nodes, then the node is fully expanded.

        Args:

        Returns:
                bool: True for fully expanded, False for not fully expanded
        """
        num_moves = (self.state.num_moves)
        if (num_moves == 0):
            return True
        return False  # else return False

    def __repr__(self):
        s = (f"Node; {id(self)} children: {len(self.children)}; visits: {self.visits}; reward: {self.reward/self.visits:.4f}; "
            + repr(self.state)
            )
        return s


def utcbeam(budget, beamsize:int, rootpop:list, alpha:float):
    """Beam search version of tree search where instead of returning the single best child, return the top k children.

    Args:
        budget (_type_): alloted number of simulations
        beamsize (int): k or number of top candidates to return
        rootpop (list): list of nodes to search
        alpha (float): parameter for controlling elasticity of Boltzmann distribution

    Returns:
        list of Nodes: list of top k Nodes
    """
    allchildren = []
    rootpop = [x for x in rootpop if not x.state.terminal()]
    for node in rootpop:
        for iter in range(int(budget)//len(rootpop)):
            front = treepolicy(node, alpha=alpha)

            reward = defaultpolicy(front.state)

            backup(node=front, reward=reward)
        allchildren.extend(node.children)
    return topk(k=beamsize, nodelist=allchildren)

def topk(k:int, nodelist=[]):
    """Calculates the score for all the nodes in nodelist, sorts them by score, and then returns the top k nodes

    Args:
        k (int): _description_
        nodelist (list, optional): _description_. Defaults to [].

    Returns:
        list of Nodes: list of top k Nodes
    """
    scored_nodes = [(n, get_score(n)) for n in nodelist]
    sorted_nodes = sorted(scored_nodes, key=lambda x: x[1])
    topk = sorted_nodes[-k:]
    topk2 = [x for x, _ in topk]
    return topk2


def utcsearch(budget, root, alpha):
    """Function to expand the search tree based on the set tree policy.

    Step 1: for every simulation in the alloted budget, instantiate a new node & state.
    Step 2: apply the treepolicy to the new node & state (selection & expansion)
    Step 3: calculate the reward based on the defaultpolicy (simulation)
    Step 4: update the node the with the reward and number of visits (backpropagation)

    Args:
            budget (int): total number of iterations allotted to MCTS
            root (Node Class): the input node that is the start point of the search
            num_moves_lambda (func, optional): idk i still don't know what this function is supposed to do it's always None. Defaults to None.

    Returns:
            Node class: the best child node aka the child node with the highest reward
    """
    for iter in range(int(budget)):
        front = treepolicy(root, alpha=alpha)
        logger.info(f"This is the front node, the resulting node from treepolicy: {front}")
        reward = defaultpolicy(front.state)
        backup(node=front, reward=reward)
    return bestchild(root, 0, deterministic=True, alpha=alpha)


def treepolicy(node, alpha):
    """Function to either exploit a known child node or explore/expand to a new child node.

    If the current/root node has no children, then expand the children.
    Alternatively (aka half of the time), choose the best known child node.
    If the node already has children and the best child route is not taken, then expand the node.
    If the node is in a terminal state, then select the best child.

    Args:
            node (Node class): root or current node to either expand or exploit
            num_moves_lambda (_type_): _description_

    Returns:
            Node class: returns either the newly expnaded child node or the best known child node
    """
    # a hack to force 'exploitation' in a game where there are many options, and you may never/not want to fully expand first
    while (node.state.terminal() == False):
        if (len(node.children) == 0):
            if not node.fully_expanded():
                return expand(node)
            else:
                return node
        else:
            if node.fully_expanded() == False:
                return expand(node)
            else:
                node = bestchild(node, scalar=scalar, alpha=alpha)
    return node

def expand(node):
    """Function to expand new children nodes.

    Step 1: get the states of the input node's children nodes in a list
    Step 2: instantiate the next state of the input node
    Step 3: while the next state is a move that has already has been tried before and the state is non-terminal, continue to get the next state (aka continue to pick new moves) until a novel state is found

    Args:
            node (Node class): the current node to be expanded

    Returns:
            Node class: latest created child node
    """
    tried_children = [c.state for c in node.children]
    new_state = node.state.next_state()

    while new_state in tried_children and new_state.terminal() == False:
        new_state = node.state.next_state()
    node.add_child(new_state)
    return node.children[-1]

def get_score(node, scalar=scalar):
    """Calculate score of node based on UCB1 tree policy

    Args:
        node (Node): node
        scalar (int, optional): constant that controls exploration vs. exploitation trade off. Defaults to scalar.

    Returns:
        float: score of node
    """
    exploit = node.reward / node.visits
    explore = math.sqrt(2.0 * math.log(node.parent.visits) / float(node.visits))
    score = exploit + scalar * explore
    return score

# current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)
def bestchild(node, scalar, alpha, deterministic:bool=False):
    """Function to look for the best child node (aka has the highest score via UCB1) among all possible children.

    for every child node, calculate the reward with UCB1. Then, pick the child with the highest score.

    Args:
            node (Node class): current node
            scalar (int): constant that balances exploitation vs. exploration tradeoff in tree policy (UCB1)

    Returns:
            _type_: _description_
    """
    allchildren = []
    allscores = []

    bestscore = 0.0
    bestchildren = []

    for c in node.children:
        score = get_score(c)

        if score == bestscore:
            bestchildren.append(c)

        if score > bestscore:
            bestchildren = [c]
            bestscore = score

        # else:
        allchildren.append(c)
        allscores.append(score)

    # deterministic
    if deterministic:
        bestchild = random.choice(bestchildren) # in case of ties in reward values between bestchildren
        # logger.info(f"Deterministic {bestchildren}")
    # probabilistic
    else:
        probs = boltzmann(x=allscores, alpha=alpha)
        bestchild = random.choices(allchildren, weights=probs)[0]
        # logger.info(f"PROBABILITY {allchildren}")
    return bestchild


def defaultpolicy(state):
    """Default policy for rollout/simulation (?)

    While the state is non-terminal, create the next state (aka randomly choose a new move) and calculate the reward

    Args:
            state (State class): _description_

    Returns:
            _type_: reward function call of the next state
    """
    while state.terminal() == False:
        try:
            state = state.next_state()
        except Exception as err:
            raise ValueError(f"Failed on {state}") from err
    return state.reward()  # , state.calc_mae()


def backup(node, reward):
    """Backpropagation function; update the nodes with the number of visits and reward calculated from the simulation.

    Args:
            node (_type_): _description_
            reward (_type_): _description_
    """
    while node != None:
        node.visits += 1
        node.reward += reward
        node = node.parent
    return


def top_percent_search(root_node, top_percent=0.05):
    """Search for and save the top n% of non-best/non-fully expanded children

    Args:
        root_node (_type_): root node
        top_percent (float, optional): desired percentage cutoff. Defaults to 0.05.
    """
    def collect_all_nodes(node, all_nodes):
        reward = node.reward / node.visits
        # print(reward)
        all_nodes.append((reward, node))
        for child in node.children:
            collect_all_nodes(child, all_nodes)
        return all_nodes

    all_nodes = collect_all_nodes(root_node, [])
    all_nodes.sort(reverse=True, key=lambda x: x[0])
    top_count = max(1, int(top_percent * len(all_nodes)))
    top_nodes = [node for _, node in all_nodes[:top_count]]

    return top_nodes


# def tree_search(root_node, sa_target=0, logp_target=2.3406, thresh=2, history=[]):
#     if root_node.state.sa_score is None:
#         loss = 100
#     elif root_node.state.logp is None:
#         loss = 100
#     else:
#         loss = (
#             abs(root_node.state.sa_score - sa_target)
#             + abs(root_node.state.logp - logp_target)
#         ) / 2
#     if loss < thresh:
#         history.append(root_node)
#     if len(root_node.children) == 0:
#         return history
#     for child in root_node.children:
#         tmp = tree_search(
#             child,
#             sa_target=sa_target,
#             logp_target=logp_target,
#             thresh=thresh,
#             history=history,
#         )
#         history = tmp
#     return history

# def top_percent_search(root_node, top_percent=0.05):
#     """Search for and save the top n% of non-best/non-fully expanded children

#     Args:
#         root_node (_type_): root node
#         top_percent (float, optional): desired percentage cutoff. Defaults to 0.05.
#     """
#     def collect_all_nodes(node, all_nodes):
#         reward = node.reward / node.visits
#         # Check if node is already in all_nodes
#         if node not in all_nodes:
#             all_nodes.append((reward, node))
#         for child in node.children:
#             collect_all_nodes(child, all_nodes)
#         return all_nodes

#     all_nodes = collect_all_nodes(root_node, [])
#     all_nodes.sort(reverse=True, key=lambda x: x[0])
#     top_count = max(1, int(top_percent * len(all_nodes)))
#     top_nodes = [node for _, node in all_nodes[:top_count]]

#     return top_nodes
