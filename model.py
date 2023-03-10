import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Agent():

    def __init__(self, graph, start_node, q, max_steps, seed=None):
        '''
        Initialize the agent, giving where GRAPH is the graph it should traverse,
        and START_NODE is the node where the agent begins its job search

        Arguments:
        graph: NetworkX Graph object
        start_node: int - The index of the node in GRAPH where the agent should start
            its job search.
        '''
        self.graph = graph              # connectivity of the various locations
        self.start_node = start_node    # home node
        self.current_node = start_node  # where the job seeker is now
        self.n_steps = 0                # number of steps taken away from home node in order to find a job
        self.q = q                      # the probability that an individual job application fails
        self.max_steps = max_steps      # the number of steps the agent will take before stopping its search
        self.seed = seed                # a seed that can be used to make random actions of the agent predictable
    
    def job_search(self):
        '''
        Simulates a job search within the current node, using Agent.q as the probability
        of being rejected from each indiviudal job, and uses the "population" attribute
        of the node as the number of available jobs.

        Returns: Boolean - True if the agent finds a job, False if not.
        '''
        pop_in_current_node = int(self.graph.nodes[self.current_node]['population'])
        p_no_job = self.q ** pop_in_current_node
        return np.random.rand() > p_no_job


    def step(self):
        '''
        Simulates the process of stepping from one node to the next

        Returns: int - The index of the new node.
        '''

        connected_nodes = list(self.graph.neighbors(self.current_node))
        #print(f'current node: {self.current_node} \nconnected nodes: {connected_nodes}')
        ind = np.random.randint(len(connected_nodes))                        # pick index of one of the connected nodes
        self.current_node = connected_nodes[ind]                             # pick the connected node to go to
        self.n_steps += 1
        return self.current_node

    def reset(self):
        self.n_steps = 0
        self.current_node = self.start_node

    def simulate(self, n_iter):
        '''
        Simulates the agent's job search process N_ITER times, and returns the node of the job location,
        or -1 if no job is found.

        Arguments:
        n_iter: int - The number of times for the simulation to be run

        Returns: np.ndarray - the indices of the nodes where jobs are found, where -1 indicates that no job was found.
        '''

        def simulate_single_iter():
            while self.n_steps < self.max_steps:
                if self.job_search():
                    return self.current_node
                self.step()
            return -1

        if self.seed != None:
            np.random.seed(self.seed)
        destinations = np.array([])
        for i in range(n_iter):
            destinations = np.append(destinations, simulate_single_iter())
            self.reset()

        return destinations