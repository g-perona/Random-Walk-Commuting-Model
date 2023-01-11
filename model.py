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
            dest = simulate_single_iter()
            if dest == -1:
                continue
            destinations = np.append(destinations, dest)
            self.reset()

        return pd.DataFrame(np.array(np.unique(destinations, return_counts=True)).T, columns=['destination', 'flow'])

class Simulation():

    def __init__(self, graph, q, max_steps, n_agents, seed=None):
        '''
        Initialize the simulation, where GRAPH is the graph the agents should traverse.
        Arguments:
        graph: NetworkX Graph object - 
        q: double in (0,1) - The probability that an individual job application will fail
        max_steps: int - the max number of inter-zone steps the agents will make
        n_agents: int - gives the number of agents that will be used in the simulation
        '''
        self.graph = graph              # connectivity of the various locations
        self.q = q                      # the probability that an individual job application fails
        self.max_steps = max_steps      # the number of steps the agent will take before stopping its search
        self.n_agents = n_agents
        self.seed = seed                # a seed that can be used to make random actions of the agent predictable

        self.populations = {node: graph.nodes[node]['population'] for node in graph.nodes}
        self.tot_pop = np.sum(list(self.populations.values()))
        self.agents_per_node = {node: int(n_agents * (self.populations[node]) / self.tot_pop) for node in graph.nodes}


    def simulate(self):

        flow_data = pd.DataFrame(columns=['origin', 'destination', 'flow'])

        for i, origin in enumerate(self.graph.nodes):
            node_pop = self.populations[origin]
            n_agents = self.agents_per_node[origin]
            a = Agent(graph=self.graph, start_node=origin, q=self.q, max_steps=self.max_steps, seed=self.seed)

            dest_data = a.simulate(n_agents)
            dest_data['origin'] = np.repeat(origin, len(dest_data))
            dest_data['flow'] = dest_data['flow'].astype('int')

            flow_data = pd.concat((flow_data, dest_data), axis=0, ignore_index=True)

        return flow_data
