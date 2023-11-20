from tkinter import *
from tkinter.ttk import *
from tkinter import simpledialog
from tkinter import messagebox
import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class QLearningAgent:
    def __init__(self, nodes, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_values = {}
        self.nodes = nodes

    def get_q_value(self, state, action):
        return self.q_values.get((state, action), 0.0)

    def choose_action(self, state, available_actions):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)  # exploration
        else:
            q_values = [self.get_q_value(state, action) for action in available_actions]
            max_q = max(q_values)
            best_actions = [action for action, q in zip(available_actions, q_values) if q == max_q]
            return random.choice(best_actions)  # exploitation

    def update_q_value(self, state, action, reward, next_state):
        current_q = self.get_q_value(state, action)
        max_next_q = max([self.get_q_value(next_state, next_action) for next_action in self.nodes])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_values[(state, action)] = new_q

class Node:
    def __init__(self, capacity):
        self.capacity = capacity
        self.jobs = []
        self.execution_times = {}

    def add_job(self, job, execution_time):
        if self.remaining_capacity() >= 1:
            self.jobs.append(job)
            self.execution_times[job] = execution_time
            return True
        return False

    def remaining_capacity(self):
        return self.capacity - len(self.jobs)

    def get_jobs(self):
        return self.jobs

    def get_execution_time(self, job):
        return self.execution_times.get(job, 0)

class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.migrate_counter = 0
        self.migration_history = []  # Store migration history
        self.q_learning_agent = QLearningAgent(self.nodes)

    def display_migration(self, job, source_node, destination_node):
        self.migration_history.append((job, source_node, destination_node))

    def print_jobs(self):
        for node, n in self.nodes.items():
            print(f"Node: {node}, Jobs: {n.get_jobs()}, Execution Times: {n.execution_times}")

    def update_q_agent_nodes(self):
        self.q_learning_agent.nodes = list(self.nodes.keys())

    def add_node(self, node, capacity):
        self.nodes[node] = Node(capacity)
        self.edges[node] = []

    def add_edge(self, node1, node2):
        if node1 not in self.nodes or node2 not in self.nodes:
            return False
        self.edges[node1].append(node2)
        self.edges[node2].append(node1)
        self.q_learning_agent.nodes = list(self.nodes.keys())  # Update available nodes for the Q-learning agent
        return True

    def submit_job(self, job):
        random_node = random.choice(list(self.nodes.keys()))
        execution_time = self.simulate_job_execution()
        if self.nodes[random_node].add_job(job, execution_time):
            return random_node
        else:
            return self.migrate_job_with_rl(job, random_node)

    def migrate_job_with_rl(self, job, assigned_node):
        adjacent_nodes = self.edges[assigned_node]
        available_actions = [node for node in adjacent_nodes if self.nodes[node].remaining_capacity() >= 1]

        if not available_actions:
            return None

        state = assigned_node
        action = self.q_learning_agent.choose_action(state, available_actions)
        next_state = action

        execution_time = self.nodes[action].get_execution_time(job)
        if self.nodes[action].add_job(job, execution_time):
            self.migrate_counter += 1
            self.display_migration(job, assigned_node, action)
            reward = -execution_time  # Penalize for migration time
        else:
            reward = 0

        self.q_learning_agent.update_q_value(state, action, reward, next_state)
        return action

    def simulate_job_execution(self):
        return random.uniform(1, 10)

    def get_migrate_counter(self):
        return self.migrate_counter

    def visualize_graph(self):
        G = nx.Graph()
        pos = {}

        for node, data in self.nodes.items():
            G.add_node(node)
            pos[node] = (random.uniform(0, 1), random.uniform(0, 1))
            G.nodes[node]['capacity'] = data.capacity

        for node, edges in self.edges.items():
            for edge in edges:
                G.add_edge(node, edge)

        node_labels = {node: f"{node}\nCapacity: {G.nodes[node]['capacity']}" for node in G.nodes}

        nx.draw(G, pos, with_labels=True, labels=node_labels, font_weight='bold')
        plt.show()

class JobSchedulerGUI:
    def __init__(self, master):
        self.master = master
        master.title("Job Scheduler")

        width = master.winfo_screenwidth()
        height = master.winfo_screenheight()
        master.geometry(f"{width}x{height}+0+0")

        self.style =Style()
        self.style.configure('TButton', font=('Trebuchet MS', 18), padding=10)
        self.style.configure('TLabel', font=('Trebuchet MS', 20))
        self.style.configure('TEntry', font=('Trebuchet MS', 18), padding=10)
        master.configure(bg='black')
    

        self.graph = Graph()

        self.label = Label(master, text="Job Scheduler", style='TLabel',font='calibri',foreground='red')
        self.label.pack(pady=10)
        # self.node_entry = Entry(master, style='TEntry')
        # self.node_entry.pack(pady=5)
        # self.node_entry.insert(0, "Enter no. of nodes")

        self.capacity_entries = []
        self.connection_entries = []

        self.add_node_button = Button(master, text="Add Nodes", command=self.add_nodes, style='TButton')
        self.add_node_button.pack(side='top',pady=20)

        self.add_edge_button = Button(master, text="Add Edges", command=self.add_edges, style='TButton')
        self.add_edge_button.pack(side='top',pady=20)

        self.run_scheduler_button = Button(master, text="Run Scheduler", command=self.run_scheduler, style='TButton')
        self.run_scheduler_button.pack(side='top',pady=20)

        self.visualize_graph_button = Button(master, text="Visualize Graph", command=self.visualize_graph, style='TButton')
        self.visualize_graph_button.pack(side='top',pady=20)

        self.quit_button = Button(master, text="Quit", command=master.quit, style='TButton')
        self.quit_button.pack(side='top', pady=20)

    def add_nodes(self):
        num_nodes = simpledialog.askinteger("Number of Nodes", "Enter the number of nodes:", parent=self.master)
        if num_nodes is not None:
            for i in range(num_nodes):
                capacity = simpledialog.askinteger(f"Capacity of Node {i + 1}", f"Enter the capacity of node {i + 1}:", parent=self.master)
                node_name = f"N{i + 1}"
                self.graph.add_node(node_name, capacity)

    def add_edges(self):
        for i in range(len(self.graph.nodes)):
            node = "N" + str(i + 1)
            num_connections_input = simpledialog.askinteger(f"Number of connections for node {node}", f"Enter the number of connections for node {node}:", parent=self.master)
            if num_connections_input is not None:
                num_connections = int(num_connections_input)
            else:
                num_connections = 0  # or handle the case when the user cancels the input

            for _ in range(num_connections):
                connected_node = simpledialog.askstring(f"Node connected to {node}", f"Enter a node connected to {node}:", parent=self.master)
                self.graph.add_edge(node, connected_node)

    def run_scheduler(self):
        jobs = ["J14", "J1", "J2", "J3", "J4", "J5", "J13", "J8"]
        unprocessed_jobs = []

        for job in jobs:
            processed_by = self.graph.submit_job(job)
            if not processed_by:
                unprocessed_jobs.append(job)

        migrate_counter = self.graph.get_migrate_counter()

        # Show migration information
        migration_history = self.graph.migration_history
        migration_info = "\n".join([f"{job} migrated from {source} to {destination}" for job, source, destination in migration_history])
        messagebox.showinfo("Migration History", migration_info)

        # Show job information
        job_info = ""
        for node, n in self.graph.nodes.items():
            job_info += f"Node: {node}, Jobs: {n.get_jobs()}, Execution Times: {n.execution_times}\n"
        messagebox.showinfo("Job Information", job_info)

        # Show the result
        messagebox.showinfo("Scheduler Result", f"Number of times migrations: {migrate_counter}")

        if unprocessed_jobs:
            messagebox.showinfo("Unprocessed Jobs", f"Jobs couldn't be processed: {', '.join(unprocessed_jobs)}")

    def visualize_graph(self):
        self.graph.visualize_graph()

if __name__ == "__main__":
    root = Tk()
    app = JobSchedulerGUI(root)
    root.mainloop()
