import random
import math
from math import acos, sqrt, degrees
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import edmonds_karp
import dinic
import mplcursors
AVG_DISTANCE = [100,120,140,160,180,200,220,240,260,280,300]
global_obj = {}

def create_satellite_data(satellites):
	global global_obj;
	# print(satellites)
	satellite_data = []
	avg_x_coor, avg_y_coor = 0, 0
	for i in range(satellites):
		obj = {}
		obj['index'] = i
		obj['x_coor'] = 50-100*random.randrange(2,4)
		obj['y_coor'] = 50-100*random.random()
		obj['label'] = "S" + str(i+1)
		satellite_data.append(obj)
		avg_x_coor += obj['x_coor']
		avg_y_coor += obj['y_coor']
	global_obj['avg_x_coor'] = int(avg_x_coor/satellites)
	global_obj['avg_y_coor'] = int(avg_y_coor/satellites)
	print(global_obj)
	return satellite_data


def calculate_radius():
	global AVG_DISTANCE
	radius = (3/2)*AVG_DISTANCE
	uniform = random.random()
	radius = radius * np.sqrt(uniform)
	return radius


def create_satellite_data_new(satellites):
	# print(satellites)
	satellite_data = []
	for i in range(satellites):
		obj = {}
		obj['index'] = i
		radius = calculate_radius()
		theta = random.random()*180 - (90)
		theta_rad = math.radians(theta)
		obj['x_coor'] = radius * math.cos(theta_rad)
		obj['y_coor'] = radius * math.sin(theta_rad)
		obj['label'] = "S" + str(i+1)
		satellite_data.append(obj)

	print("satellite_data", satellite_data)
	return satellite_data

def create_obj_data_new(objects, satellite_data):

	largest_x_coor_satellite = find_largest_x_coor_satellite(satellite_data)
	
	obj_data = []
	for i in range(int(objects/2)):
		x_coor = 100*random.random()
		while x_coor < largest_x_coor_satellite:
			x_coor = 50-100*random.randrange(-20,-2)
		y_coor = random.uniform(0,50)
		obj = {
			'x_coor' : x_coor,
			'y_coor' : y_coor,
			'label' : "O" + str(i+1),
			'index' : len(satellite_data) + i
		}
		obj_data.append(obj)
	for i in range(int(objects/2), objects):
		x_coor = 100*random.random()
		while x_coor < largest_x_coor_satellite:
			x_coor = 50-100*random.randrange(-20,-2)
		y_coor = 50-100*random.random()
		# For High priorities together
		# y_coor = random.uniform(-50, -20)
		obj = {
			'x_coor' : x_coor,
			'y_coor' : y_coor,
			'label' : "O" + str(i+1),
			'index' : len(satellite_data) + i
		}
		obj_data.append(obj)
	# For High priorities together
	for i in range(int(len(obj_data)/2)):
		obj_data[i]['priority'] = random.randint(9,10)
	for i in range(int(len(obj_data)/2), len(obj_data)):
		obj_data[i]['priority'] = random.randint(1,3)

	# for i in range(len(obj_data)):
	# 	obj_data[i]['priority'] = random.randint(1,10)
	return obj_data


def find_largest_x_coor_satellite(data):
	max_coor = data[0]['x_coor']

	for i in range(len(data)):
		if max_coor < data[i]['x_coor']:
			max_coor = data[i]['x_coor']
	return max_coor

def average_upper_diagonal(arr):
    total = 0
    count = 0
    rows = len(arr)
    cols = len(arr[0])

    for i in range(rows):
        for j in range(i + 1, cols):
            total += arr[i][j]
            count += 1

    if count > 0:
        average = total / count
        return average
    else:
        return 0  # Return 0 if there are no upper diagonal values

def scale_correlation_matrix(correlation_matrix):
	matrix = np.array(correlation_matrix)
	min_value = np.min(matrix)
	max_value = np.max(matrix)
	scaled_matrix = (matrix - min_value) / (max_value - min_value)
	# print("scaled_matrix", scaled_matrix)
	return scaled_matrix


def generate_correlation_matrix(obj_data):
	n = len(obj_data)
	correlation_matrix = [[0]*n for _ in range(n)]
	for i in range(n):
		for j in range(n):
			dist = math.sqrt( (obj_data[j]['x_coor'] - obj_data[i]['x_coor'])**2 + (obj_data[j]['y_coor'] - obj_data[i]['y_coor'])**2 )
			# angle = math.atan2(obj_data[j]['y_coor'] - obj_data[i]['y_coor'], obj_data[j]['x_coor'] - obj_data[i]['x_coor'])

			# prob = dist * angle
			# prob_scaled = (prob - min_value) / (max_value - min_value)
			correlation_matrix[i][j] = dist
			# print("dist", dist, i, j, correlation_matrix)

	# print("correlation_matrix",correlation_matrix)
	avg_value = average_upper_diagonal(correlation_matrix)
	correlation_matrix = scale_correlation_matrix(correlation_matrix)
	return correlation_matrix


def create_obj_data(objects, satellite_data):

	largest_x_coor_satellite = find_largest_x_coor_satellite(satellite_data)
	
	obj_data = []
	for i in range(int(objects)):
		x_coor = 100*random.random()
		while x_coor < largest_x_coor_satellite:
			x_coor = 50-100*random.randrange(-20,-2)
		y_coor = random.uniform(0,50)
		obj = {
			'x_coor' : x_coor,
			'y_coor' : y_coor,
			'label' : "O" + str(i+1),
			'index' : len(satellite_data) + i
		}
		obj_data.append(obj)
	# for i in range(int(objects/2), objects):
	# 	x_coor = 100*random.random()
	# 	while x_coor < largest_x_coor_satellite:
	# 		x_coor = 50-100*random.randrange(-20,-2)
	# 	y_coor = 50-100*random.random()
	# 	# For High priorities together
	# 	# y_coor = random.uniform(-50, -20)
	# 	obj = {
	# 		'x_coor' : x_coor,
	# 		'y_coor' : y_coor,
	# 		'label' : "O" + str(i+1),
	# 		'index' : len(satellite_data) + i
	# 	}
	# 	obj_data.append(obj)
	# For High priorities together
	# for i in range(int(len(obj_data)/2)):
	# 	obj_data[i]['priority'] = random.randint(9,10)
	# for i in range(int(len(obj_data)/2), len(obj_data)):
	# 	obj_data[i]['priority'] = random.randint(1,3)

	for i in range(len(obj_data)):
		obj_data[i]['priority'] = random.randint(1,10)
	return obj_data

def plot_data(satellite_data, obj_data):
	satellite_x_coor = [obj["x_coor"] for obj in satellite_data]
	satellite_y_coor = [obj["y_coor"] for obj in satellite_data]

	obj_x_coor = [obj["x_coor"] for obj in obj_data]
	obj_y_coor = [obj["y_coor"] for obj in obj_data]

	plt.scatter(satellite_x_coor, satellite_y_coor,c='r')
	plt.scatter(obj_x_coor, obj_y_coor,c='g')
	for i in range(len(satellite_data)):
		plt.text(x=satellite_data[i]['x_coor'],y=satellite_data[i]['y_coor'],s=satellite_data[i]['label'],fontdict=dict(color='black',size=12))

	for i in range(len(obj_data)):
		plt.text(x=obj_data[i]['x_coor'],y=obj_data[i]['y_coor'],s=obj_data[i]['label'],fontdict=dict(color='black',size=14))

	# plt.show()
	return plt

def object_detection(satellite_data, obj_data, G, plt):
	for i in range(len(satellite_data)):
		for j in range(len(obj_data)):
			dist = math.sqrt( (obj_data[j]['x_coor'] - satellite_data[i]['x_coor'])**2 + (obj_data[j]['y_coor'] - satellite_data[i]['y_coor'])**2 )
			print("Distance of Satellite",i,"and object",j,"is",dist)

			angle = math.atan2(obj_data[j]['y_coor'] - satellite_data[i]['y_coor'], obj_data[j]['x_coor'] - satellite_data[i]['x_coor'])
			deg_angle = degrees(angle)
			print("Angle of Satellite",i,"and object",j,"is",deg_angle)

			
			if -22.5 < deg_angle < 22.5:
				print("Type 1")
				angle_rng = 45
			elif -45 < deg_angle < 45:
				print("Type 2")
				angle_rng = 90
			elif -67.5 < deg_angle < 67.5:
				print("Type 3")
				angle_rng = 135
			elif -90 < deg_angle < 90:
				print("Type 4")
				angle_rng = 180

			p_angle = (1-(2*abs(deg_angle)/angle_rng))  
			# print("Prob of angle is:",p_angle)
			#to find distance range
			# distance_rng = ((angle_rng/deg_angle)**2)*10
			distance_rng = 200*sqrt(146/angle_rng)

			print("Distance",dist, "distance_rng", distance_rng)
			# Detecting object based on Distance
			if dist > distance_rng:
				p_dis = 0
				print("Probabilty of dist detecting is:",p_dis)
			else:
				p_dis = (1-(dist/distance_rng))
				# print("Prob of dist is:",p_dis)
				p = p_angle * p_dis
				print("Probability of detecting object is:",p)
				plt.plot([satellite_data[i]['x_coor'], obj_data[j]['x_coor']], [satellite_data[i]['y_coor'], obj_data[j]['y_coor']])
				# Add edges to graph
				G.add_edge(i, j+len(satellite_data),prob = p)
			print("-----------------------------")
	# plt.show()
	return G

def create_graph(satellite_data, obj_data):
	G = nx.DiGraph()
	for i in range(len(satellite_data)):
		G.add_node(satellite_data[i]['index'], pos=(satellite_data[i]['x_coor'], satellite_data[i]['y_coor']))
		
	for i in range(len(obj_data)):
		G.add_node(obj_data[i]['index'], pos=(obj_data[i]['x_coor'], obj_data[i]['y_coor']))
	
	return G


def add_source_and_sink(satellite_data, obj_data, G):
	satellite_x_coor = [obj["x_coor"] for obj in satellite_data]
	satellite_y_coor = [obj["y_coor"] for obj in satellite_data]

	obj_x_coor = [obj["x_coor"] for obj in obj_data]
	obj_y_coor = [obj["y_coor"] for obj in obj_data]

	mean_s_y=sum(satellite_y_coor)/len(satellite_y_coor)
	mean_o_y=sum(obj_y_coor)/len(obj_y_coor)

	#Source node coordinates
	min_x_s=min(satellite_x_coor)

	delta_x = 30

	# Create the dummy node coordinate
	source_coord = (min_x_s - delta_x, mean_s_y)
	print(source_coord)

	#Destination node coordinates
	max_x_o=max(obj_x_coor)

	delta_x = 20

	# Create the dummy node coordinate
	dest_coord = (max_x_o + delta_x, mean_o_y)
	print(dest_coord)

	# add a dummy source node and connect it to all satellite nodes
	no_of_nodes = len(G.nodes())
	G.add_node(no_of_nodes,pos=source_coord)
	G.add_edges_from([(no_of_nodes, i) for i in range(len(satellite_data))])

	G.add_node(no_of_nodes+1,pos=dest_coord)
	G.add_edges_from([(j+len(satellite_data),no_of_nodes+1) for j in range(len(obj_data))])
	return source_coord, dest_coord, G

def show_updated_graph(G):
	pos = nx.get_node_attributes(G, 'pos')
	nx.draw(G, pos=pos, with_labels=True)
	plt.show()

# def assign_priorities(G, obj_data):
# 	# Assigning priorities
# 	priority_list=[]
# 	obj_nodes = [obj["index"] for obj in obj_data]
# 	print(obj_nodes)

# 	for node in range(len(G.nodes())):
# 	  if node in obj_nodes:
# 	    priority = G.nodes[node]['priority']*10
# 	    priority_list.append(priority)
# 	    print("Priority of node", node, "is", priority)
# 	  else:
# 	    priority_list.append('priority_not_assigned')
# 	print(priority_list)

# def generate_capacities(obj_data):


def assign_capacities(G, no_of_satellites, no_of_objects):
	for (i,j) in G.edges():
		print(i,j)

	obj_indexes = range(no_of_satellites, no_of_satellites + no_of_objects)
	print(obj_indexes)

	# for node in len(G.nodes())




# A Python implementation of the Ford-Fulkerson algorithm
# for finding the maximum flow in a flow network.

from collections import deque

# # Returns true if there is a path from source to sink in
# # residual graph. Also fills parent[] to store the path
# def bfs(graph, source, sink, parent):
    
#     # Mark all the vertices as not visited
#     visited = [False] * len(graph)
    
#     # Create a queue for BFS
#     queue = deque()
    
#     # Mark the source node as visited and enqueue it
#     queue.append(source)
#     visited[source] = True
    
#     # Standard BFS Loop
#     while queue:
        
#         # Dequeue a vertex from queue and print it
#         u = queue.popleft()
        
#         # Get all adjacent vertices of the dequeued vertex u
#         # If a adjacent has not been visited, then mark it
#         # visited and enqueue it
#         for ind, val in enumerate(graph[u]):
#             if visited[ind] == False and val > 0 :
#                 queue.append(ind)
#                 visited[ind] = True
#                 parent[ind] = u
    
#     # If we can reach sink in BFS starting from source then
#     # return true else false
#     return True if visited[sink] else False
 
 
# # Returns the maximum flow from source to sink in the given graph
# def ford_fulkerson(graph, source, sink):
    
#     # This array is filled by BFS and to store path
#     parent = [-1] * len(graph)
 
#     max_flow = 0  # There is no flow initially
 
#     # Augument the flow while there is path from source to sink
#     while bfs(graph, source, sink, parent) :
 
#         # Find minimum residual capacity of the edges along the
#         # path filled by BFS. Or we can say find the maximum flow
#         # through the path found.
#         path_flow = float("Inf")
#         s = sink
#         while(s != source):
#             path_flow = min(path_flow, graph[parent[s]][s])
#             satellite = s
#             s = parent[s]
 
#         # Add path flow to overall flow
#         max_flow +=  path_flow
#         print(satellite, "object detected", parent[-1], path_flow)
#         # update residual capacities of the edges and reverse edges
#         # along the path
#         v = sink
#         while(v !=  source):
#             u = parent[v]
#             graph[u][v] -= path_flow
#             graph[v][u] += path_flow
#             v = parent[v]
 
#     # return maximum flow
#     return max_flow

# def ford_fulkerson_with_priorities(graph, source, sink, no_of_satellites, obj_data):
    
#     # This array is filled by BFS and to store path
#     parent = [-1] * len(graph)
 
#     max_flow = 0  # There is no flow initially
 
#     # Augument the flow while there is path from source to sink
#     while bfs(graph, source, sink, parent) :
 
#         # Find minimum residual capacity of the edges along the
#         # path filled by BFS. Or we can say find the maximum flow
#         # through the path found.
#         path_flow = float("Inf")
#         s = sink
#         while(s != source):
#             path_flow = min(path_flow, graph[parent[s]][s])
#             satellite = s
#             s = parent[s]
 
#         # Add path flow to overall flow
#         # max_flow += path_flow
#         max_flow +=  path_flow/(obj_data[parent[-1] - no_of_satellites]['priority']*10)
#         print(satellite, "object detected", parent[-1], path_flow/(obj_data[parent[-1] - no_of_satellites]['priority']*10), " with priority ", obj_data[parent[-1] - no_of_satellites]['priority'])
#         # print("path_flow", path_flow/obj_data[parent[-1] - no_of_satellites]['priority'], obj_data[parent[-1] - no_of_satellites]['priority'])
 
#         # update residual capacities of the edges and reverse edges
#         # along the path
#         v = sink
#         while(v !=  source):
#             u = parent[v]
#             graph[u][v] -= path_flow
#             graph[v][u] += path_flow
#             v = parent[v]
 
#     # return maximum flow
#     return max_flow

# Define the Ford-Fulkerson algorithm function
def ford_fulkerson(graph, source, sink):
    """
    :param graph: the input graph in adjacency matrix format
    :param source: the source vertex
    :param sink: the sink vertex
    :return: the maximum flow from source to sink
    """

    # Create a residual graph with the same structure as the input graph
    residual_graph = [[graph[i][j] for j in range(len(graph))] for i in range(len(graph))]

    # Create an array to store the path from source to sink
    path = [-1] * len(graph)

    # Initialize the maximum flow to 0
    max_flow = 0

    # While there is a path from source to sink
    while bfs(residual_graph, source, sink, path):

        # Find the minimum residual capacity of the edges along the path
        min_capacity = float('inf')
        node = sink
        while node != source:
            prev = path[node]
            min_capacity = min(min_capacity, residual_graph[prev][node])
            satellite = node
            node = prev

        # Update the residual capacities of the edges along the path
        node = sink
        while node != source:
            prev = path[node]
            residual_graph[prev][node] -= min_capacity
            residual_graph[node][prev] += min_capacity
            node = prev

        # Add the minimum residual capacity to the maximum flow
        max_flow += min_capacity
        print(satellite, " detected ", path[-1], " with flow", min_capacity)
    # Return the maximum flow
    return max_flow


def ford_fulkerson_with_priorities(graph, source, sink, no_of_satellites, obj_data):
    """
    :param graph: the input graph in adjacency matrix format
    :param source: the source vertex
    :param sink: the sink vertex
    :return: the maximum flow from source to sink
    """

    # Create a residual graph with the same structure as the input graph
    residual_graph = [[graph[i][j] for j in range(len(graph))] for i in range(len(graph))]

    # Create an array to store the path from source to sink
    path = [-1] * len(graph)

    # Initialize the maximum flow to 0
    max_flow = 0

    # While there is a path from source to sink
    while bfs(residual_graph, source, sink, path):

        # Find the minimum residual capacity of the edges along the path
        min_capacity = float('inf')
        node = sink
        while node != source:
            prev = path[node]
            min_capacity = min(min_capacity, residual_graph[prev][node])
            satellite = node
            node = prev

        # print(node, "object detected", path[-1], path_flow/(obj_data[parent[-1] - no_of_satellites]['priority']*10), " with priority ", obj_data[parent[-1] - no_of_satellites]['priority'])
        # Update the residual capacities of the edges along the path
        node = sink
        while node != source:
            prev = path[node]
            residual_graph[prev][node] -= min_capacity
            residual_graph[node][prev] += min_capacity
            node = prev

        # Add the minimum residual capacity to the maximum flow
        max_flow += min_capacity
        print(satellite, " detected ", path[-1], " with flow", min_capacity)

    # Return the maximum flow
    return max_flow



# Define the BFS function to find a path from source to sink
def bfs(residual_graph, source, sink, path):
    """
    :param residual_graph: the residual graph
    :param source: the source vertex
    :param sink: the sink vertex
    :param path: the path from source to sink
    :return: True if there is a path from source to sink, False otherwise
    """

    # Initialize the visited array to False
    visited = [False] * len(residual_graph)

    # Initialize the queue with the source vertex
    queue = [source]

    # Mark the source vertex as visited
    visited[source] = True

    # While the queue is not empty
    while queue:

        # Dequeue a vertex from the queue
        vertex = queue.pop(0)

        # Traverse all the adjacent vertices of the dequeued vertex
        for neighbor, capacity in enumerate(residual_graph[vertex]):

            # If the residual capacity is greater than 0 and the vertex has not been visited
            if capacity > 0 and not visited[neighbor]:

                # Mark the neighbor as visited and add it to the queue
                visited[neighbor] = True
                queue.append(neighbor)

                # Update the path to the neighbor vertex
                path[neighbor] = vertex

                # If the sink vertex is found, return True
                if neighbor == sink:
                    return True

    # If the sink vertex is not found, return False
    return False



def sum_of_from_node_capacities(node, edge_prob_ss):
	capacity = 0
	for i,j,probability in edge_prob_ss:
		if j == node:
			capacity = capacity + probability
	return capacity

def find_max_flow(G):
	edge_prob_ss = [(u, v, probability) for u, v, probability in G.edges(data='prob')]
	# print("edge_capacity", edge_capacity)

	edge_capacity = []
	for i, j in G.edges():
		if i == len(G.nodes()) - 2:
			capacity = 10000
		elif j == len(G.nodes()) - 1:
			capacity = sum_of_from_node_capacities(i,edge_prob_ss)
		else:
			probability = [p for (x, y, p) in edge_prob_ss if (x == i and y == j)][0]
			capacity = probability
		edge_capacity.append((i, j, capacity))
	# print("edge_capacity", edge_capacity)
	n = len(G.nodes())


	# Create an empty adjacency matrix filled with zeros
	graph = [[0] * n for _ in range(n)]

	# Update the matrix with the capacities of the edges
	for u, v, capacity in edge_capacity:
	    graph[u][v] = capacity

	# print(graph)

	source = n-2
	sink = n-1
	max_flow = ford_fulkerson(graph, source, sink)

	# print the maximum flow
	print("Maximum flow:", max_flow)

def calculate_correlation_constants(correlation_matrix, obj_data):
	priorities = [x['priority'] for x in obj_data]
	# print("priorities", priorities)
	correlation_constants = [1]*len(obj_data)
	for i in range(len(priorities)):
		value = 1
		for j in range(len(priorities)):
			if priorities[j] > priorities[i]:
				value = value * (1 - correlation_matrix[i][j])
		correlation_constants[i] = value
	return correlation_constants




def find_max_flow_with_priorities(G, no_of_satellites, obj_data):
	edge_prob_ss = [(u, v, probability) for u, v, probability in G.edges(data='prob')]
	# print(edge_prob_ss)

	edge_capacity = []
	for i, j in G.edges():
		if i == len(G.nodes()) - 2:
			capacity = 10000
		elif j == len(G.nodes()) - 1:
			capacity = obj_data[i - no_of_satellites]['priority'] * 2
		else:
			probability = [p for (x, y, p) in edge_prob_ss if (x == i and y == j)][0]
			capacity = obj_data[j - no_of_satellites]['priority'] * 10 * probability
		edge_capacity.append((i, j, capacity))
	for i in range(len(obj_data)):
		print("object ", obj_data[i]['index'], " is having priority ", obj_data[i]['priority'])
	# print("edge_capacity", edge_capacity)

	n = len(G.nodes())


	# Create an empty adjacency matrix filled with zeros
	graph = [[0] * n for _ in range(n)]

	# Update the matrix with the capacities of the edges
	for u, v, capacity in edge_capacity:
	    graph[u][v] = capacity

	# print(graph)

	source = n-2
	sink = n-1
	max_flow = ford_fulkerson_with_priorities(graph, source, sink, no_of_satellites, obj_data)

	# print the maximum flow
	print("Maximum flow:", max_flow)

def find_max_flow_with_priorities_correlation(G, no_of_satellites, obj_data, correlation_matrix):
	edge_prob_ss = [(u, v, probability) for u, v, probability in G.edges(data='prob')]
	print(edge_prob_ss)

	correlation_constants = calculate_correlation_constants(correlation_matrix, obj_data);
	edge_capacity = []
	for i, j in G.edges():
		if i == len(G.nodes()) - 2:
			capacity = 10000
		elif j == len(G.nodes()) - 1:
			capacity = obj_data[i - no_of_satellites]['priority'] * 2 * correlation_constants[i - no_of_satellites]
		else:
			probability = [p for (x, y, p) in edge_prob_ss if (x == i and y == j)][0]
			capacity = obj_data[j - no_of_satellites]['priority'] * 10 * probability
		edge_capacity.append((i, j, capacity))
	for i in range(len(obj_data)):
		print("object ", obj_data[i]['index'], " is having priority ", obj_data[i]['priority'])
	# print("edge_capacity", edge_capacity)

	n = len(G.nodes())


	# Create an empty adjacency matrix filled with zeros
	graph = [[0] * n for _ in range(n)]

	# Update the matrix with the capacities of the edges
	for u, v, capacity in edge_capacity:
	    graph[u][v] = capacity

	# print(graph)

	source = n-2
	sink = n-1
	max_flow = ford_fulkerson_with_priorities(graph, source, sink, no_of_satellites, obj_data)

	# print the maximum flow
	print("Maximum flow:", max_flow)

def max_flow_with_edmonds_karp_with_priorities(G, no_of_satellites, obj_data):
	edge_prob_ss = [(u, v, probability) for u, v, probability in G.edges(data='prob')]
	# print(edge_prob_ss)
	edge_capacity = []
	for i, j in G.edges():
		if i == len(G.nodes()) - 2:
			capacity = 10000
		elif j == len(G.nodes()) - 1:
			capacity = obj_data[i - no_of_satellites]['priority'] * 2
		else:
			probability = [p for (x, y, p) in edge_prob_ss if (x == i and y == j)][0]
			capacity = obj_data[j - no_of_satellites]['priority'] * 10 * probability
		edge_capacity.append((i, j, capacity))
	for i in range(len(obj_data)):
		print("object ", obj_data[i]['index'], " is having priority ", obj_data[i]['priority'])
	# print("edge_capacity", edge_capacity)

	n = len(G.nodes())


	# Create an empty adjacency matrix filled with zeros
	graph = [[0] * n for _ in range(n)]

	# Update the matrix with the capacities of the edges
	for u, v, capacity in edge_capacity:
	    graph[u][v] = capacity

	# print(graph)

	source = n-2
	sink = n-1
	max_flow, paths = edmonds_karp.edmonds_karp_with_priorities(graph, source, sink)


	# print the maximum flow
	# print("Maximum flow:", max_flow, paths)
	avg_no_of_objects_detected = calculate_avg_no_of_objects_detected(paths, no_of_satellites)
	no_of_features = range(1,no_of_satellites+1)
	# plot_graph(avg_no_of_objects_detected)
	fig = plt.figure(figsize=(12, 8))
	plt.plot(no_of_features,avg_no_of_objects_detected)
	plt.xticks(no_of_features)
	font = {'family': 'Arial', 'size': 14}
	plt.xlabel('Number of Satellites', fontdict=font)
	plt.ylabel('Avg Number of Objects Detected Per Satellite', fontdict=font)
	# plt.title("Satellites vs Avg Objects Detected", fontdict=font)
	plt.xticks(rotation=90)  
	cursor = mplcursors.cursor(hover=True)
	@cursor.connect("add")
	def on_add(sel):
	    x = sel.target[0]
	    y = sel.target[1]
	    sel.annotation.set_text(f'x: {x:.2f}\ny: {y:.2f}')
	plt.show()

def max_flow_with_edmonds_karp_with_priorities_correlation(G, no_of_satellites, obj_data, correlation_matrix):
	edge_prob_ss = [(u, v, probability) for u, v, probability in G.edges(data='prob')]
	# print(edge_prob_ss)
	correlation_constants = calculate_correlation_constants(correlation_matrix, obj_data);
	edge_capacity = []
	for i, j in G.edges():
		if i == len(G.nodes()) - 2:
			capacity = 10000
		elif j == len(G.nodes()) - 1:
			capacity = obj_data[i - no_of_satellites]['priority'] * 2 * correlation_constants[i - no_of_satellites]
		else:
			probability = [p for (x, y, p) in edge_prob_ss if (x == i and y == j)][0]
			capacity = obj_data[j - no_of_satellites]['priority'] * 10 * probability
		edge_capacity.append((i, j, capacity))
	for i in range(len(obj_data)):
		print("object ", obj_data[i]['index'], " is having priority ", obj_data[i]['priority'])
	# print("edge_capacity", edge_capacity)

	n = len(G.nodes())


	# Create an empty adjacency matrix filled with zeros
	graph = [[0] * n for _ in range(n)]

	# Update the matrix with the capacities of the edges
	for u, v, capacity in edge_capacity:
	    graph[u][v] = capacity

	# print(graph)

	source = n-2
	sink = n-1
	max_flow, paths = edmonds_karp.edmonds_karp_with_priorities(graph, source, sink)


	# print the maximum flow
	print("Maximum flow:", max_flow, paths)
	avg_no_of_objects_detected = calculate_avg_no_of_objects_detected(paths, no_of_satellites)
	no_of_features = range(1,no_of_satellites+1)
	# plot_graph(avg_no_of_objects_detected)
	fig = plt.figure(figsize=(12, 8))
	plt.plot(no_of_features,avg_no_of_objects_detected)
	plt.xticks(no_of_features)
	font = {'family': 'Arial', 'size': 14}
	plt.xlabel('Number of Satellites', fontdict=font)
	plt.ylabel('Avg Number of Objects Detected Per Satellite', fontdict=font)
	# plt.title("Satellites vs Avg Objects Detected", fontdict=font)
	plt.xticks(rotation=90)  
	cursor = mplcursors.cursor(hover=True)
	@cursor.connect("add")
	def on_add(sel):
	    x = sel.target[0]
	    y = sel.target[1]
	    sel.annotation.set_text(f'x: {x:.2f}\ny: {y:.2f}')
	plt.show()

def trim_paths(paths):
	result = []
	for i in range(len(paths)):
		path = paths[i][1]
		result.append(path)
	return result

def calculate_avg_no_of_objects_detected(paths, no_of_satellites):
	paths = trim_paths(paths)
	# print(paths)

	obj = {key: 0 for key in range(no_of_satellites)}
	# print(obj)
	for i in range(len(paths)):
		obj[paths[i][0]] += 1
	# print(obj)

	result = [obj[0]]
	for i in range(1, no_of_satellites):
		value = ((result[i-1] * i) + obj[i])/(i+1)
		result.append(int(value))
	# print(result)
	return result

def max_flow_with_edmonds_karp(G):
	edge_prob_ss = [(u, v, probability) for u, v, probability in G.edges(data='prob')]
	# print("edge_capacity", edge_capacity)

	edge_capacity = []
	for i, j in G.edges():
		if i == len(G.nodes()) - 2:
			capacity = 10000
		elif j == len(G.nodes()) - 1:
			capacity = sum_of_from_node_capacities(i,edge_prob_ss)
		else:
			probability = [p for (x, y, p) in edge_prob_ss if (x == i and y == j)][0]
			capacity = probability
		edge_capacity.append((i, j, capacity))
	# print("edge_capacity", edge_capacity)
	n = len(G.nodes())


	# Create an empty adjacency matrix filled with zeros
	graph = [[0] * n for _ in range(n)]

	# Update the matrix with the capacities of the edges
	for u, v, capacity in edge_capacity:
	    graph[u][v] = capacity

	# print(graph)

	source = n-2
	sink = n-1
	max_flow = edmonds_karp.edmonds_karp(graph, source, sink)

	# print the maximum flow
	print("Maximum flow:", max_flow)




def test_ff(G1):
	edge_prob_ss = [(u, v, probability) for u, v, probability in G1.edges(data='prob')]
	# print("edge_capacity", edge_prob_ss)

	# edge_capacity = []
	# for i, j in G.edges():
	# 	if i == len(G.nodes()) - 2:
	# 		capacity = 10000
	# 	elif j == len(G.nodes()) - 1:
	# 		capacity = sum_of_from_node_capacities(i,edge_prob_ss)
	# 	else:
	# 		probability = [p for (x, y, p) in edge_prob_ss if (x == i and y == j)][0]
	# 		capacity = probability
	# 	edge_capacity.append((i, j, capacity))
	# print("edge_capacity", edge_capacity)
	n = len(G1.nodes())


	# Create an empty adjacency matrix filled with zeros
	graph = [[0] * n for _ in range(n)]

	# Update the matrix with the capacities of the edges
	for u, v, capacity in edge_prob_ss:
	    graph[u][v] = capacity

	# print(graph)

	source = n-2
	sink = n-1
	# print(source, sink)
	max_flow = ford_fulkerson(graph, source, sink)

	# print the maximum flow
	print("Maximum flow:", max_flow)



def max_flow_with_dinic_with_priorities(G, no_of_satellites, obj_data):
	edge_prob_ss = [(u, v, probability) for u, v, probability in G.edges(data='prob')]
	# print(edge_prob_ss)

	edge_capacity = []
	for i, j in G.edges():
		if i == len(G.nodes()) - 2:
			capacity = 10000
		elif j == len(G.nodes()) - 1:
			capacity = obj_data[i - no_of_satellites]['priority'] * 5
		else:
			probability = [p for (x, y, p) in edge_prob_ss if (x == i and y == j)][0]
			capacity = obj_data[j - no_of_satellites]['priority'] * 10 * probability
		edge_capacity.append((i, j, capacity))
	# for i in range(len(obj_data)):
		# print("object ", obj_data[i]['index'], " is having priority ", obj_data[i]['priority'])
	# print("edge_capacity", edge_capacity)

	n = len(G.nodes())


	# Create an empty adjacency matrix filled with zeros
	graph = [[0] * n for _ in range(n)]

	# Update the matrix with the capacities of the edges
	for u, v, capacity in edge_capacity:
	    graph[u][v] = capacity

	# print(graph)

	source = n-2
	sink = n-1
	max_flow = dinic.dinic(graph, source, sink)

	# print the maximum flow
	print("Maximum flow:", max_flow)


def max_flow_with_dinic(G):
	edge_prob_ss = [(u, v, probability) for u, v, probability in G.edges(data='prob')]
	# print("edge_capacity", edge_capacity)

	edge_capacity = []
	for i, j in G.edges():
		if i == len(G.nodes()) - 2:
			capacity = 10000
		elif j == len(G.nodes()) - 1:
			capacity = sum_of_from_node_capacities(i,edge_prob_ss)
		else:
			probability = [p for (x, y, p) in edge_prob_ss if (x == i and y == j)][0]
			capacity = probability
		edge_capacity.append((i, j, capacity))
	# print("edge_capacity", edge_capacity)
	n = len(G.nodes())


	# Create an empty adjacency matrix filled with zeros
	graph = [[0] * n for _ in range(n)]

	# Update the matrix with the capacities of the edges
	for u, v, capacity in edge_capacity:
	    graph[u][v] = capacity

	# print(graph)

	source = n-2
	sink = n-1
	max_flow = dinic.dinic(graph, source, sink)

	# print the maximum flow
	print("Maximum flow:", max_flow)


