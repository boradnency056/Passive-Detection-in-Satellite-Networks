#code starts here
import functions
import networkx as nx
import time

start = time.time()

no_of_satellites = 2
no_of_objects = 3

satellite_data = functions.create_satellite_data(no_of_satellites)
# satellite_data = functions.create_satellite_data_new(no_of_satellites)
# obj_data = functions.create_obj_data(no_of_objects, satellite_data)
obj_data = functions.create_obj_data_new(no_of_objects, satellite_data)

plt = functions.plot_data(satellite_data, obj_data)
G = functions.create_graph(satellite_data, obj_data)
G = functions.object_detection(satellite_data, obj_data, G, plt)
print(G)

# edge_prob = [(u, v, probability) for u, v, probability in G.edges(data='prob')]
# print(edge_prob)


source, sink, G = functions.add_source_and_sink(satellite_data, obj_data, G)
functions.show_updated_graph(G)
# functions.find_max_flow(G)
functions.find_max_flow_with_priorities(G, len(satellite_data), obj_data)
# functions.max_flow_with_edmonds_karp(G)
functions.max_flow_with_edmonds_karp_with_priorities(G, len(satellite_data), obj_data)
# G = functions.assign_capacities(G, len(satellite_data), len(obj_data))

# functions.max_flow_with_dinic(G)
# functions.max_flow_with_dinic_with_priorities(G, len(satellite_data), obj_data)

end = time.time()
print("Running time ", end-start)

# edge_prob_ss = [(u, v, probability) for u, v, probability in G.edges(data='prob')]
# print(edge_prob_ss)



# functions.flow_maximization(G)

# G1 = nx.DiGraph()
# G1.add_node(0)
# G1.add_node(1)
# G1.add_node(2)
# G1.add_node(3)
# G1.add_node(4)
# G1.add_node(5)

# G1.add_edge(0,2, prob = 4)
# G1.add_edge(0,3, prob = 8)
# G1.add_edge(1,3, prob = 9)
# G1.add_edge(3,2, prob = 6)
# G1.add_edge(0,1, prob = 2)
# G1.add_edge(4,0, prob = 10)
# G1.add_edge(4,1, prob = 10)
# G1.add_edge(2,5, prob = 10)
# G1.add_edge(3,5, prob = 10)

# functions.test_ff(G1)


		

