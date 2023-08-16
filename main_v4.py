#code starts here
import functions_v4 as functions4
import networkx as nx
import time

start = time.time()

no_of_satellites = 50
no_of_objects = 500

satellite_data = functions4.create_satellite_data(no_of_satellites)
# satellite_data = functions4.create_satellite_data_new(no_of_satellites)
obj_data = functions4.create_obj_data(no_of_objects, satellite_data)
# obj_data = functions4.create_obj_data_new(no_of_objects, satellite_data)
correlation_matrix = functions4.generate_correlation_matrix(obj_data)

plt = functions4.plot_data(satellite_data, obj_data)
G = functions4.create_graph(satellite_data, obj_data)
G = functions4.object_detection(satellite_data, obj_data, G, plt)
print(G)

# edge_prob = [(u, v, probability) for u, v, probability in G.edges(data='prob')]
# print(edge_prob)


source, sink, G = functions4.add_source_and_sink(satellite_data, obj_data, G)
functions4.show_updated_graph(G)
# functions4.find_max_flow(G)
functions4.find_max_flow_with_priorities(G, len(satellite_data), obj_data)
functions4.find_max_flow_with_priorities_correlation(G, len(satellite_data), obj_data, correlation_matrix)
# functions4.max_flow_with_edmonds_karp(G)
functions4.max_flow_with_edmonds_karp_with_priorities(G, len(satellite_data), obj_data)
functions4.max_flow_with_edmonds_karp_with_priorities_correlation(G, len(satellite_data), obj_data, correlation_matrix)
# G = functions4.assign_capacities(G, len(satellite_data), len(obj_data))

# functions4.max_flow_with_dinic(G)
# functions4.max_flow_with_dinic_with_priorities(G, len(satellite_data), obj_data)

end = time.time()
print("Running time ", end-start)

# edge_prob_ss = [(u, v, probability) for u, v, probability in G.edges(data='prob')]
# print(edge_prob_ss)



# functions4.flow_maximization(G)

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

# functions4.test_ff(G1)


		

