#code starts here
import functions_v2
import networkx as nx
import time
import matplotlib.pyplot as plt
import mplcursors
ANGLES = [20,40,60,80]
AVG_NO_OF_OBJECTS_DETECTED_PER_SATELLITE = []

start = time.time()

no_of_satellites = 50
no_of_objects = 500

satellite_data = functions_v2.create_satellite_data(no_of_satellites)
# satellite_data = functions.create_satellite_data_new(no_of_satellites)
# obj_data = functions.create_obj_data(no_of_objects, satellite_data)
for i in ANGLES:
	# obj_data = functions_v2.create_obj_data_new(no_of_objects, satellite_data, i)
	obj_data = functions_v2.create_obj_data_angle(no_of_objects, satellite_data, i)

	# plt = functions.plot_data(satellite_data, obj_data)
	G = functions_v2.create_graph(satellite_data, obj_data)
	G = functions_v2.object_detection(satellite_data, obj_data, G, plt)


	source, sink, G = functions_v2.add_source_and_sink(satellite_data, obj_data, G)
	# functions.show_updated_graph(G)
	# functions.find_max_flow(G)
	# functions.find_max_flow_with_priorities(G, len(satellite_data), obj_data)
	# functions.max_flow_with_edmonds_karp(G)
	count = functions_v2.max_flow_with_edmonds_karp_with_priorities(G, len(satellite_data), obj_data)
	AVG_NO_OF_OBJECTS_DETECTED_PER_SATELLITE.append(count)


fig = plt.figure(figsize=(12, 8))
plt.plot(ANGLES,AVG_NO_OF_OBJECTS_DETECTED_PER_SATELLITE)
plt.xticks(ANGLES)
font = {'family': 'Arial', 'size': 14}
plt.xlabel('Angle from Satellites', fontdict=font)
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


		

