from collections import deque

class Edge:
    def __init__(self, v, flow, capacity, rev):
        self.v = v
        self.flow = flow
        self.capacity = capacity
        self.rev = rev

def add_edge(adj, u, v, capacity):
    # Forward edge: capacity is positive
    forward = Edge(v, 0, capacity, len(adj[v]))
    
    # Backward edge: capacity is 0
    backward = Edge(u, 0, 0, len(adj[u]))
    
    adj[u].append(forward)
    adj[v].append(backward)

def bfs(adj, level, source, sink):
    level[:] = [-1] * len(adj)
    level[source] = 0
    queue = deque([source])
    
    while queue:
        u = queue.popleft()
        for edge in adj[u]:
            v = edge.v
            if level[v] < 0 and edge.flow < edge.capacity:
                level[v] = level[u] + 1
                queue.append(v)
    
    return level[sink] >= 0

def send_flow(adj, u, sink, flow, level, start):
    if u == sink:
        return flow
    
    for i in range(start[u], len(adj[u])):
        edge = adj[u][i]
        if level[edge.v] == level[u] + 1 and edge.flow < edge.capacity:
            min_flow = min(flow, edge.capacity - edge.flow)
            bottleneck = send_flow(adj, edge.v, sink, min_flow, level, start)
            
            if bottleneck > 0:
                edge.flow += bottleneck
                adj[edge.v][edge.rev].flow -= bottleneck
                return bottleneck
    
    return 0

def dinic(adj, source, sink):
    level = [-1] * len(adj)
    max_flow = 0
    
    while bfs(adj, level, source, sink):
        start = [0] * len(adj)
        while True:
            flow = send_flow(adj, source, sink, float('inf'), level, start)
            if flow == 0:
                break
            max_flow += flow
    
    return max_flow

# # Example usage
# if __name__ == '__main__':
#     # Initialize the graph
#     n = 4  # Number of nodes
#     adj = [[] for _ in range(n)]  # Adjacency list
    
#     # Add edges
#     add_edge(adj, 0, 1, 3)  # Edge from node 0 to node 1 with capacity 3
#     add_edge(adj, 0, 2, 2)  # Edge from node 0 to node 2 with capacity 2
#     add_edge(adj, 1, 2, 2)  # Edge from node 1 to node 2 with capacity 2
#     add_edge(adj, 1, 3, 3)  # Edge from node 1 to node 3 with capacity 3
#     add_edge(adj, 2, 3, 4)  # Edge from node 2 to node 3 with capacity 4
    
#     source = 0  # Source node
#     sink = 3  # Sink node
    
#     max_flow = dinic(adj, source, sink)
#     print("Maximum flow:", max_flow)
