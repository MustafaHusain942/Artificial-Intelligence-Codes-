from collections import deque

class Graph:
    def __init__(self):
        self.graph = {}
    
    def add_edge(self, node, neighbor):
        if node not in self.graph:
            self.graph[node] = []
        self.graph[node].append(neighbor)
    
    def bfs(self, start):
        visited = set()
        queue = deque([start])
        result = []
        
        visited.add(start)
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            for neighbor in self.graph.get(current, []):
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)
        
        return result

graph = Graph()

graph.add_edge('A', 'B')
graph.add_edge('A', 'C')
graph.add_edge('B', 'D')
graph.add_edge('B', 'E')
graph.add_edge('C', 'F')
graph.add_edge('E', 'F')

bfs_result = graph.bfs('A')
print(bfs_result)