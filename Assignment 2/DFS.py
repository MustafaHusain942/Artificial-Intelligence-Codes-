from collections import defaultdict

def add_edge(graph, node, neighbor):
    graph[node].append(neighbor)

def dfs(graph, start):
    visited = set()
    stack = [start]
    result = []

    while stack:
        current = stack.pop()
        if current not in visited:
            visited.add(current)
            result.append(current)
            stack.extend(reversed(graph[current]))  # Reverse to maintain order

    return result

# Create graph using defaultdict
graph = defaultdict(list)

add_edge(graph, 'A', 'B')
add_edge(graph, 'A', 'C')
add_edge(graph, 'B', 'D')
add_edge(graph, 'B', 'E')
add_edge(graph, 'C', 'F')
add_edge(graph, 'E', 'F')

# Perform DFS
dfs_result = dfs(graph, 'A')
print(dfs_result)
