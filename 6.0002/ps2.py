# 6.0002 Problem Set 2 Fall 2021
# Graph Optimization
# Name:
# Collaborators:
# Time:

#
# Finding shortest paths to drive from home to work on a road network
#

from graph import DirectedRoad, Node, RoadMap


# PROBLEM 2: Building the Road Network
#
# PROBLEM 2a: Designing your Graph
#
# What do the graph's nodes represent in this problem? What
# do the graph's edges represent? Where are the times
# represented?
#
# Write your answer below as a comment:
# Nodes = the places, each individual destination or starting point
# Edges = the road types, each line connecting the nodes
# Times = times between each connected node, along each edge 
#         (possibly times the traffic multiplier)

# PROBLEM 2b: Implementing load_map
def load_map(map_filename):
    """
    Parses the map file and constructs a road map (graph).

    Travel time and traffic multiplier should be cast to a float.

    Parameters:
        map_filename : String
            name of the map file

    Assumes:
        Each entry in the map file consists of the following format, separated by spaces:
            source_node destination_node travel_time road_type traffic_multiplier

        Note: hill road types always are uphill in the source to destination direction and
              downhill in the destination to the source direction. Downhill travel takes
              half as long as uphill travel. The travel_time represents the time to travel
              from source to destination (uphill).

        e.g.
            N0 N1 10 highway 1
        This entry would become two directed roads; one from 'N0' to 'N1' on a highway with
        a weight of 10.0, and another road from 'N1' to 'N0' on a highway using the same weight.

        e.g.
            N2 N3 7 hill 2
        This entry would become to directed roads; one from 'N2' to 'N3' on a hill road with
        a weight of 7.0, and another road from 'N3' to 'N2' on a hill road with a weight of 3.5.

    Returns:
        a directed road map representing the inputted map
    """
    file = open(map_filename, 'r')
    m = RoadMap()
    for i in file:
        #going through each line of the file
        stop = i.split()
        #insert the nodes
        if not m.contains_node(Node(stop[0])):
            m.insert_node(Node(stop[0]))
        if not m.contains_node(Node(stop[1])):
            m.insert_node(Node(stop[1]))
        
        #each road is two directions
        #floats
        time = float(stop[2])
        traffic = float(stop[4])
        #first direction
        d1 = DirectedRoad(stop[0], stop[1], time, stop[3], traffic)

        #second direction
        #if the road type is hill
        if stop[3] == 'hill':
            time = time/2.0
        d2 = DirectedRoad(stop[1], stop[0], time, stop[3], traffic)
        
        #insert roads
        m.insert_road(d1)
        m.insert_road(d2)
    return m

# PROBLEM 2c: Testing load_map
# Include the lines used to test load_map below, but comment them out after testing
#road_map = load_map("maps/test_load_map.txt")



# PROBLEM 3: Finding the Shortest Path using Optimized Search Method



# Problem 3a: Objective function
#
# What is the objective function for this problem? What are the constraints?
#
# Answer:
# Starting and ending at specific nodes, and moving along adjacent nodes
# we are finding the distance with the least time traveled.
# 

# PROBLEM 3b: Implement find_optimal_path
def find_optimal_path(roadmap, start, end, restricted_roads, has_traffic=False):
    """
    Finds the shortest path between start and end nodes on the road map,
    without using any restricted roads,
    following traffic conditions.
    Use Dijkstra's algorithm.

    Parameters:
    roadmap - RoadMap
        The graph on which to carry out the search
    start - Node
        node at which to start
    end - Node
        node at which to end
    restricted_roads - list[string]
        Road Types not allowed on path
    has_traffic - boolean
        flag to indicate whether to get shortest path during traffic or not

    Returns:
    A tuple of the form (best_path, best_time).
        The first item is the shortest path from start to end, represented by
        a list of nodes (Nodes).
        The second item is a float, the length (time traveled)
        of the best path.

    If there exists no path that satisfies constraints, then return None.
    """
    # start or end node not in roadmap
    if start not in roadmap.get_all_nodes() or end not in roadmap.get_all_nodes():
        return None
    
    #if start and end are the same
    if start == end:
        return ([start], 0)
    
    
    #all nodes which haven't been visted 
    unvisited = roadmap.get_all_nodes()
    travel_time = {node: float('inf') for node in roadmap.get_all_nodes()}
    #to go backwards and keep track of the nodes used
    predecessor = {node: None for node in roadmap.get_all_nodes()}
    #keep count of times
    travel_time[start] = 0
    #actual path from start to end
    path = []
    
    while len(unvisited) != 0:
        #least travel time node
        current = min(unvisited, key=lambda node: travel_time[node])
        if travel_time[current] == float('inf'):
            break
        if current == end:
            break
        #going through each road that isn't restricted from the current node
        for road in roadmap.get_reachable_roads_from_node(current, restricted_roads):
            #adding the new time (taking into account traffic)
            new_time = travel_time[current] + road.get_travel_time(has_traffic)
            #if time is less than, update the current and new time
            if new_time < travel_time[road.get_destination_node()]:
                travel_time[road.get_destination_node()] = new_time
                #keeps track of path
                predecessor[road.get_destination_node()] = current 
        #after done, remove the current node - mark as visited
        unvisited.remove(current)
    current = end
    #go through predecessor and add them to the path, in the correct order
    while predecessor[current] != None:
        path.insert(0, current)
        current = predecessor[current]
    if path != []:
        path.insert(0, current)
    else:
        return None
    return (path, travel_time[end])
        
    
                    
                    
     

# PROBLEM 4a: Implement optimal_path_no_traffic
def find_optimal_path_no_traffic(filename, start, end):
    """
    Finds the shortest path from start to end during conditions of no traffic.

    You must use find_optimal_path and load_map.

    Parameters:
    filename - name of the map file that contains the graph
    start - Node, node object at which to start
    end - Node, node object at which to end

    Returns:
    list of Node objects, the shortest path from start to end in normal traffic.
    If there exists no path, then return None.
    """
    road_map = load_map(filename)
    path = find_optimal_path(road_map, start, end, [], has_traffic=(False))
    return path[0]

# PROBLEM 4b: Implement optimal_path_restricted
def find_optimal_path_restricted(filename, start, end):
    """
    Finds the shortest path from start to end when local roads and hill roads cannot be used.

    You must use find_optimal_path and load_map.

    Parameters:
    filename - name of the map file that contains the graph
    start - Node, node object at which to start
    end - Node, node object at which to end

    Returns:
    list of Node objects, the shortest path from start to end given the aforementioned conditions,
    If there exists no path that satisfies constraints, then return None.
    """
    road_map = load_map(filename)
    path = find_optimal_path(road_map, start, end, ['local', 'hill'])
    return path[0]


# PROBLEM 4c: Implement optimal_path_heavy_traffic
def find_optimal_path_in_traffic_no_toll(filename, start, end):
    """
    Finds the shortest path from start to end when toll roads cannot be used and in traffic,
    i.e. when all roads' travel times are multiplied by their traffic multipliers.

    You must use find_optimal_path and load_map.

    Parameters:
    filename - name of the map file that contains the graph
    start - Node, node object at which to start
    end - Node, node object at which to end; you may assume that start != end

    Returns:
    The shortest path from start to end given the aforementioned conditions,
    represented by a list of nodes (Nodes).

    If there exists no path that satisfies the constraints, then return None.
    """
    road_map = load_map(filename)
    path = find_optimal_path(road_map, start, end, ['toll'], has_traffic=(True))
    return path[0]


if __name__ == '__main__':

    # UNCOMMENT THE FOLLOWING LINES TO DEBUG
    pass
    # rmap = load_map('./maps/small_map.txt')

    # start = Node('N0')
    # end = Node('N4')
    # restricted_roads = []

    # print(find_optimal_path(rmap, start, end, restricted_roads))
    
    tester = load_map('test_load_map.txt')
    
    