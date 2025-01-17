import enum

import networkx as nx


class NodeType(enum.Enum):
    AP = 0
    STA = 1


# main graph
G = nx.DiGraph()
G.add_nodes_from([
    (0, {'type': NodeType.AP}),
    (1, {'type': NodeType.STA}),
    (2, {'type': NodeType.STA}),
    (3, {'type': NodeType.STA}),
    (4, {'type': NodeType.STA}),
    (5, {'type': NodeType.AP}),
    (6, {'type': NodeType.STA}),
    (7, {'type': NodeType.STA}),
    (8, {'type': NodeType.STA}),
    (9, {'type': NodeType.STA}),
    (10, {'type': NodeType.AP}),
    (11, {'type': NodeType.STA}),
    (12, {'type': NodeType.STA}),
    (13, {'type': NodeType.STA}),
    (14, {'type': NodeType.STA}),
    (15, {'type': NodeType.AP}),
    (16, {'type': NodeType.STA}),
    (17, {'type': NodeType.STA}),
    (18, {'type': NodeType.STA}),
    (19, {'type': NodeType.STA})
])
G.add_edges_from([
    # active connections
    (0, 1, {'active': True}),
    (0, 2, {'active': True}),
    (0, 3, {'active': False}),
    (0, 4, {'active': True}),
    (5, 6, {'active': True}),
    (5, 7, {'active': True}),
    (5, 8, {'active': True}),
    (5, 9, {'active': False}),
    (10, 11, {'active': True}),
    (10, 12, {'active': True}),
    (10, 13, {'active': True}),
    (10, 14, {'active': True}),
    (15, 16, {'active': False}),
    (15, 17, {'active': False}),
    (15, 18, {'active': True}),
    (15, 19, {'active': True}),
    # selected connection
    (5, 7, {'selected': True}),
    # mean RSSI values between APs from beacons
    (0, 5, {'RSSI': -50.0}),
    (5, 0, {'RSSI': -50.0}),
    (0, 10, {'RSSI': -55.0}),
    (10, 0, {'RSSI': -55.0}),
    (0, 15, {'RSSI': -60.0}),
    (15, 0, {'RSSI': -60.0}),
    (5, 10, {'RSSI': -53.0}),
    (10, 5, {'RSSI': -53.0}),
    (5, 15, {'RSSI': -57.0}),
    (15, 5, {'RSSI': -57.0}),
    (10, 15, {'RSSI': -51.0}),
    (15, 10, {'RSSI': -51.0})
])

history = []

# historical example 1
history.append(nx.DiGraph())
history[0].add_nodes_from(G.nodes(data=True))
history[0].add_edges_from([
    (0, 4, {'rate': 68.8}),
    (15, 18, {'rate': 104.2})
])
history[0].nodes[0]['tx_power'] = 18.92
history[0].nodes[0]['MCS'] = 5
history[0].nodes[15]['tx_power'] = 16.52
history[0].nodes[15]['MCS'] = 8
history[0].nodes[4]['RSSI'] = -60.0
history[0].nodes[18]['RSSI'] = -50.0

# historical example 2
history.append(nx.DiGraph())
history[1].add_nodes_from(G.nodes(data=True))
history[1].add_edges_from([
    (10, 11, {'rate': 104.2}),
    (5, 7, {'rate': 143.2})
])
history[1].nodes[10]['tx_power'] = 11.95
history[1].nodes[10]['MCS'] = 8
history[1].nodes[5]['tx_power'] = 18.08
history[1].nodes[5]['MCS'] = 11
history[1].nodes[11]['RSSI'] = -53.0
history[1].nodes[7]['RSSI'] = -37.0

# historical example 3
history.append(nx.DiGraph())
history[2].add_nodes_from(G.nodes(data=True))
history[2].add_edges_from([
    (10, 13, {'rate': 114.7}),
    (5, 6, {'rate': 114.7})
])
history[2].nodes[10]['tx_power'] = 14.71
history[2].nodes[10]['MCS'] = 9
history[2].nodes[5]['tx_power'] = 14.88
history[2].nodes[5]['MCS'] = 9
history[2].nodes[13]['RSSI'] = -41.0
history[2].nodes[6]['RSSI'] = -40.0

# historical example 4
history.append(nx.DiGraph())
history[3].add_nodes_from(G.nodes(data=True))
history[3].add_edges_from([
    (0, 2, {'rate': 51.6}),
    (15, 18, {'rate': 114.7})
])
history[3].nodes[0]['tx_power'] = 16.07
history[3].nodes[0]['MCS'] = 4
history[3].nodes[15]['tx_power'] = 16.27
history[3].nodes[15]['MCS'] = 9
history[3].nodes[2]['RSSI'] = -65.0
history[3].nodes[18]['RSSI'] = -40.0

# historical example 5
history.append(nx.DiGraph())
history[4].add_nodes_from(G.nodes(data=True))
history[4].add_edges_from([
    (0, 1, {'rate': 143.2}),
])
history[4].nodes[0]['tx_power'] = 20.0
history[4].nodes[0]['MCS'] = 11
history[4].nodes[1]['RSSI'] = -30.0

# target graph
target = nx.DiGraph()
target.add_nodes_from(G.nodes(data=True))
target.add_edges_from([
    (5, 7),
    (10, 12),
])
target.nodes[5]['tx_power'] = 19.54
target.nodes[5]['MCS'] = 9
target.nodes[10]['tx_power'] = 13.64
target.nodes[10]['MCS'] = 8
