"""OntoCalc Prototype Executor (lightweight)

This prototype implements a minimal parser for a subset of the DSL (script parsing is naive),
a simple in-memory graph model, union-find based merge (coequalizer-ish), a safe compute engine,
provenance logging, and SPARQL runner placeholder.

This is meant as a starting point and demonstration.
"""

from decimal import Decimal, getcontext
from datetime import datetime
import re
import json
import math
from collections import defaultdict, namedtuple

getcontext().prec = 28

# Simple Node and Edge models
class Node:
    def __init__(self, nid, types=None):
        self.id = nid
        self.types = set(types or [])
        self.props = defaultdict(list)
        self.provenance = []

    def add_prop(self, p, v, prov=None):
        self.props[p].append(v)
        if prov:
            self.provenance.append(prov)

class Edge:
    def __init__(self, subj, pred, obj, prov=None):
        self.subj = subj
        self.pred = pred
        self.obj = obj
        self.provenance = prov

class Graph:
    def __init__(self, name):
        self.name = name
        self.nodes = {}  # nid -> Node
        self.edges = []  # Edge list
        self.meta = {}

    def add_node(self, nid, types=None):
        if nid not in self.nodes:
            self.nodes[nid] = Node(nid, types)
        else:
            if types:
                self.nodes[nid].types.update(types)
        return self.nodes[nid]

    def add_edge(self, subj, pred, obj, prov=None):
        self.edges.append(Edge(subj, pred, obj, prov))

    def find_node(self, nid):
        return self.nodes.get(nid)

# Union-Find for equivalence classes across graphs
class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x
        if self.parent[x] == x:
            return x
        self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra = self.find(a); rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self):
        out = {}
        for k in self.parent:
            r = self.find(k)
            out.setdefault(r, []).append(k)
        return out

# Simple parser helpers (very small subset)
map_re = re.compile(r'map\s+([A-Za-z_\.]+)\s*->\s*([A-Za-z_\.]+)\s*\(confidence=(0?\.\d+|1(?:\.0+)?)\)\s*;')
merge_re = re.compile(r'merge\s+([A-Za-z_]+)\s*,\s*([A-Za-z_]+)\s+as\s+([A-Za-z_]+)\s+using\s+policy\s+([A-Za-z_\-]+)\s*;')
compute_re = re.compile(r'compute\s+([A-Za-z_\.]+)\.([A-Za-z_\.]+)\.([A-Za-z_]+)\s+using\s+formula\s+"([^"]+)"\s*;')

def safe_eval_formula(formula, env):
    # Allow only decimal-safe arithmetic and math functions: + - * / pow, min, max
    allowed_funcs = {
        'pow': pow, 'min': min, 'max': max
    }
    # Replace identifiers with Decimal(...) in formula
    expr = formula
    for k,v in env.items():
        if isinstance(v, Decimal):
            expr = re.sub(r'\b'+re.escape(k)+r'\b', f'Decimal("{str(v)}")', expr)
        else:
            expr = re.sub(r'\b'+re.escape(k)+r'\b', str(v), expr)
    # Disallow suspicious characters
    if re.search(r'[;\n\r]', expr):
        raise ValueError("Unsafe formula")
    # Evaluate in limited namespace
    allowed = {'Decimal': Decimal, '__builtins__': None}
    allowed.update(allowed_funcs)
    val = eval(expr, allowed, {})
    if isinstance(val, Decimal):
        return val
    else:
        return Decimal(str(val))

# Minimal executor
class Executor:
    def __init__(self):
        self.graphs = {}
        self.mappings = []  # tuples (A.term, B.term, confidence)
        self.provenance_log = []

    def load_graph(self, name, graph):
        self.graphs[name] = graph

    def record_mapping(self, left, right, confidence):
        self.mappings.append((left, right, float(confidence)))
        self.provenance_log.append({'op':'map_suggest', 'left':left, 'right':right, 'confidence':float(confidence), 'when':datetime.utcnow().isoformat()})

    def approve_mappings(self, threshold=0.9):
        # Approve any mapping >= threshold
        self.approved = [m for m in self.mappings if m[2] >= threshold]
        self.provenance_log.append({'op':'approve_mappings', 'threshold':threshold, 'approved_count':len(self.approved), 'when':datetime.utcnow().isoformat()})
        return self.approved

    def merge(self, left_name, right_name, out_name, policy='union-props'):
        GA = self.graphs[left_name]
        GB = self.graphs[right_name]
        uf = UnionFind()
        # mapping ids are like Schema.Term -> Schema.Term, here we expect simple names like ProductionBatch -> EmissionEntry
        for (l,r,conf) in getattr(self,'approved',[]):
            # naive: union label wrappers
            uf.union(('L',l), ('R',r))
        # include nodes from graphs into uf to ensure existence
        for nid in list(GA.nodes.keys()):
            uf.find(('L', nid))
        for nid in list(GB.nodes.keys()):
            uf.find(('R', nid))
        groups = uf.groups()
        # produce new graph
        M = Graph(out_name)
        rep_to_new = {}
        for rep, members in groups.items():
            # create representative id
            new_id = "N_" + str(len(rep_to_new)+1)
            rep_to_new[rep] = new_id
            M.add_node(new_id)
        # map nodes from GA
        def map_node(label, nid):
            rep = uf.find((label, nid))
            # find rep's new id
            for r, members in groups.items():
                if (label, nid) in members:
                    return rep_to_new[r]
            # fallback: create unique
            nid_new = "N_extra_"+nid
            M.add_node(nid_new)
            return nid_new
        # merge node props using simple policy: union-props
        for nid,node in GA.nodes.items():
            newid = map_node('L', nid)
            for p, vals in node.props.items():
                for v in vals:
                    M.nodes[newid].add_prop(p, v, prov={'from':left_name,'orig':nid})
        for nid,node in GB.nodes.items():
            newid = map_node('R', nid)
            for p, vals in node.props.items():
                for v in vals:
                    M.nodes[newid].add_prop(p, v, prov={'from':right_name,'orig':nid})
        # merge edges (remap)
        for e in GA.edges:
            s = map_node('L', e.subj); o = map_node('L', e.obj)
            M.add_edge(s, e.pred, o, prov={'from':left_name})
        for e in GB.edges:
            s = map_node('R', e.subj); o = map_node('R', e.obj)
            M.add_edge(s, e.pred, o, prov={'from':right_name})
        # store
        self.graphs[out_name] = M
        self.provenance_log.append({'op':'merge','left':left_name,'right':right_name,'out':out_name,'policy':policy,'when':datetime.utcnow().isoformat()})
        return M

    def compute(self, graph_name, node_type, node_id, prop, formula):
        G = self.graphs[graph_name]
        # find node by matching type and id (simple heuristic: node props may contain batchOf etc)
        # here we look for node whose original id mapping includes node_id
        target = None
        for nid,node in G.nodes.items():
            # check if any provenance orig equals node_id
            origins = [p['orig'] for p in node.provenance if 'orig' in p]
            if node_id in origins or nid == node_id:
                target = node
                break
        if target is None:
            # try direct lookup
            target = G.find_node(node_id)
        # prepare env from node props (flatten)
        env = {}
        for k, vals in target.props.items():
            # pick numeric first if exists
            for v in vals:
                try:
                    env[k] = Decimal(str(v))
                    break
                except Exception:
                    env[k] = v
        # Evaluate formula
        val = safe_eval_formula(formula, env)
        # store
        target.add_prop(prop, val, prov={'op':'compute','formula':formula,'inputs':env,'when':datetime.utcnow().isoformat()})
        self.provenance_log.append({'op':'compute','graph':graph_name,'node':node_id,'prop':prop,'value':str(val),'formula':formula,'when':datetime.utcnow().isoformat()})
        return val

    def export_graph_json(self, name):
        G = self.graphs[name]
        out = {'name':G.name, 'nodes':{}, 'edges':[]}
        for nid,node in G.nodes.items():
            out['nodes'][nid] = {'types':list(node.types), 'props':{k:[str(x) for x in v] for k,v in node.props.items()}, 'prov':node.provenance}
        for e in G.edges:
            out['edges'].append({'subj':e.subj,'pred':e.pred,'obj':e.obj,'prov':e.provenance})
        return out

# If run as script, run a demo
if __name__ == '__main__':
    ex = Executor()
    # create factory and emission graphs
    GA = Graph('FactoryData')
    GA.add_node('Batch_2025_11_01', types=['ProductionBatch'])
    GA.nodes['Batch_2025_11_01'].add_prop('quantity', Decimal('1000'), prov={'orig':'Batch_2025_11_01'})
    GA.nodes['Batch_2025_11_01'].add_prop('batchOf', 'WidgetX', prov={'orig':'Batch_2025_11_01'})
    GA.add_node('F1', types=['Factory'])
    GA.add_edge('F1', 'produces', 'Batch_2025_11_01')

    GB = Graph('EmissionData')
    GB.add_node('Entry_1', types=['EmissionEntry'])
    GB.nodes['Entry_1'].add_prop('activity', Decimal('1000'), prov={'orig':'Entry_1'})
    GB.nodes['Entry_1'].add_prop('emissionFactor', Decimal('0.75'), prov={'orig':'Entry_1'})

    ex.load_graph('FactoryData', GA)
    ex.load_graph('EmissionData', GB)

    # map suggestion and approve
    ex.record_mapping('ProductionBatch', 'EmissionEntry', 0.92)
    ex.record_mapping('quantity', 'activity', 0.98)
    approved = ex.approve_mappings(0.9)
    print("Approved mappings:", approved)

    # merge
    M = ex.merge('FactoryData', 'EmissionData', 'M', policy='union-props')
    print("Merged nodes:", list(M.nodes.keys()))

    # compute emissions
    val = ex.compute('M', 'EmissionEntry', 'Entry_1', 'emissions', "activity * emissionFactor")
    print("Computed emissions:", val)

    # export
    print(json.dumps(ex.export_graph_json('M'), indent=2))
