import numpy as np
import random
import networkx as nx
import pandas as pd
import sys


E = "25000"
r = sys.argv[1]

for om in range(4, 8):
    # Read a state network:
    inputfile = "/Users/jelena/Desktop/regularized_higher_order_mapeq/memory_networks/memory_networks_benchmark_2/state_networks/network_om" + str(om) +"_E" + E + "_mu1_sample" + str(r) + ".net"

    vertices, vertices_print, states = [], [], {}
    links, weight = {}, []
    in_degree, in_strength, out_degree, out_strength = {}, {}, {}, {}
    total_degree, total_strength = 0, 0

    with open(inputfile, "r") as fin:
        line = fin.readline()    
        line_vertices = line
        
        line = fin.readline()
        while not (line.startswith("*States")):
            vertices.append(line.split('"')[1])
            vertices_print.append(line)
            line = fin.readline()
        line_states = line
            
        line = fin.readline()
        while not (line.startswith("*Links")):
            state_name = line.split('"')[1]
            states[line.split(' ')[0]] = state_name
            in_degree[state_name], in_strength[state_name] = 0, 0
            out_degree[state_name], out_strength[state_name] = 0, 0
            line = fin.readline()
        line_links = line
            
        line = fin.readline()
        while line:
            arc = line.split()
            n1, n2, w = arc[0], arc[1], int(arc[2])
            links[states[n1] + "\t" + states[n2]] = w
            weight.append(w)
            out_degree[states[n1]] += 1
            out_strength[states[n1]] += w
            in_degree[states[n2]] += 1
            in_strength[states[n2]] += w   
            total_degree += 2
            total_strength += 2
            line = fin.readline()

    in_mass, out_mass = {}, {}        
    for snode in states.values():
        if in_degree[snode] > 0:
            in_mass[snode] = in_strength[snode] / in_degree[snode]
        if out_degree[snode] > 0:
            out_mass[snode] = out_strength[snode] / out_degree[snode]           
    in_mass_min  = min(in_mass.values())
    out_mass_min = min(out_mass.values()) 

    num_nodes = len(vertices)
    prior_norm = np.log(num_nodes) / (2 * num_nodes * num_nodes * num_nodes) * total_degree / total_strength   
            
    outputfile = "posterior_network_infomap/network_om" + str(om) +"_E" + E + "_mu1_sample" + str(r) + ".net"
    states_prior = {}
    with open(outputfile, "w") as fout:
        
        fout.write(line_vertices)
        for line in vertices_print:
            fout.write(line)

        fout.write("*States\n")
        state_ID = 0
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                source = vertices[i]
                target = vertices[j]
                node_ID = j + 1
                state_ID += 1
                state_label = "{" + str(source) + "}_" + str(target)
                states_prior[state_label] = state_ID
                if not (state_label in in_mass):
                    in_mass[state_label] = in_mass_min
                if not (state_label in out_mass):
                    out_mass[state_label] = out_mass_min                
                fout.write(str(state_ID) + ' ' + str(node_ID) + ' "' + state_label + '"\n')
        
        fout.write("*Links\n")
        for n1 in range(num_nodes):
            for n2 in range(num_nodes):
                if n2 == n1:
                    continue
                for n3 in range(num_nodes):
                    for n4 in range(num_nodes):
                        if (n4 == n3) | (n4==n2):
                            continue
                    
                        state_source = "{" + str(vertices[n1]) + "}_" + str(vertices[n2])
                        state_target = "{" + str(vertices[n3]) + "}_" + str(vertices[n4])
                        prior_link = state_source + '\t' + state_target
                        
                        if prior_link in links:
                            w = links[prior_link] + prior_norm * out_mass[state_source] * in_mass[state_target]
                        else:
                            w = prior_norm * out_mass[state_source] * in_mass[state_target]            
                                                        
                        fout.write(str(states_prior[state_source]) + " " + str(states_prior[state_target]) + " " + str(w) + "\n")