from tqdm import tqdm
import os
import pandas as pd
import pickle
import networkx as nx
import numpy as np
import argparse

def parse_options():
    parser = argparse.ArgumentParser(description='Extracting PDG step2.')
    parser.add_argument('-i', '--input', help='The dir path of input', type=str, default='./data/dots')
    parser.add_argument('-o', '--output', help='The dir path of output', type=str, default='./data/pdgs')
    args = parser.parse_args()
    return args

keywords = ["system","execve","recv","recvfrom","getenv","read","bcopy","streadd","strccpy","strcadd","fgets","strecpy","strtrns","syslog","getopt","getopt_long","getpass","getchar","getc","fgetc","vfscanf","fscanf","alloca","_alloca","scanf","wscanf","sscanf","swscanf","vscanf","vsscanf","strlen","wcslen","strtok","strtok_r","wcstok","strcat","strncat","wcscat","wcsncat","strcpy","strncpy","wcscpy","wcsncpy","memcpy","wmemcpy","stpcpy","stpncpy","wcpcpy","wcpncpy","memmove","wmemmove","memcmp","wmemcmp","memset","wmemset","gets","sprintf","vsprintf","swprintf","vswprintf","snprintf","vsnprintf","realpath","getwd","wctomb","wcrtomb","wcstombs","wcsrtombs","wcsnrtombs"]

def graph_extraction(dot):
    #graph = nx.drawing.nx_agraph.read_dot(dot)
    graph = nx.drawing.nx_pydot.read_dot(dot)
    return graph

def contain_keywords(string):
	global keywords
	for keyword in keywords:
		if string.find(keyword) >= 0:
			return keyword
	return None

def find_sensitive_lines(node_code):
    sensitive_ret = []
    for node, code in node_code.items():
        ret = contain_keywords(code)
        if ret:
            sensitive_ret.append(node)
    return sensitive_ret

def forward_subgraph(G,sub_G_path,start_node):
    for n in G.successors(start_node):
        if (start_node,n) not in sub_G_path:
            sub_G_path.append((start_node,n))
            if start_node!=n:
                forward_subgraph(G,sub_G_path,n)

def backward_subgraph(G,sub_G_path,start_node):
    for n in G.predecessors(start_node):
        if (n,start_node) not in sub_G_path:
            sub_G_path.append((n,start_node))
            if start_node != n:
                backward_subgraph(G,sub_G_path,n)

def gen_straight_subgraph(G,start_nodes):
    sub_G_path=[]
    for start_node in start_nodes:
        forward_subgraph(G,sub_G_path,start_node)
        backward_subgraph(G, sub_G_path, start_node)
    return list(set(sub_G_path))


def create_dataset(input_path,output_path):

    dataset = pd.DataFrame(columns=['id', 'code', 'label','adj'])

    vul_dir = input_path+'Vul'
    vul_files = os.listdir(vul_dir)
    vul_files=tqdm(vul_files,desc="Vul Files",dynamic_ncols=True)
    i = 0
    for file_name in vul_files:
        try:
            # print(file_name)
            file_path = os.path.join(vul_dir, file_name)
            pdg = graph_extraction(file_path)
            labels_dict = nx.get_node_attributes(pdg, 'label')
            labels_code = dict()
            for label, all_code in labels_dict.items():
                # code = all_code.split('code:')[1].split('\\n')[0]
                code = all_code[all_code.index(",") + 1:all_code.rfind(')') ].split('\\n')[0]
                code = code.replace("&quot;", '"')
                code = code.replace("&lt;", '<')
                code = code.replace("&gt;", '>')
                code = code.replace("static void", "void")
                labels_code[label] = code
            code_list=list(labels_code.values())
            adj = np.array(nx.adjacency_matrix(pdg).todense())

            sensitive_lines = find_sensitive_lines(labels_code)
            straight_subgraph_path = gen_straight_subgraph(pdg, sensitive_lines)
            straight_subgraph = nx.DiGraph()
            straight_subgraph.add_edges_from(straight_subgraph_path)
            straight_code_list = []
            straight_adj = None
            for node in straight_subgraph.nodes:
                straight_code_list.append(labels_code[node])
            if len(straight_subgraph.nodes)>0:
                straight_adj = np.array(nx.adjacency_matrix(straight_subgraph).todense())

            temp_df = pd.DataFrame({'id': [i], 'code': [code_list], 'label': [1],'adj':[adj]})
            dataset = pd.concat([dataset, temp_df], ignore_index=True)
            i += 1
            # if len(straight_code_list)>0:
            #     temp_df = pd.DataFrame({'id': [i], 'code': [straight_code_list], 'label': [1], 'adj': [straight_adj]})
            #     dataset = pd.concat([dataset, temp_df], ignore_index=True)
            #     i += 1
        except:
            print(file_name)

    datanum = i
    novul_dir = input_path+'No-Vul'
    novul_files = os.listdir(novul_dir)
    novul_files = tqdm(novul_files, desc="No Vul Files", dynamic_ncols=True)
    i = 0
    for file_name in novul_files:
        try:
            file_path = os.path.join(novul_dir, file_name)
            pdg = graph_extraction(file_path)
            labels_dict = nx.get_node_attributes(pdg, 'label')
            labels_code = dict()
            for label, all_code in labels_dict.items():
                # code = all_code.split('code:')[1].split('\\n')[0]
                code = all_code[all_code.index(",") + 1:all_code.rfind(')')].split('\\n')[0]
                code = code.replace("&quot;", '"')
                code = code.replace("&lt;", '<')
                code = code.replace("&gt;", '>')
                code = code.replace("static void", "void")
                labels_code[label] = code
            code_list = list(labels_code.values())
            adj = np.array(nx.adjacency_matrix(pdg).todense())

            sensitive_lines = find_sensitive_lines(labels_code)
            straight_subgraph_path = gen_straight_subgraph(pdg, sensitive_lines)
            straight_subgraph = nx.DiGraph()
            straight_subgraph.add_edges_from(straight_subgraph_path)
            straight_code_list = []
            straight_adj = None
            for node in straight_subgraph.nodes:
                straight_code_list.append(labels_code[node])
            if len(straight_subgraph.nodes) > 0:
                straight_adj = np.array(nx.adjacency_matrix(straight_subgraph).todense())

            temp_df = pd.DataFrame({'id': [i+datanum], 'code': [code_list], 'label': [0],'adj':[adj]})
            dataset = pd.concat([dataset, temp_df], ignore_index=True)
            i += 1
            # if len(straight_code_list)>0:
            #     temp_df = pd.DataFrame({'id': [i+datanum], 'code': [straight_code_list], 'label': [0], 'adj': [straight_adj]})
            #     dataset = pd.concat([dataset, temp_df], ignore_index=True)
            #     i += 1
        except:
            print(file_name)

    with open(output_path+'PDGs.pkl', 'wb') as f:
        pickle.dump(dataset, f)

    print("PDG dataset saved")

def main():
    args = parse_options()
    input_path = args.input
    output_path = args.output
    if input_path[-1] == '/':
        input_path = input_path
    else:
        input_path += '/'
    if output_path[-1] == '/':
        output_path = output_path
    else:
        output_path += '/'
    modify_dot_files(input_path)
    create_dataset(input_path,output_path)

if __name__ == '__main__':
    main()
    # python 3\ gen_pdg_dataset.py -i ./data/dots/ -o ./data/pdgs/
