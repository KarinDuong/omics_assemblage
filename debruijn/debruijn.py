#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""Perform assembly based on debruijn graph."""

import argparse
import os
import sys
from pathlib import Path
from networkx import (
    DiGraph,
    all_simple_paths,
    lowest_common_ancestor,
    has_path,
    random_layout,
    draw,
    spring_layout,
)
import matplotlib
from operator import itemgetter
import random

random.seed(9001)
from random import randint
import statistics
import textwrap
import matplotlib.pyplot as plt
from typing import Iterator, Dict, List

from itertools import islice, combinations

matplotlib.use("Agg")

__author__ = "Duong Karine"
__version__ = "1.0.0"
__maintainer__ = "Duong Karine"
__email__ = "karine.duong@etu.u-paris.fr"
__status__ = "Developpement"


def isfile(path: str) -> Path:  # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file

    :raises ArgumentTypeError: If file does not exist

    :return: (Path) Path object of the input file
    """
    myfile = Path(path)
    if not myfile.is_file():
        if myfile.is_dir():
            msg = f"{myfile.name} is a directory."
        else:
            msg = f"{myfile.name} does not exist."
        raise argparse.ArgumentTypeError(msg)
    return myfile


def get_arguments():  # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description=__doc__, usage="{0} -h".format(sys.argv[0])
    )
    parser.add_argument(
        "-i", dest="fastq_file", type=isfile, required=True, help="Fastq file"
    )
    parser.add_argument(
        "-k", dest="kmer_size", type=int, default=22, help="k-mer size (default 22)"
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=Path,
        default=Path(os.curdir + os.sep + "contigs.fasta"),
        help="Output contigs in fasta file (default contigs.fasta)",
    )
    parser.add_argument(
        "-f", dest="graphimg_file", type=Path, help="Save graph as an image (png)"
    )
    return parser.parse_args()


def read_fastq(fastq_file: Path) -> Iterator[str]:
    """Extract reads from fastq files.

    :param fastq_file: (Path) Path to the fastq file.
    :return: A generator object that iterate the read sequences.
    """
    with open(fastq_file, 'r') as file_fastq:
        for lines in iter(lambda: list(islice(file_fastq, 4)), []):
            yield lines[1].strip()


def cut_kmer(read: str, kmer_size: int) -> Iterator[str]:
    """Cut read into kmers of size kmer_size.

    :param read: (str) Sequence of a read.
    :return: A generator object that provides the kmers (str) of size kmer_size.
    """
    for i in range(len(read)-kmer_size+1):
        yield read[i:i+kmer_size]
    

def build_kmer_dict(fastq_file: Path, kmer_size: int) -> Dict[str, int]:
    """Build a dictionnary object of all kmer occurrences in the fastq file

    :param fastq_file: (str) Path to the fastq file.
    :return: A dictionnary object that identify all kmer occurrences.
    """
    dict_kmer = dict()
        
    read_seq = read_fastq(fastq_file)
    for read in read_seq:
        kmer = cut_kmer(read, kmer_size)
    
        for i in kmer: 
            if i in dict_kmer:
                dict_kmer[i] += 1
            else:
                dict_kmer[i] = 1
    return dict_kmer
        

def build_graph(kmer_dict: Dict[str, int]) -> DiGraph:
    """Build the debruijn graph

    :param kmer_dict: A dictionnary object that identify all kmer occurrences.
    :return: A directed graph (nx) of all kmer substring and weight (occurrence).
    """
    digraph = DiGraph()
    for kmer, weight in kmer_dict.items():
        prefixe = kmer[:-1]
        suffixe = kmer[1:]
        digraph.add_edge(prefixe, suffixe, weight= weight) 
    return digraph


def remove_paths(
    graph: DiGraph,
    path_list: List[List[str]],
    delete_entry_node: bool,
    delete_sink_node: bool,
) -> DiGraph:
    """Remove a list of path in a graph. A path is set of connected node in
    the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    for path in path_list:
        if not delete_sink_node and not delete_entry_node: 
            graph.remove_nodes_from(path[1:-1])
        elif not delete_entry_node:
            graph.remove_nodes_from(path[1:])
        elif not delete_sink_node: 
            graph.remove_nodes_from(path[:-1])
        else:
            graph.remove_nodes_from(path)
    return graph


def select_best_path(
    graph: DiGraph,
    path_list: List[List[str]],
    path_length: List[int],
    weight_avg_list: List[float],
    delete_entry_node: bool = False,
    delete_sink_node: bool = False,
) -> DiGraph:
    """Select the best path between different paths

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param path_length_list: (list) A list of length of each path
    :param weight_avg_list: (list) A list of average weight of each path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    if statistics.stdev(weight_avg_list) != 0 and len(weight_avg_list) >= 2:
        index_weight_max = weight_avg_list.index(max(weight_avg_list))
        # path_list.pop(index_weight_max)
        del path_list[index_weight_max]
    elif statistics.stdev(path_length) != 0 and len(path_length) >= 2:
        index_lenth_max = path_length.index(max(path_length))
        # path_list.pop(index_lenth_max)
        del path_list[index_lenth_max]
    else:
        random_path = random.randint(0, len(path_list)-1)
        # path_list.pop(random_path)
        del path_list[random_path] 
    graph = remove_paths(graph, path_list, delete_entry_node, delete_sink_node)
    return graph


def path_average_weight(graph: DiGraph, path: List[str]) -> float:
    """Compute the weight of a path

    :param graph: (nx.DiGraph) A directed graph object
    :param path: (list) A path consist of a list of nodes
    :return: (float) The average weight of a path
    """
    return statistics.mean(
        [d["weight"] for (u, v, d) in graph.subgraph(path).edges(data=True)]
    )


def solve_bubble(graph: DiGraph, ancestor_node: str, descendant_node: str) -> DiGraph:
    """Explore and solve bubble issue

    :param graph: (nx.DiGraph) A directed graph object
    :param ancestor_node: (str) An upstream node in the graph
    :param descendant_node: (str) A downstream node in the graph
    :return: (nx.DiGraph) A directed graph object
    """
    all_path = list(all_simple_paths(graph, ancestor_node, descendant_node))
    avg_weight = [path_average_weight(graph, x) for x in all_path]
    lenght_path = [len(x)-1 for x in all_path]
    return select_best_path(graph, all_path, lenght_path, avg_weight)


def simplify_bubbles(graph: DiGraph) -> DiGraph:
    """Detect and explode bubbles

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    bubble = False
    for node in graph.nodes():
        liste_predecesseurs = list(graph.predecessors(node))
        if len(liste_predecesseurs) > 1:
            for pred_i_node, pred_j_node in combinations(liste_predecesseurs, 2):
                node_ancetre = lowest_common_ancestor(graph, pred_i_node, pred_j_node)
                if node_ancetre != None:
                    bubble = True
                    break
        if bubble:
            break
    if bubble:
        graph = simplify_bubbles(solve_bubble(graph, node_ancetre, node))
    return graph
            
        
def solve_entry_tips(graph: DiGraph, starting_nodes: List[str]) -> DiGraph:
    """Remove entry tips

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of starting nodes
    :return: (nx.DiGraph) A directed graph object
    """
    for node in graph.nodes():
        start_node_that_have_path_with_this_node = [i for i in starting_nodes if has_path(graph, i, node)] 
        
        if len(start_node_that_have_path_with_this_node) > 1:
            simple_path_exist_with_this_node = []
            for i in start_node_that_have_path_with_this_node:
                simple_path_exist_with_this_node = [path for entry in start_node_that_have_path_with_this_node
                                                    for path in all_simple_paths(graph, entry, node) if len(path)>1]
            
            if len(simple_path_exist_with_this_node) > 1:
                avg_weight_paths = [path_average_weight(graph, path) for path in simple_path_exist_with_this_node]
                length_path = [len(path) for path in simple_path_exist_with_this_node]

                graph = select_best_path(
                    graph, simple_path_exist_with_this_node, length_path, avg_weight_paths,
                    delete_entry_node=True, delete_sink_node=False
                )

                starting_nodes = get_starting_nodes(graph)
                return solve_entry_tips(graph, starting_nodes)
    return graph
    

def solve_out_tips(graph: DiGraph, ending_nodes: List[str]) -> DiGraph:
    """Remove out tips

    :param graph: (nx.DiGraph) A directed graph object
    :param ending_nodes: (list) A list of ending nodes
    :return: (nx.DiGraph) A directed graph object
    """
    for current_node in graph.nodes:
        # Identifier les successeurs qui mènent aux nœuds de sortie
        connected_ending_nodes = [
            successor for successor in graph.successors(current_node)
            if any(has_path(graph, successor, exit) for exit in ending_nodes) or successor in ending_nodes
        ]

        # Si plusieurs successeurs mènent à des nœuds de sortie
        if len(connected_ending_nodes) > 1:
            potential_paths = []

            # Récupérer les chemins simples entre le nœud actuel et les successeurs
            for successor in connected_ending_nodes:
                if successor in ending_nodes:
                    new_paths = [[current_node, successor]]  # Le successeur est directement un nœud de sortie
                else:
                    new_paths = [
                        path for exit in ending_nodes
                        for path in all_simple_paths(graph, current_node, exit)
                        if path[1] == successor  # Vérifier que le successeur est bien sur le chemin
                    ]

                for path in new_paths:
                    if len(path) > 1:  # Ignorer les chemins directs ou de longueur 1
                        potential_paths.append(path)

            # Si plusieurs chemins sont possibles, on simplifie en sélectionnant le meilleur
            if len(potential_paths) > 1:
                # Calculer les poids et les longueurs des chemins
                average_weights = [path_average_weight(graph, path) for path in potential_paths]
                path_lengths = [len(path) for path in potential_paths]

                # Sélectionner le meilleur chemin et éliminer les autres
                graph = select_best_path(
                    graph, potential_paths, path_lengths, average_weights,
                    delete_entry_node=False, delete_sink_node=True
                )

                # Mettre à jour la liste des nœuds de sortie après simplification
                ending_nodes = get_sink_nodes(graph)

                # Répéter le processus récursivement
                return solve_out_tips(graph, ending_nodes)

    return graph


def get_starting_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without predecessors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without predecessors
    """
    list_node = list()
    for node in graph.nodes():
        predecessor_node = graph.predecessors(node)
        if len(list(predecessor_node)) == 0:
            list_node.append(node)
    return list_node


def get_sink_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without successors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without successors
    """
    list_node = list()
    for node in graph.nodes():
        successors_node = graph.successors(node)
        if len(list(successors_node)) == 0:
            list_node.append(node)
    return list_node


def get_contigs(
    graph: DiGraph, starting_nodes: List[str], ending_nodes: List[str]
) -> List:
    """Extract the contigs from the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of nodes without predecessors
    :param ending_nodes: (list) A list of nodes without successors
    :return: (list) List of [contiguous sequence and their length]
    """
    
    list_contig = []
    
    for source in starting_nodes:
        for target in ending_nodes:
            if has_path(graph, source, target):
                for path in all_simple_paths(graph, source, target):
                    contig = path[0]
                    for node in path[1:]:
                        contig += node[-1]
                    list_contig.append((contig, len(contig)))
    return list_contig


def save_contigs(contigs_list: List[str], output_file: Path) -> None:
    """Write all contigs in fasta format

    :param contig_list: (list) List of [contiguous sequence and their length]
    :param output_file: (Path) Path to the output file
    """
    with open(output_file, 'w') as output:
        for count, contig_info in enumerate(contigs_list):
            contig, len_contig = contig_info
            output.write(f">contig_{count} len={len_contig}\n")
            output.write(textwrap.fill(contig, width=80) + "\n")


def draw_graph(graph: DiGraph, graphimg_file: Path) -> None:  # pragma: no cover
    """Draw the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param graphimg_file: (Path) Path to the output file
    """
    fig, ax = plt.subplots()
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] > 3]
    # print(elarge)
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] <= 3]
    # print(elarge)
    # Draw the graph with networkx
    # pos=nx.spring_layout(graph)
    pos = nx.random_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=6)
    nx.draw_networkx_edges(graph, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(
        graph, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )
    # nx.draw_networkx(graph, pos, node_size=10, with_labels=False)
    # save image
    plt.savefig(graphimg_file.resolve())


# ==============================================================
# Main program
# ==============================================================
def main() -> None:  # pragma: no cover
    """
    Main program function
    """
    # Get arguments
    args = get_arguments()

    # Lecture du fichier et construction du graph
    kmer_dictionnary = build_kmer_dict(args.fastq_file, args.kmer_size)
    graph = build_graph(kmer_dictionnary)
    
    # Résolution des bulles
    graph = simplify_bubbles(graph)
    
    # Résolution des pointes d’entrée et de sortie
    starting_nodes = get_starting_nodes(graph)
    graph = solve_entry_tips(graph, starting_nodes)
    
    ending_nodes = get_sink_nodes(graph)
    graph = solve_out_tips(graph, ending_nodes)
    
    # Ecriture du/des contigs 
    starting_nodes = get_starting_nodes(graph)
    ending_nodes = get_sink_nodes(graph)
    contigs = get_contigs(graph, starting_nodes, ending_nodes)
    
    save_contigs(contigs, args.output_file)
    
    # Resultat BLAST:
    # > makeblastdb -in data/eva71.fna -dbtype nucl
    # > blastn -query result/output.fasta -db data/eva71.fna
    #     Score = 13649 bits (7391),  Expect = 0.0
    #     Identities = 7391/7391 (100%), Gaps = 0/7391 (0%)
    #     Strand=Plus/Plus


    # Fonctions de dessin du graphe    
    # A decommenter si vous souhaitez visualiser un petit
    # graphe
    # Plot the graph
    # if args.graphimg_file:
    #     draw_graph(graph, args.graphimg_file)


if __name__ == "__main__":  # pragma: no cover
    main()
