# -*- coding:utf-8 -*-
# Author: Jiaying Lu
# Date: 2019-03-03

import os
import sys
import json
import math
import tqdm

from nltk.tree import Tree
import networkx as nx
from nltk.stem import WordNetLemmatizer

from . import utils
from .utils import convert_elit_dep_to_old_version
from .dependency_parsing import construct_dependency_tree_from_parser_res, extract_noun_phrase_heuristically_recursion, generate_NPs_from_all_NP_idxes
from .dependency_parsing import extract_verbal_predicates_prefilter


LEMMATIZER = None

def load_lemmatizer():
    global LEMMATIZER
    if LEMMATIZER == None:
        LEMMATIZER = WordNetLemmatizer()
    return LEMMATIZER


def lemmatize_verb_trigger(trigger, trigger_idx, trigger_headword_idx, tokens, pos_taggers):
    """
    Lemmatization for verbs and phrasal verbs
    Args:
        trigger: string
        trigger_idx: tuple of int
        trigger_headword_idx: int
        tokens: list of string
        pos_taggers: list of string
    Returns:
        trigger_lemma: string
    """
    trigger_lemma = trigger
    lemmatizer = load_lemmatizer()
    if len(trigger_idx) == 1:
        trigger_lemma = lemmatizer.lemmatize(trigger.lower(), pos='v')
    else:
        trigger_head = tokens[trigger_headword_idx]
        if pos_taggers[trigger_headword_idx].startswith('VB'):
            trigger_head_lemma = lemmatizer.lemmatize(trigger_head.lower(), pos='v')
            trigger_lemma = trigger_lemma.replace(trigger_head, trigger_head_lemma)
    return trigger_lemma


def construct_dep_tree_structure_entity_sent_graph(elit_sent_res, tok_offset=0, merge_pronoun=True):
    """
    For each sentence, construct a graph which retain the dep tree structure, but only has all entity nodes and one root node.
    Args:
        elit_sent_res: dict
        tok_offset: int, optional
        merge_pronoun: bool, optional
    Return:
        graph_sent: nx.Graph()
    """
    graph_sent = nx.Graph()
    dep_root, dep_tok_root, dep_rel_root = construct_dependency_tree_from_parser_res(elit_sent_res['dep'], elit_sent_res['tok'], elit_sent_res['pos'], elit_sent_res['off'])
    #Tree.fromstring('%s'%(dep_root)).pretty_print()
    #Tree.fromstring('%s'%(dep_tok_root)).pretty_print()
    #Tree.fromstring('%s'%(dep_rel_root)).pretty_print()
    # Get NPs
    all_NP_idxes = {}
    all_NP_idxes = extract_noun_phrase_heuristically_recursion(elit_sent_res['dep'], elit_sent_res['tok'], elit_sent_res['pos'], dep_root, None, None, all_NP_idxes)
    NPs, NP_idxes, NP_headword_idxes = generate_NPs_from_all_NP_idxes(elit_sent_res['tok'], all_NP_idxes, elit_sent_res['off'])
    NP_headwidx_info_map = {}  # headword_idx -> (mention, (s_idx, e_idx), is_pronoun)
    for i in range(len(NPs)):
        NP_idx_sorted = sorted(NP_idxes[i])
        is_pronoun = False
        if len(NP_idxes[i]) == 1 and elit_sent_res['pos'][NP_idxes[i][0]].startswith('PRP'):
            is_pronoun = True
        NP_headwidx_info_map[NP_headword_idxes[i]] = (NPs[i], (NP_idx_sorted[0], NP_idx_sorted[-1]), is_pronoun)
    #print('all_NP_idxes', all_NP_idxes)
    #print('NP_headwidx_info_map', NP_headwidx_info_map)

    # layer traversal
    to_traversal = [(dep_root, None)]
    while to_traversal:
        node, pnode_headwidx = to_traversal.pop(0)
        node_tidx = node.label() if isinstance(node, Tree) else int(node)
        # add nodes to sent_graph
        if node == dep_root:  # root
            graph_sent.add_node('root_%d'%(tok_offset+node_tidx), node_type='sent_root', idx=[(tok_offset+node_tidx, (tok_offset+node_tidx, tok_offset+node_tidx))], mention_type='root_virtual')
            pnode_headwidx = node_tidx
        elif node_tidx in NP_headwidx_info_map:  # node is the headword node of NP
            mention = NP_headwidx_info_map[node_tidx][0].lower()
            mention_type = 'pronoun' if NP_headwidx_info_map[node_tidx][2] else ''
            mention_idx_add_offset = (NP_headwidx_info_map[node_tidx][1][0]+tok_offset , NP_headwidx_info_map[node_tidx][1][1]+tok_offset)
            # add node
            if mention not in graph_sent:
                graph_sent.add_node(mention, node_type='entity', idx=[mention_idx_add_offset], mention_type=mention_type)
            else:
                graph_sent.nodes[mention]['idx'].append(mention_idx_add_offset)
            # add edge
            pnode_mention = NP_headwidx_info_map[pnode_headwidx][0].lower() if pnode_headwidx != dep_root.label() else 'root_%d'%(tok_offset+dep_root.label())
            tree_distance = utils.calculate_node_pair_tree_distance_index_version(dep_root, pnode_headwidx, node_tidx)
            graph_sent.add_edge(pnode_mention, mention, weight=1/tree_distance)
            pnode_headwidx = node_tidx
        else:
            pass
        # add next layer nodes
        if isinstance(node, Tree):
            for child in node:
                to_traversal.append((child, pnode_headwidx))
    return graph_sent


def construct_dep_tree_structure_entity_sent_graph_with_static_emb(elit_sent_res, tok_offset=0, merge_pronoun=True):
    """
    For each sentence, construct a graph which retain the dep tree structure, but only has all entity nodes and one root node.
    Args:
        elit_sent_res: dict
        tok_offset: int, optional
        merge_pronoun: bool, optional
    Return:
        graph_sent: nx.Graph()
    """
    graph_sent = nx.Graph()
    dep_root, dep_tok_root, dep_rel_root = construct_dependency_tree_from_parser_res(elit_sent_res['dep'], elit_sent_res['tok'], elit_sent_res['pos'], elit_sent_res['off'])
    #Tree.fromstring('%s'%(dep_root)).pretty_print()
    #Tree.fromstring('%s'%(dep_tok_root)).pretty_print()
    #Tree.fromstring('%s'%(dep_rel_root)).pretty_print()
    # Get NPs
    all_NP_idxes = {}
    all_NP_idxes = extract_noun_phrase_heuristically_recursion(elit_sent_res['dep'], elit_sent_res['tok'], elit_sent_res['pos'], dep_root, None, None, all_NP_idxes)
    NPs, NP_idxes, NP_headword_idxes = generate_NPs_from_all_NP_idxes(elit_sent_res['tok'], all_NP_idxes, elit_sent_res['off'])
    NP_headwidx_info_map = {}  # headword_idx -> (mention, (s_idx, e_idx), is_pronoun)
    for i in range(len(NPs)):
        NP_idx_sorted = sorted(NP_idxes[i])
        is_pronoun = False
        if len(NP_idxes[i]) == 1 and elit_sent_res['pos'][NP_idxes[i][0]].startswith('PRP'):
            is_pronoun = True
        NP_headwidx_info_map[NP_headword_idxes[i]] = (NPs[i], (NP_idx_sorted[0], NP_idx_sorted[-1]), is_pronoun)
    #print('all_NP_idxes', all_NP_idxes)
    #print('NP_headwidx_info_map', NP_headwidx_info_map)

    # layer traversal
    to_traversal = [(dep_root, None)]
    while to_traversal:
        node, pnode_headwidx = to_traversal.pop(0)
        node_tidx = node.label() if isinstance(node, Tree) else int(node)
        # add nodes to sent_graph
        if node == dep_root:  # root
            graph_sent.add_node('root_%d'%(tok_offset+node_tidx), node_type='sent_root', idx=[(tok_offset+node_tidx, (tok_offset+node_tidx, tok_offset+node_tidx))], mention_type='root_virtual')
            pnode_headwidx = node_tidx
        elif node_tidx in NP_headwidx_info_map:  # node is the headword node of NP
            mention = NP_headwidx_info_map[node_tidx][0].lower()
            mention_type = 'pronoun' if NP_headwidx_info_map[node_tidx][2] else ''
            mention_idx_add_offset = (NP_headwidx_info_map[node_tidx][1][0]+tok_offset , NP_headwidx_info_map[node_tidx][1][1]+tok_offset)
            # add node
            if mention not in graph_sent:
                graph_sent.add_node(mention, node_type='entity', idx=[mention_idx_add_offset], mention_type=mention_type)
            else:
                graph_sent.nodes[mention]['idx'].append(mention_idx_add_offset)
            # add edge
            if pnode_headwidx != dep_root.label():
                pnode_mention = NP_headwidx_info_map[pnode_headwidx][0].lower()
                mention_in_list = elit_sent_res['tok'][NP_headwidx_info_map[node_tidx][1][0]:NP_headwidx_info_map[node_tidx][1][1]+1]
                mention_pnode_in_list = elit_sent_res['tok'][NP_headwidx_info_map[pnode_headwidx][1][0]:NP_headwidx_info_map[pnode_headwidx][1][1]+1]
                emb_cosine_sim = utils.cal_phrase_emb_cosine_sim_by_GloVe(mention_in_list, mention_pnode_in_list)
            else:
                pnode_mention = 'root_%d'%(tok_offset+dep_root.label())
                mention_in_list = elit_sent_res['tok'][NP_headwidx_info_map[node_tidx][1][0]:NP_headwidx_info_map[node_tidx][1][1]+1]
                mention_pnode_in_list = [elit_sent_res['tok'][dep_root.label()]]
                emb_cosine_sim = utils.cal_phrase_emb_cosine_sim_by_GloVe(mention_in_list, mention_pnode_in_list)
                #print('cal_cosine_sim', mention_in_list, mention_pnode_in_list, emb_cosine_sim)
            graph_sent.add_edge(pnode_mention, mention, weight=abs(emb_cosine_sim))
            pnode_headwidx = node_tidx
        else:
            pass
        # add next layer nodes
        if isinstance(node, Tree):
            for child in node:
                to_traversal.append((child, pnode_headwidx))
    return graph_sent


def construct_dep_tree_structure_entity_sent_graph_add_sibling_edges(elit_sent_res, tok_offset=0, merge_pronoun=True):
    """
    For each sentence, construct a graph which retain the dep tree structure, but only has all entity nodes and one root node.
    All nodes on the same level to the root would be linked, edge weights are tree distances.
    Args:
        elit_sent_res: dict
        tok_offset: int, optional
        merge_pronoun: bool, optional
    Return:
        graph_sent: nx.Graph()
    """
    graph_sent = nx.Graph()
    dep_root, dep_tok_root, dep_rel_root = construct_dependency_tree_from_parser_res(elit_sent_res['dep'], elit_sent_res['tok'], elit_sent_res['pos'], elit_sent_res['off'])
    #Tree.fromstring('%s'%(dep_root)).pretty_print()
    #Tree.fromstring('%s'%(dep_tok_root)).pretty_print()
    #Tree.fromstring('%s'%(dep_rel_root)).pretty_print()
    # Get NPs
    all_NP_idxes = {}
    all_NP_idxes = extract_noun_phrase_heuristically_recursion(elit_sent_res['dep'], elit_sent_res['tok'], elit_sent_res['pos'], dep_root, None, None, all_NP_idxes)
    NPs, NP_idxes, NP_headword_idxes = generate_NPs_from_all_NP_idxes(elit_sent_res['tok'], all_NP_idxes, elit_sent_res['off'])
    NP_headwidx_info_map = {}  # headword_idx -> (mention, (s_idx, e_idx), is_pronoun)
    for i in range(len(NPs)):
        NP_idx_sorted = sorted(NP_idxes[i])
        is_pronoun = False
        if len(NP_idxes[i]) == 1 and elit_sent_res['pos'][NP_idxes[i][0]].startswith('PRP'):
            is_pronoun = True
        NP_headwidx_info_map[NP_headword_idxes[i]] = (NPs[i], (NP_idx_sorted[0], NP_idx_sorted[-1]), is_pronoun)
    #print('all_NP_idxes', all_NP_idxes)
    #print('NP_headwidx_info_map', NP_headwidx_info_map)

    # layer traversal to construct the tree structure graph
    to_traversal = [(dep_root, None)]
    layer_wise_mentions = []   # idx#0: children of root, and so on
    headword_idx_layer_map = {dep_root.label():-1}
    while to_traversal:
        node, pnode_headwidx = to_traversal.pop(0)
        node_tidx = node.label() if isinstance(node, Tree) else int(node)
        # add nodes to sent_graph
        if node == dep_root:  # root
            graph_sent.add_node('root_%d'%(tok_offset+node_tidx), node_type='sent_root', idx=[(tok_offset+node_tidx, (tok_offset+node_tidx, tok_offset+node_tidx))], mention_type='root_virtual')
            pnode_headwidx = node_tidx
        elif node_tidx in NP_headwidx_info_map:  # node is the headword node of NP
            mention = NP_headwidx_info_map[node_tidx][0].lower()
            mention_type = 'pronoun' if NP_headwidx_info_map[node_tidx][2] else ''
            mention_idx_add_offset = (NP_headwidx_info_map[node_tidx][1][0]+tok_offset , NP_headwidx_info_map[node_tidx][1][1]+tok_offset)
            # add node
            if mention not in graph_sent:
                graph_sent.add_node(mention, node_type='entity', idx=[mention_idx_add_offset], mention_type=mention_type)
            else:
                graph_sent.nodes[mention]['idx'].append(mention_idx_add_offset)
            # add edge
            pnode_mention = NP_headwidx_info_map[pnode_headwidx][0].lower() if pnode_headwidx != dep_root.label() else 'root_%d'%(tok_offset+dep_root.label())
            headword_idx_layer_map[node_tidx] = headword_idx_layer_map[pnode_headwidx] + 1
            tree_distance = utils.calculate_node_pair_tree_distance_index_version(dep_root, pnode_headwidx, node_tidx)
            if (pnode_mention, mention) not in graph_sent.edges():
                graph_sent.add_edge(pnode_mention, mention, weight=1/tree_distance)
            else:
                graph_sent.edges[(pnode_mention, mention)]['weight'] += (1/tree_distance)
            ## add mention to layer_wise_mentions
            if len(layer_wise_mentions) < headword_idx_layer_map[node_tidx] + 1:
                layer_wise_mentions.append([])
            layer_wise_mentions[headword_idx_layer_map[node_tidx]].append((mention, node_tidx))
            ## End add mention to layer_wise_mentions
            pnode_headwidx = node_tidx
        else:
            pass
        # add next layer nodes
        if isinstance(node, Tree):
            for child in node:
                to_traversal.append((child, pnode_headwidx))

    # add edges for siblings
    for mentions in layer_wise_mentions:
        for i in range(len(mentions)):
            for j in range(i+1, len(mentions)): 
                mention_i, headwidx_i = mentions[i]
                mention_j, headwidx_j = mentions[j]
                tree_distance = utils.calculate_node_pair_tree_distance_index_version(dep_root, headwidx_i, headwidx_j)
                if (mention_i, mention_j) not in graph_sent.edges():
                    graph_sent.add_edge(mention_i, mention_j, weight=1/tree_distance)
                else:
                    graph_sent.edges[(mention_i, mention_j)]['weight'] += (1/tree_distance)
    return graph_sent


def construct_dep_tree_structure_entity_doc_graph(elit_doc_res, merge_pronoun=True, use_emb_distance=False, add_sibling_edges=False, inter_sent_edge_weight=0.5):
    """
    Args:
        elit_doc_res: dict
        merge_pronoun: bool, optional
        use_emb_distance: bool, optional
        add_sibling_edges: bool, optional
        inter_sent_edge_weight: float, optional
    Return:
        graph_doc: nx.Graph()
    """
    graph_doc = nx.Graph()
    tok_offset = 0
    last_sent_root_mention = ''
    last_sent_tok_list = []
    for elit_para_res in elit_doc_res:
        for elit_sent_res in elit_para_res['sens']:
            if len(elit_sent_res['dep']) <= 1:
                tok_offset += len(elit_sent_res['tok'])
                continue
            elit_sent_res['dep'] = convert_elit_dep_to_old_version(elit_sent_res['dep'])
            cur_sent_tok_list = elit_sent_res['tok']
            if use_emb_distance == True:
                graph_sent = construct_dep_tree_structure_entity_sent_graph_with_static_emb(elit_sent_res, tok_offset, merge_pronoun)
            elif add_sibling_edges == True:
                graph_sent = construct_dep_tree_structure_entity_sent_graph_add_sibling_edges(elit_sent_res, tok_offset, merge_pronoun)
            else:
                graph_sent = construct_dep_tree_structure_entity_sent_graph(elit_sent_res, tok_offset, merge_pronoun)
            #print('tok', elit_sent_res['tok'])
            #print('dep', elit_sent_res['dep'])
            #print('Done')
            tok_offset += len(elit_sent_res['tok'])
            # merge subgraphs
            for n_mention in graph_sent:
                n_dict = graph_sent.nodes[n_mention]
                if n_mention not in graph_doc:
                    graph_doc.add_node(n_mention, node_type=n_dict['node_type'], idx=n_dict['idx'], mention_type=n_dict['mention_type'])
                else:
                    graph_doc.nodes[n_mention]['idx'].extend(n_dict['idx'])
            for edge_mention_tuple in graph_sent.edges():
                edge_weight = graph_sent.edges[edge_mention_tuple]['weight']
                if edge_mention_tuple not in graph_doc.edges():
                    graph_doc.add_edge(edge_mention_tuple[0], edge_mention_tuple[1], weight=edge_weight)
                else:
                    graph_doc.edges[edge_mention_tuple]['weight'] += edge_weight
            # add edges between sent_root
            cur_sent_root_mention = ''
            for n_mention in graph_sent:
                if graph_sent.nodes[n_mention]['node_type'] == 'sent_root':
                    cur_sent_root_mention = n_mention
                    break
            if last_sent_root_mention != '' and cur_sent_root_mention != '':
                if not use_emb_distance:
                    graph_doc.add_edge(last_sent_root_mention, cur_sent_root_mention, weight=inter_sent_edge_weight)  #TODO: other way for inter- sent edge weight
                else:
                    print('hit L323 emb')
                    sent_cosine_sim = utils.cal_phrase_emb_cosine_sim_by_GloVe(last_sent_tok_list, cur_sent_tok_list)
                    graph_doc.add_edge(last_sent_root_mention, cur_sent_root_mention, weight=abs(sent_cosine_sim))
            last_sent_root_mention = cur_sent_root_mention
            last_sent_tok_list = cur_sent_tok_list
    return graph_doc


def construct_dep_tree_structure_entity_event_sent_graph(elit_sent_res, tok_offset=0, merge_pronoun=False, merge_vague_verb=False):
    """
    For each sentence, construct a graph which retain the dep tree structure. All entity and event nodes would be included. Default setting NOT merge pronouns and vague verbs.
    Args:
        elit_sent_res: dict
        tok_offset: int, optional
        merge_pronoun: bool, optional
        merge_vague_verb: bool, optional
    Returns:
        graph_sent: nx.Graph()
        sent_root_mention: string
    """
    graph_sent = nx.Graph()
    dep_root, dep_tok_root, dep_rel_root = construct_dependency_tree_from_parser_res(elit_sent_res['dep'], elit_sent_res['tok'], elit_sent_res['pos'], elit_sent_res['off'])
    #Tree.fromstring('%s'%(dep_root)).pretty_print()
    #Tree.fromstring('%s'%(dep_tok_root)).pretty_print()
    #Tree.fromstring('%s'%(dep_rel_root)).pretty_print()
    # Get NPs
    all_NP_idxes = {}
    all_NP_idxes = extract_noun_phrase_heuristically_recursion(elit_sent_res['dep'], elit_sent_res['tok'], elit_sent_res['pos'], dep_root, None, None, all_NP_idxes)
    NPs, NP_idxes, NP_headword_idxes = generate_NPs_from_all_NP_idxes(elit_sent_res['tok'], all_NP_idxes, elit_sent_res['off'])
    valid_node_info_map = {}   # headword_idx -> (mention, (s_idx, e_idx), (node_type, mention_type))
    for i in range(len(NPs)):
        NP_idx_sorted = sorted(NP_idxes[i])
        is_pronoun = False
        if len(NP_idxes[i]) == 1 and elit_sent_res['pos'][NP_idxes[i][0]].startswith('PRP'):
            is_pronoun = True
        mention = NPs[i].lower()
        if not merge_pronoun and is_pronoun:
            mention = '%s_%d' % (NPs[i].lower(), NP_headword_idxes[i]+tok_offset)
        valid_node_info_map[NP_headword_idxes[i]] = (mention, (NP_idx_sorted[0], NP_idx_sorted[-1]), ('entity', 'prounoun' if is_pronoun else ''))
    #print('all_NP_idxes', all_NP_idxes)
    #print('valid_node_info_map', valid_node_info_map)
    # Get Verbs
    all_verb_idxes = extract_verbal_predicates_prefilter(elit_sent_res['dep'], elit_sent_res['tok'], elit_sent_res['pos'], do_prefilter=False)
    triggers, trigger_idxes, trigger_headword_idxes = generate_NPs_from_all_NP_idxes(elit_sent_res['tok'], all_verb_idxes, elit_sent_res['off'])
    verb_headwidx_info_map = {}
    for i in range(len(triggers)):
        trigger_idx_sorted = sorted(trigger_idxes[i])
        # "'s" -> 'is'
        trigger = triggers[i] if triggers[i] != "'s" else "is"
        trigger_lemma = lemmatize_verb_trigger(trigger, trigger_idxes[i], trigger_headword_idxes[i], elit_sent_res['tok'], elit_sent_res['pos'])
        is_vague = trigger_lemma in utils.VAGUE_VERBS
        mention = trigger_lemma
        if not merge_vague_verb and is_vague:
            mention = '%s_%d' % (trigger_lemma, trigger_headword_idxes[i]+tok_offset)
        valid_node_info_map[trigger_headword_idxes[i]] = (mention, (trigger_idx_sorted[0], trigger_idx_sorted[-1]), ('event', 'vague_verb' if is_vague else ''))
    #print('all_verb_idxes', all_verb_idxes)
    # deal with non-verb root
    tidx_root = dep_root.label()
    if tidx_root not in valid_node_info_map:  # may contain two roots
        sent_root_mention = 'root_%d'%(tok_offset+tidx_root)
        valid_node_info_map[tidx_root] =(sent_root_mention, (tidx_root, tidx_root), ('sent_root', 'root_virtual'))
    else:
        sent_root_mention = valid_node_info_map[tidx_root][0]
    #print('after deal with non-verb root')
    #print('valid_node_info_map', valid_node_info_map)

    # layer traversal
    to_traversal = [(dep_root, None)]   # (cnode, pnode_headwidx)
    while to_traversal:
        node, pnode_headwidx = to_traversal.pop(0)
        node_tidx = node.label() if isinstance(node, Tree) else int(node)
        # add nodes to sent_graph
        if node_tidx in valid_node_info_map:
            mention = valid_node_info_map[node_tidx][0]
            node_type = valid_node_info_map[node_tidx][2][0]
            mention_type = valid_node_info_map[node_tidx][2][1]
            mention_idx_add_offset = (valid_node_info_map[node_tidx][1][0]+tok_offset, valid_node_info_map[node_tidx][1][1]+tok_offset)
            # add node
            if mention not in graph_sent:
                graph_sent.add_node(mention, node_type=node_type, idx=[mention_idx_add_offset], mention_type=mention_type)
            else:
                graph_sent.nodes[mention]['idx'].append(mention_idx_add_offset)
            # add edge
            #if elit_sent_res['dep'][node_tidx][1] != 'root' and elit_sent_res['dep'][node_tidx][0] != len(elit_sent_res['tok']):  # dep tree with multi heads
            if node_tidx != tidx_root:
                pnode_mention = valid_node_info_map[pnode_headwidx][0]
                tree_distance = utils.calculate_node_pair_tree_distance_index_version(dep_root, pnode_headwidx, node_tidx)
                graph_sent.add_edge(pnode_mention, mention, weight=1/tree_distance)
            # update pnode_headwidx
            pnode_headwidx = node_tidx
        else:
            pass
        # add next layer nodes
        if isinstance(node, Tree):
            for child in node:
                to_traversal.append((child, pnode_headwidx))
    return graph_sent, sent_root_mention


def construct_dep_tree_structure_entity_event_doc_graph(elit_doc_res, merge_pronoun=False, merge_vague_verb=False):
    """
    Args:
        elit_doc_res: dict
        merge_pronoun: bool, optional
        merge_vague_verb: bool, optional
    Return:
        graph_doc: nx.Graph()
    """
    graph_doc = nx.Graph()
    tok_offset = 0
    last_sent_root_mention = ''
    for elit_para_res in elit_doc_res:
        for elit_sent_res in elit_para_res['sens']:
            if len(elit_sent_res['dep']) <= 1:
                tok_offset += len(elit_sent_res['tok'])
                continue
            elit_sent_res['dep'] = convert_elit_dep_to_old_version(elit_sent_res['dep'])
            graph_sent, cur_sent_root_mention = construct_dep_tree_structure_entity_event_sent_graph(elit_sent_res, tok_offset, merge_pronoun, merge_vague_verb)
            #print('tok', elit_sent_res['tok'])
            #print('dep', elit_sent_res['dep'])
            #print('Done')
            tok_offset += len(elit_sent_res['tok'])
            # merge subgraphs
            for n_mention in graph_sent:
                n_dict = graph_sent.nodes[n_mention]
                if n_mention not in graph_doc:
                    graph_doc.add_node(n_mention, node_type=n_dict['node_type'], idx=n_dict['idx'], mention_type=n_dict['mention_type'])
                else:
                    graph_doc.nodes[n_mention]['idx'].extend(n_dict['idx'])
            for edge_mention_tuple in graph_sent.edges():
                edge_weight = graph_sent.edges[edge_mention_tuple]['weight']
                if edge_mention_tuple not in graph_doc.edges():
                    graph_doc.add_edge(edge_mention_tuple[0], edge_mention_tuple[1], weight=edge_weight)
                else:
                    graph_doc.edges[edge_mention_tuple]['weight'] += edge_weight
            # add edges between sent_root
            if last_sent_root_mention != '' and cur_sent_root_mention != '':
                graph_doc.add_edge(last_sent_root_mention, cur_sent_root_mention, weight=1/2.0)  #TODO: other way for inter- sent edge weight
            last_sent_root_mention = cur_sent_root_mention
    return graph_doc


def generate_entity_graph_dep_tree_structure_post_filter(elit_res_dir, output_file_path, add_coref=(False, ''), include_headline=False, merge_pronoun=True, use_emb_distance=False, add_sibling_edges=False):
    """
    Entity node graph while still retains dep tree structure. Could insert coref clusters.
    Args:
        elit_res_dir: str
        output_file_path: str
        add_coref: (boolean-flag, str-coref_dir_path), optional
        include_headline: boolean, optional
        merge_pronoun: boolean, optional
        use_emb_distance: boolean, optional
        add_sibling_edges: boolean, optional
    """
    fwrite = open(output_file_path, 'w')
    for year_file in os.listdir(elit_res_dir):
        print('Now process year-%s...' % (year_file))
        # load preprocessed coref clusters
        if add_coref[0] == True:
            coref_cluster_dir_path = add_coref[1]
            coref_cluster_dict = {}  # docno -> 2d list of tidx
            with open(os.path.join(coref_cluster_dir_path, '%slines.output'%(year_file))) as fopen:
                for line in fopen:
                    line_dict = json.loads(line.strip())
                    coref_cluster_dict[line_dict['docno']] = line_dict['predicted_clusters']
            print('load preprocessed coreference clusters DONE')
        # produce graph per doc
        with open(os.path.join(elit_res_dir, year_file)) as fopen:
            for line in tqdm.tqdm(fopen):
                line_res = json.loads(line.strip())
                docno = line_res['doc_id']
                if not include_headline:
                    paragraphs = line_res['elit_res'][1:]   # headline is in para#0
                else:
                    paragraphs = line_res['elit_res']
                # Build doc graph
                G = construct_dep_tree_structure_entity_doc_graph(paragraphs, merge_pronoun=merge_pronoun, use_emb_distance=use_emb_distance, add_sibling_edges=add_sibling_edges)
                ## add coref cluster node
                if add_coref[0] == True:
                    ## add edge weights between coref nodes
                    coref_nodes = {}  # mention -> [(headword_pos, (idx_s, idx_e)), ()]
                    coref_idxt_mention_map_temp = {}  # idx_tuple -> mention
                    for coref_cluster in coref_cluster_dict[docno]:  # add nodes first
                        for mention_idx_t in coref_cluster:
                            mention = utils.get_mention_by_document_level_idx_tuple(paragraphs, mention_idx_t).lower()
                            coref_idxt_mention_map_temp[tuple(mention_idx_t)] = mention
                            if mention not in G.nodes:
                                if mention not in coref_nodes:
                                    coref_nodes[mention] = []
                                coref_nodes[mention].append((mention_idx_t[0], mention_idx_t))
                    for mention, idx_list in coref_nodes.items():
                        G.add_node(mention, node_type='coref', idx=idx_list, mention_type='')
                    for coref_cluster in coref_cluster_dict[docno]:
                        for i in range(len(coref_cluster)-1):   # connect two adjacent mentions in cluster
                            j = i + 1
                            mention_i_idx_t, mention_j_idx_t = coref_cluster[i], coref_cluster[j]
                            mention_i = coref_idxt_mention_map_temp[tuple(mention_i_idx_t)]
                            mention_j = coref_idxt_mention_map_temp[tuple(mention_j_idx_t)]
                            edge_weight_pre = G.edges.get((mention_i, mention_j), {'weight': 0.0})['weight']
                            #print('i:', mention_i_idx_t, mention_i, ' j:', mention_j_idx_t, mention_j)
                            #print('edge_weight_pre', edge_weight_pre)
                            edge_weight_to_add = math.e ** (2 / (mention_i_idx_t[0] + mention_j_idx_t[1]))
                            G.add_edge(mention_i, mention_j, weight=edge_weight_pre + edge_weight_to_add)
                ## rank graph
                try:
                    graph_pr = nx.pagerank(G, alpha=0.85, max_iter=500)
                except nx.exception.PowerIterationFailedConvergence:
                    print('[nx PowerIterationFailedConvergence] docno=%s' % (docno))
                    continue
                graph_pr_sorted = sorted(graph_pr.items(), key=lambda _: _[1], reverse=True)
                ## collect top entities
                top10_entities = []
                for mention, score in graph_pr_sorted:
                    if G.nodes[mention]['node_type'] != 'entity':
                        continue
                    elif G.nodes[mention]['mention_type'] == 'pronoun':
                        continue
                    top10_entities.append(mention)
                    if len(top10_entities) >= 10:
                        break
                fwrite.write(json.dumps({'docno': docno, 'top10_entities': top10_entities}) + '\n')
    fwrite.close()


def generate_entity_event_graph_dep_tree_structure_post_filter(elit_res_dir, output_file_path, add_coref=(False, ''), include_headline=False, merge_pronoun=False, merge_vague_verb=False):
    """
    Entity node graph while still retains dep tree structure. Could insert coref clusters.
    Args:
        elit_res_dir: str
        output_file_path: str
        add_coref: (boolean-flag, str-coref_dir_path), optional
        include_headline: boolean, optional
        merge_pronoun: boolean, optional
        merge_vague_verb: boolean, optional
    """
    fwrite = open(output_file_path, 'w')
    for year_file in os.listdir(elit_res_dir):
        print('Now process year-%s...' % (year_file))
        # load preprocessed coref clusters
        if add_coref[0] == True:
            coref_cluster_dir_path = add_coref[1]
            coref_cluster_dict = {}  # docno -> 2d list of tidx
            with open(os.path.join(coref_cluster_dir_path, '%slines.output'%(year_file))) as fopen:
                for line in fopen:
                    line_dict = json.loads(line.strip())
                    coref_cluster_dict[line_dict['docno']] = line_dict['predicted_clusters']
            print('load preprocessed coreference clusters DONE')
        # produce graph per doc
        with open(os.path.join(elit_res_dir, year_file)) as fopen:
            for line in tqdm.tqdm(fopen):
                line_res = json.loads(line.strip())
                docno = line_res['doc_id']
                if not include_headline:
                    paragraphs = line_res['elit_res'][1:]   # headline is in para#0
                else:
                    paragraphs = line_res['elit_res']
                # Build doc graph
                G = construct_dep_tree_structure_entity_event_doc_graph(paragraphs, merge_pronoun, merge_vague_verb)
                ## add coref cluster node
                if add_coref[0] == True:
                    ## add edge weights between coref nodes
                    coref_nodes = {}  # mention -> [(headword_pos, (idx_s, idx_e)), ()]
                    coref_idxt_mention_map_temp = {}  # idx_tuple -> mention
                    for coref_cluster in coref_cluster_dict[docno]:  # add nodes first
                        for mention_idx_t in coref_cluster:
                            mention = utils.get_mention_by_document_level_idx_tuple(paragraphs, mention_idx_t).lower()
                            # mention need to be checked is_pronoun, is_vague_verb
                            is_pronoun = False
                            if not merge_pronoun and mention in utils.PRONOUNS:
                                is_pronoun = True
                                mention = '%s_%d' % (mention, mention_idx_t[0])
                            if not merge_vague_verb and mention in utils.VAGUE_VERBS:
                                mention = '%s_%d' % (mention, mention_idx_t[0])
                            coref_idxt_mention_map_temp[tuple(mention_idx_t)] = mention
                            if mention not in G.nodes:
                                if mention not in coref_nodes:
                                    coref_nodes[mention] = []
                                coref_nodes[mention].append((mention_idx_t[0], mention_idx_t))
                    for mention, idx_list in coref_nodes.items():
                        G.add_node(mention, node_type='coref', idx=idx_list, mention_type='pronoun' if is_pronoun else '')
                    for coref_cluster in coref_cluster_dict[docno]:
                        for i in range(len(coref_cluster)-1):   # connect two adjacent mentions in cluster
                            j = i + 1
                            mention_i_idx_t, mention_j_idx_t = coref_cluster[i], coref_cluster[j]
                            mention_i = coref_idxt_mention_map_temp[tuple(mention_i_idx_t)]
                            mention_j = coref_idxt_mention_map_temp[tuple(mention_j_idx_t)]
                            edge_weight_pre = G.edges.get((mention_i, mention_j), {'weight': 0.0})['weight']
                            #print('i:', mention_i_idx_t, mention_i, ' j:', mention_j_idx_t, mention_j)
                            #print('edge_weight_pre', edge_weight_pre)
                            edge_weight_to_add = math.e ** (2 / (mention_i_idx_t[0] + mention_j_idx_t[1]))
                            G.add_edge(mention_i, mention_j, weight=edge_weight_pre + edge_weight_to_add)
                ## rank graph
                graph_pr = nx.pagerank(G, alpha=0.85)
                graph_pr_sorted = sorted(graph_pr.items(), key=lambda _: _[1], reverse=True)
                ## collect top entities
                top10_entities = []
                for mention, score in graph_pr_sorted:
                    if G.nodes[mention]['node_type'] != 'entity':
                        continue
                    elif G.nodes[mention]['mention_type'] == 'pronoun':
                        continue
                    top10_entities.append(mention)
                    if len(top10_entities) >= 10:
                        break
                fwrite.write(json.dumps({'docno': docno, 'top10_entities': top10_entities}) + '\n')
    fwrite.close()


def get_salient_entity_event(ground_truth_dict):
    """
    Args:
        ground_truth_dict: dict
    Returns:
        salient_entities: list
        salient_events: list
    """
    salient_entities = []
    salient_events = []
    for field, entity_dict_list in ground_truth_dict['spot'].items():
        for entity_dict in entity_dict_list:
            if entity_dict['salience'] != 1:
                continue
            salient_entities.append({
                    "mention": entity_dict['surface'],
                    "loc": entity_dict['loc'],
                    'wiki_name': entity_dict['wiki_name'],
                    'field': field,
                    })
    for field, event_dict_list in ground_truth_dict['event'].items():
        for event_dict in event_dict_list:
            if event_dict['salience'] != 1:
                continue
            salient_events.append({
                    "mention": event_dict['surface'],
                    "loc": event_dict['loc'],
                    'frame_name': event_dict['frame_name'],
                    'field': field,
                    })
    return salient_entities, salient_events


def generate_entity_event_nodes(elit_paragraphs):
    """
    Args:
        elit_paragraphs: dict
    Returns:
        entity_nodes: dict, mention-> {idx: [(headword_pos, (idx_s, idx_e)), ()], is_pronoun: bool}
        event_nodes: dict,  mention-> {idx: [(headword_pos, (idx_s, idx_e)), ()], is_vague: bool}
    """
    entity_nodes = {}  # mention-> {idx: [(headword_pos, (idx_s, idx_e)), ()], is_pronoun: bool}
    event_nodes = {}   # mention-> {idx: [(headword_pos, (idx_s, idx_e)), ()], is_vague: bool}
    cur_tok_offset = 0
    for elit_para_res in elit_paragraphs:   # contains multiple paragraphs
        for elit_sent_res in elit_para_res['sens']:
            if len(elit_sent_res['dep']) <= 1:
                cur_tok_offset += len(elit_sent_res['dep'])
                continue
            # tree apis are designed for old version dep result
            elit_sent_res['dep'] = convert_elit_dep_to_old_version(elit_sent_res['dep'])
            # Get NPs
            dep_root, dep_tok_root, dep_rel_root = construct_dependency_tree_from_parser_res(elit_sent_res['dep'], elit_sent_res['tok'], elit_sent_res['pos'], elit_sent_res['off'])
            all_NP_idxes = {}
            all_NP_idxes = extract_noun_phrase_heuristically_recursion(elit_sent_res['dep'], elit_sent_res['tok'], elit_sent_res['pos'], dep_root, None, None, all_NP_idxes)
            NPs, NP_idxes, NP_headword_idxes = generate_NPs_from_all_NP_idxes(elit_sent_res['tok'], all_NP_idxes, elit_sent_res['off'])
            #print('NPs', NPs)
            # Construct entity nodes by Merging Same Mention
            for idx in range(len(NPs)):
                mention = NPs[idx].lower()
                if mention not in entity_nodes:
                    is_pronoun = False
                    if len(NP_idxes[idx]) == 1 and elit_sent_res['pos'][NP_idxes[idx][0]].startswith('PRP'):
                        is_pronoun = True
                    entity_nodes[mention] = {'idx':[], 'is_pronoun': is_pronoun}
                NP_start_end_idx_tuple = (NP_idxes[idx][0]+cur_tok_offset, NP_idxes[idx][-1]+cur_tok_offset)
                entity_nodes[mention]['idx'].append((NP_headword_idxes[idx]+cur_tok_offset, NP_start_end_idx_tuple))
            # Get Verbs
            all_verb_idxes = extract_verbal_predicates_prefilter(elit_sent_res['dep'], elit_sent_res['tok'], elit_sent_res['pos'], do_prefilter=False)
            triggers, trigger_idxes, trigger_headword_idxes = generate_NPs_from_all_NP_idxes(elit_sent_res['tok'], all_verb_idxes, elit_sent_res['off'])
            #print(elit_sent_res)
            #dep_root.pretty_print()
            #dep_tok_root.pretty_print()
            #dep_rel_root.pretty_print()
            #print('triggers:', triggers)
            # Construct event nodes by stemming and merging verbs
            for idx in range(len(triggers)):
                # "'s" -> 'is'
                trigger = triggers[idx] if triggers[idx] != "'s" else "is"
                trigger_lemma = lemmatize_verb_trigger(trigger, trigger_idxes[idx], trigger_headword_idxes[idx], elit_sent_res['tok'], elit_sent_res['pos'])
                if trigger_lemma not in event_nodes:
                    event_nodes[trigger_lemma] = {'idx':[], 'is_vague': trigger_lemma in utils.VAGUE_VERBS}
                trigger_start_end_idx_tuple = ((trigger_idxes[idx][0]+cur_tok_offset, trigger_idxes[idx][-1]+cur_tok_offset))
                event_nodes[trigger_lemma]['idx'].append((trigger_headword_idxes[idx]+cur_tok_offset, trigger_start_end_idx_tuple))
            cur_tok_offset += len(elit_sent_res['dep'])
    return entity_nodes, event_nodes


def obtain_sltidx_sidx_wstidx_map(elit_paragraphs):
    """
    obtain the dict SentLevelTokIdx -> (SentIdx, WithinSentTokidx)
    Args:
        elit_paragraphs: dict
    Returns:
        elit_idx_tree_map: list, [(sent_idx, within_sent_tidx)]
        dep_root_list: list of nltk.tree.Tree(), sentence level root for token index dep tree
    """
    elit_idx_tree_map = []
    dep_root_list = []
    cur_sent_offset = 0
    for elit_para_res in elit_paragraphs:   # contains multiple paragraphs
        for elit_sent_res in elit_para_res['sens']:
            if len(elit_sent_res['dep']) <= 1:
                dep_root_list.append(None)
            else:
                # tree apis are designed for old version dep result
                elit_sent_res['dep'] = convert_elit_dep_to_old_version(elit_sent_res['dep'])
                dep_root, dep_tok_root, dep_rel_root = construct_dependency_tree_from_parser_res(elit_sent_res['dep'], elit_sent_res['tok'], elit_sent_res['pos'], elit_sent_res['off'])
                dep_root_list.append(dep_root)
            for idx in range(len(elit_sent_res['dep'])):
                elit_idx_tree_map.append((cur_sent_offset, idx))
            cur_sent_offset += 1
    return elit_idx_tree_map, dep_root_list


def entity_event_graph_position_weighting_post_filter(elit_res_dir, output_file_path, include_headline=False, entity_node_only=True):
    """
    Args:
        elit_res_dir: str
        output_file_path: str
        include_headline: boolean
        entity_node_only: boolean
    """
    fwrite = open(output_file_path, 'w')
    for year_file in os.listdir(elit_res_dir):
        print('Now process year-%s...' % (year_file))
        with open(os.path.join(elit_res_dir, year_file)) as fopen:
            for line in tqdm.tqdm(fopen):
                line_res = json.loads(line.strip())
                docno = line_res['doc_id']
                if not include_headline:
                    paragraphs = line_res['elit_res'][1:]   # headline is in para#0
                else:
                    paragraphs = line_res['elit_res']
                entity_nodes, event_nodes = generate_entity_event_nodes(paragraphs)
                #print('entity_nodes', entity_nodes)
                #print('event_nodes', event_nodes)

                # Construct Graph
                G = nx.Graph()
                ## add nodes
                for mention, v_dict in entity_nodes.items():
                    G.add_node(mention, node_type="entity", idx=v_dict['idx'], mention_type='pronoun' if v_dict['is_pronoun'] else '')
                if not entity_node_only:
                    for mention, v_dict in event_nodes.items():
                        G.add_node(mention, node_type="event", idx=v_dict['idx'], mention_type='vague_verb' if v_dict['is_vague'] else '')
                ## add edge weights
                node_list = list(G.nodes())
                for i in range(len(node_list)):
                    for j in range(i+1, len(node_list)):
                        node_i_idxes = G.nodes[node_list[i]]['idx']
                        node_j_idxes = G.nodes[node_list[j]]['idx']
                        weight = cal_edge_weight_by_mention_position(node_i_idxes, node_j_idxes)
                        G.add_edge(node_list[i], node_list[j], weight=weight)
                #print(G.nodes())
                graph_pr = nx.pagerank(G, alpha=0.85)
                graph_pr_sorted = sorted(graph_pr.items(), key=lambda _: _[1], reverse=True)
                #print(graph_pr_sorted)
                ## collect top entities
                top10_entities = []
                for mention, score in graph_pr_sorted:
                    if G.nodes[mention]['node_type'] != 'entity':
                        continue
                    elif G.nodes[mention]['mention_type'] == 'pronoun':
                        continue
                    top10_entities.append(mention)
                    if len(top10_entities) >= 10:
                        break
                #print(top10_entities)
                fwrite.write(json.dumps({'docno': docno, 'top10_entities': top10_entities}) + '\n')
    fwrite.close()


def generate_corpus_entity_event_nodes(elit_res_dir, output_dir, include_headline):
    """
    produce intermediate result of entity and event nodes
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for year_file in os.listdir(elit_res_dir):
        print('Now process year-%s...' % (year_file))
        with open(os.path.join(elit_res_dir, year_file)) as fopen, open(os.path.join(output_dir, year_file), 'w') as fwrite:
            for line in tqdm.tqdm(fopen):
                line_res = json.loads(line.strip())
                docno = line_res['doc_id']
                if not include_headline:
                    paragraphs = line_res['elit_res'][1:]   # headline is in para#0
                else:
                    paragraphs = line_res['elit_res']
                entity_nodes, event_nodes = generate_entity_event_nodes(paragraphs)
                res_out = {'docno': docno,
                           'entity_nodes': entity_nodes,
                           'event_nodes': event_nodes
                          }
                fwrite.write(json.dumps(res_out) + '\n')


def entity_event_graph_position_weighting_post_filter_with_coref(elit_res_dir, coref_res_dir, output_file_path, include_headline=False, entity_node_only=True):
    """
    Args:
        elit_res_dir: str
        coref_res_dir: str
        output_file_path: str
        include_headline: boolean
        entity_node_only: boolean
    """
    fwrite = open(output_file_path, 'w')
    for year_file in os.listdir(elit_res_dir):
        coref_res_path = os.path.join(coref_res_dir, year_file)
        elit_res_path = os.path.join(elit_res_dir, year_file)

        # load coref result first since the size is less
        with open(coref_res_path) as fopen:
            for line in fopen:
                line_dict = json.loads(line.strip())
                docno = line_dict['doc_id']
                tokens = line_dict['coref_res']['document']  # 1D list
                coref_clusters = line_dict['coref_res']['clusters']  # 2d list

                # TODO: deal with index mismatch
                # TODO: find a corefrence resolution system that take tokens as input
    fwrite.close()


def cal_edge_weight_by_mention_position(mention1_idx, mention2_idx):
    """
    $$ edgeWeight=\sum_{i \in mentionC1, j \in mentionC2 } \frac{1}{abs(headWordPos_i-headWordPos_j)} $$
    Mention cluster is consist of all mentions with same tokens occured in the document.
    Args:
        metion1_idx: list of [(headword_pos, (idx_s, idx_e)), ()]
        metion2_idx: list of [(headword_pos, (idx_s, idx_e)), ()]
    Return:
        weight: float
    """
    weight = 0.0
    for m1 in mention1_idx:
        for m2 in mention2_idx:
            dist = abs(m1[0]-m2[0])
            if dist == 0.0:
                #print('division by zero')
                #print(mentions1)
                #print(mentions2)
                weight += 1.0  # coref clusters may have overlaps
            else:
                weight += (1.0 / dist)
    return weight


def generate_entity_graph_tree_distance_post_filter(elit_res_dir, entity_event_nodes_dir_path, output_file_path, include_headline=False, entity_node_only=True):
    """
    Args:
        elit_res_dir: str
        entity_event_nodes_dir_path: str, optional, could be empty string
        output_file_path: str
        include_headline: boolean
        entity_node_only: boolean
    """
    fwrite = open(output_file_path, 'w')
    for year_file in os.listdir(elit_res_dir):
        print('Now process year-%s...' % (year_file))
        # load preprocessed nodes
        entity_event_node_dict = {}
        if len(entity_event_nodes_dir_path) != 0:
            with open(os.path.join(entity_event_nodes_dir_path, year_file)) as fopen:
                for line in fopen:
                    line_dict = json.loads(line.strip())
                    entity_event_node_dict[line_dict['docno']] = line_dict
            print('load preprocessed entity event nodes DONE')
        # produce graph per doc
        with open(os.path.join(elit_res_dir, year_file)) as fopen:
            for line in tqdm.tqdm(fopen):
                line_res = json.loads(line.strip())
                docno = line_res['doc_id']
                if not include_headline:
                    paragraphs = line_res['elit_res'][1:]   # headline is in para#0
                else:
                    paragraphs = line_res['elit_res']
                # Build nodes
                if entity_event_node_dict:
                    preprocessed_nodes = entity_event_node_dict[docno]
                    entity_nodes, event_nodes = preprocessed_nodes['entity_nodes'], preprocessed_nodes['event_nodes']
                else:
                    entity_nodes, event_nodes = generate_entity_event_nodes(paragraphs)
                # Obtain necessary mapping and dep trees
                elit_idx_tree_map, dep_root_list = obtain_sltidx_sidx_wstidx_map(paragraphs)
                #print('elit_idx_tree_map', elit_idx_tree_map)
                # Construct Graph
                G = nx.Graph()
                ## add nodes
                for mention, v_dict in entity_nodes.items():
                    G.add_node(mention, node_type="entity", idx=v_dict['idx'], mention_type='pronoun' if v_dict['is_pronoun'] else '')
                if not entity_node_only:
                    for mention, v_dict in event_nodes.items():
                        G.add_node(mention, node_type="event", idx=v_dict['idx'], mention_type='vague_verb' if v_dict['is_vague'] else '')
                ## add edge weights
                node_list = list(G.nodes())
                for i in range(len(node_list)):
                    for j in range(i+1, len(node_list)):
                        node_i_idxes = G.nodes[node_list[i]]['idx']  # stores doc-level token idx
                        node_j_idxes = G.nodes[node_list[j]]['idx']
                        edge_weight = cal_edge_weight_based_on_dep_tree_distance(node_i_idxes, node_j_idxes, elit_idx_tree_map, dep_root_list)
                        if edge_weight > 0.0:
                            #print('node_i', node_list[i], node_i_idxes)
                            #print('node_j', node_list[j], node_j_idxes)
                            #print('edge weight', edge_weight)
                            G.add_edge(node_list[i], node_list[j], weight=edge_weight)
                ## rank graph
                graph_pr = nx.pagerank(G, alpha=0.85)
                graph_pr_sorted = sorted(graph_pr.items(), key=lambda _: _[1], reverse=True)
                ## collect top entities
                top10_entities = []
                for mention, score in graph_pr_sorted:
                    if G.nodes[mention]['node_type'] != 'entity':
                        continue
                    elif G.nodes[mention]['mention_type'] == 'pronoun':
                        continue
                    top10_entities.append(mention)
                    if len(top10_entities) >= 10:
                        break
                fwrite.write(json.dumps({'docno': docno, 'top10_entities': top10_entities}) + '\n')
    fwrite.close()


def cal_edge_weight_based_on_dep_tree_distance(node_i_idxes, node_j_idxes, elit_idx_tree_map, dep_root_list):
    """
    Only calculate within- sentence tree distance
    Args:
        node_i_idxes: list of [(headword_pos, (idx_s, idx_e)), ()]
        node_j_idxes: list of [(headword_pos, (idx_s, idx_e)), ()]
        elit_idx_tree_map: list, [(sent_idx, within_sent_tidx)]
        dep_root_list: list of nltk.tree.Tree()
    Returns:
        weight: float
    """
    weight = 0.0
    for m_i in node_i_idxes:
        for m_j in node_j_idxes:
            sent_idx_i, within_sent_tidx_i = elit_idx_tree_map[m_i[0]]
            sent_idx_j, within_sent_tidx_j = elit_idx_tree_map[m_j[0]]
            if sent_idx_i != sent_idx_j:
                continue
            tree_dist = utils.calculate_node_pair_tree_distance_index_version(dep_root_list[sent_idx_i], within_sent_tidx_i, within_sent_tidx_j)
            """
            # debug
            from nltk.tree import Tree
            print('m_i', m_i, sent_idx_i, within_sent_tidx_i)
            print('m_j', m_j, sent_idx_j, within_sent_tidx_j)
            root = Tree.fromstring('%s'%(dep_root_list[sent_idx_i]))
            root.pretty_print()
            print('tree_dis:', tree_dis)
            """
            if tree_dist == 0.0:
                weight += 1.0  # division zero exception
            else:
                weight += (1.0 / tree_dist)
    return weight


def generate_entity_event_graph_tree_distance_with_coref_post_filter(elit_res_dir, entity_event_nodes_dir_path, coref_cluster_dir_path, output_file_path, include_headline=False, entity_node_only=True):
    """
    Insert coreference clusters into ee graph. Basically add more edges
    Args:
        elit_res_dir: str
        entity_event_nodes_dir_path: str, optional, could be empty string
        coref_cluster_dir_path: str
        output_file_path: str
        include_headline: boolean
        entity_node_only: boolean
    """
    fwrite = open(output_file_path, 'w')
    for year_file in os.listdir(elit_res_dir):
        print('Now process year-%s...' % (year_file))
        # load preprocessed nodes
        entity_event_node_dict = {}
        if len(entity_event_nodes_dir_path) != 0:
            with open(os.path.join(entity_event_nodes_dir_path, year_file)) as fopen:
                for line in fopen:
                    line_dict = json.loads(line.strip())
                    entity_event_node_dict[line_dict['docno']] = line_dict
            print('load preprocessed entity event nodes DONE')
        # load preprocessed coref clusters
        coref_cluster_dict = {}  # docno -> 2d list of tidx
        with open(os.path.join(coref_cluster_dir_path, '%slines.output'%(year_file))) as fopen:
            for line in fopen:
                line_dict = json.loads(line.strip())
                coref_cluster_dict[line_dict['docno']] = line_dict['predicted_clusters']
        print('load preprocessed coreference clusters DONE')
        # produce graph per doc
        with open(os.path.join(elit_res_dir, year_file)) as fopen:
            for line in tqdm.tqdm(fopen):
                line_res = json.loads(line.strip())
                docno = line_res['doc_id']
                if not include_headline:
                    paragraphs = line_res['elit_res'][1:]   # headline is in para#0
                else:
                    paragraphs = line_res['elit_res']
                # Build nodes
                if entity_event_node_dict:
                    preprocessed_nodes = entity_event_node_dict[docno]
                    entity_nodes, event_nodes = preprocessed_nodes['entity_nodes'], preprocessed_nodes['event_nodes']
                else:
                    entity_nodes, event_nodes = generate_entity_event_nodes(paragraphs)
                # Obtain necessary mapping and dep trees
                elit_idx_tree_map, dep_root_list = obtain_sltidx_sidx_wstidx_map(paragraphs)
                #print('elit_idx_tree_map', elit_idx_tree_map)
                # Construct Graph
                G = nx.Graph()
                ## add nodes
                for mention, v_dict in entity_nodes.items():
                    G.add_node(mention, node_type="entity", idx=v_dict['idx'], mention_type='pronoun' if v_dict['is_pronoun'] else '')
                if not entity_node_only:
                    for mention, v_dict in event_nodes.items():
                        G.add_node(mention, node_type="event", idx=v_dict['idx'], mention_type='vague_verb' if v_dict['is_vague'] else '')
                ## add edge weights between nodes
                node_list = list(G.nodes())
                for i in range(len(node_list)):
                    for j in range(i+1, len(node_list)):
                        node_i_idxes = G.nodes[node_list[i]]['idx']  # stores doc-level token idx
                        node_j_idxes = G.nodes[node_list[j]]['idx']
                        edge_weight = cal_edge_weight_based_on_dep_tree_distance(node_i_idxes, node_j_idxes, elit_idx_tree_map, dep_root_list)
                        if edge_weight > 0.0:
                            G.add_edge(node_list[i], node_list[j], weight=edge_weight)
                ## add edge weights between coref nodes
                coref_nodes = {}  # mention -> [(headword_pos, (idx_s, idx_e)), ()]
                coref_idxt_mention_map_temp = {}  # tuple -> mention
                for coref_cluster in coref_cluster_dict[docno]:  # add nodes first
                    for mention_idx_t in coref_cluster:
                        mention = utils.get_mention_by_document_level_idx_tuple(paragraphs, mention_idx_t).lower()
                        coref_idxt_mention_map_temp[tuple(mention_idx_t)] = mention
                        if mention not in G.nodes:
                            if mention not in coref_nodes:
                                coref_nodes[mention] = []
                            coref_nodes[mention].append((mention_idx_t[0], mention_idx_t))
                for mention, idx_list in coref_nodes.items():
                    G.add_node(mention, node_type='coref', idx=idx_list, mention_type='')
                for coref_cluster in coref_cluster_dict[docno]:
                    for i in range(len(coref_cluster)-1):   # connect two adjacent mentions in cluster
                        j = i + 1
                        mention_i_idx_t, mention_j_idx_t = coref_cluster[i], coref_cluster[j]
                        mention_i = coref_idxt_mention_map_temp[tuple(mention_i_idx_t)]
                        mention_j = coref_idxt_mention_map_temp[tuple(mention_j_idx_t)]
                        edge_weight_pre = G.edges.get((mention_i, mention_j), {'weight': 0.0})['weight']
                        #print('i:', mention_i_idx_t, mention_i, ' j:', mention_j_idx_t, mention_j)
                        #print('edge_weight_pre', edge_weight_pre)
                        edge_weight_to_add = math.e ** (2 / (mention_i_idx_t[0] + mention_j_idx_t[1]))
                        G.add_edge(mention_i, mention_j, weight=edge_weight_pre + edge_weight_to_add)
                ## rank graph
                graph_pr = nx.pagerank(G, alpha=0.85)
                graph_pr_sorted = sorted(graph_pr.items(), key=lambda _: _[1], reverse=True)
                ## collect top entities
                top10_entities = []
                for mention, score in graph_pr_sorted:
                    if G.nodes[mention]['node_type'] != 'entity':
                        continue
                    elif G.nodes[mention]['mention_type'] == 'pronoun':
                        continue
                    top10_entities.append(mention)
                    if len(top10_entities) >= 10:
                        break
                fwrite.write(json.dumps({'docno': docno, 'top10_entities': top10_entities}) + '\n')
    fwrite.close()


if __name__ == '__main__':
    elit_res_dir = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/test_set_elit_dep_trees'
    #output_path = './data/Xiong_SIGIR18/baseline_results/entity_only_graph_postfilter_position_weight.json'
    #entity_event_graph_position_weighting_post_filter(elit_res_dir, output_path, include_headline=False, entity_node_only=True)

    output_path = './data/Xiong_SIGIR18/baseline_results/entity_event_graph_postfilter_position_weight.json'
    #entity_event_graph_position_weighting_post_filter(elit_res_dir, output_path, include_headline=False, entity_node_only=False)

    intermediate_output_dir = './data/Xiong_SIGIR18/test_set_entity_event_nodes_wo_headline'
    #generate_corpus_entity_event_nodes(elit_res_dir, intermediate_output_dir, include_headline=False)
    intermediate_output_dir = './data/Xiong_SIGIR18/test_set_entity_event_nodes_w_headline'
    #generate_corpus_entity_event_nodes(elit_res_dir, intermediate_output_dir, include_headline=True)

    entity_event_nodes_dir_path = ''
    output_file_path = './data/Xiong_SIGIR18/baseline_results/entity_graph_tree_distance_postfilter_wo_headline.json'
    #generate_entity_graph_tree_distance_post_filter(elit_res_dir, entity_event_nodes_dir_path, output_file_path, include_headline=False, entity_node_only=True)
    entity_event_nodes_dir_path = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/test_set_entity_event_nodes_wo_headline'
    output_file_path = './data/Xiong_SIGIR18/baseline_results/entity_event_graph_tree_distance_postfilter_wo_headline.json'
    #generate_entity_graph_tree_distance_post_filter(elit_res_dir, entity_event_nodes_dir_path, output_file_path, include_headline=False, entity_node_only=False)

    coref_cluster_dir_path = './data/Xiong_SIGIR18/test_set_spanbert_wo_headline/spanbert_elit_index_matching_output'
    output_file_path = './data/Xiong_SIGIR18/baseline_results/entity_graph_tree_distance_with_coref_postfilter_wo_headline.json'
    #generate_entity_event_graph_tree_distance_with_coref_post_filter(elit_res_dir, entity_event_nodes_dir_path, coref_cluster_dir_path, output_file_path, include_headline=False, entity_node_only=True)
    output_file_path = './data/Xiong_SIGIR18/baseline_results/entity_event_graph_tree_distance_with_coref_postfilter_wo_headline.json'
    #generate_entity_event_graph_tree_distance_with_coref_post_filter(elit_res_dir, entity_event_nodes_dir_path, coref_cluster_dir_path, output_file_path, include_headline=False, entity_node_only=False)

    output_file_path = './data/Xiong_SIGIR18/baseline_results/entity_graph_dep_tree_strucutre_wo_coref_postfilter_wo_headline.json'
    #generate_entity_graph_dep_tree_structure_post_filter(elit_res_dir, output_file_path, add_coref=(False, ''), include_headline=False, merge_pronoun=True)
    output_file_path = './data/Xiong_SIGIR18/baseline_results/entity_graph_dep_tree_strucutre_w_coref_postfilter_wo_headline.json'
    #generate_entity_graph_dep_tree_structure_post_filter(elit_res_dir, output_file_path, add_coref=(True, coref_cluster_dir_path), include_headline=False, merge_pronoun=True)

    output_file_path = './data/Xiong_SIGIR18/baseline_results/entity_event_graph_dep_tree_strucutre_wo_coref_postfilter_wo_headline.json'
    #generate_entity_event_graph_dep_tree_structure_post_filter(elit_res_dir, output_file_path, add_coref=(True, coref_cluster_dir_path), include_headline=False, merge_pronoun=False, merge_vague_verb=False)
    output_file_path = './data/Xiong_SIGIR18/baseline_results/entity_event_graph_dep_tree_strucutre_w_coref_postfilter_wo_headline.json'
    #generate_entity_event_graph_dep_tree_structure_post_filter(elit_res_dir, output_file_path, add_coref=(True, coref_cluster_dir_path), include_headline=False, merge_pronoun=False, merge_vague_verb=False)


    output_file_path = './data/Xiong_SIGIR18/baseline_results/entity_graph_dep_tree_strucutre_GloVe_dist_w_coref_postfilter_wo_headline.json'
    #generate_entity_graph_dep_tree_structure_post_filter(elit_res_dir, output_file_path, add_coref=(True, coref_cluster_dir_path), include_headline=False, merge_pronoun=True, use_emb_distance=True)
    output_file_path = './data/Xiong_SIGIR18/baseline_results/entity_graph_dep_tree_strucutre_add_sibling_edges_w_coref_postfilter_wo_headline.json'
    generate_entity_graph_dep_tree_structure_post_filter(elit_res_dir, output_file_path, add_coref=(True, coref_cluster_dir_path), include_headline=False, merge_pronoun=True, use_emb_distance=False, add_sibling_edges=True)
