# -*- coding:utf-8 -*-
# Author: Jiaying Lu
# Date: 2019-04-02

import os
import math
import json
import collections
import tqdm
from nltk.stem import WordNetLemmatizer
import networkx

from . import utils
from .dependency_parsing import construct_dependency_tree_from_parser_res, extract_noun_phrase_heuristically_recursion, generate_NPs_from_all_NP_idxes, extract_verbal_predicates_prefilter
from .graph_construct import lemmatize_verb_trigger


LEMMATIZER = None

def load_lemmatizer():
    global LEMMATIZER
    if LEMMATIZER == None:
        LEMMATIZER = WordNetLemmatizer()
    return LEMMATIZER


def extract_noun_phrases_in_title(elit_res_dir, output_file_path, add_first_sentence=False):
    """
    Ranked by phrase frequency
    Args:
        elit_res_dir: str
        output_file_path: str
        add_first_sentence: boolean, optional
    """
    fwrite = open(output_file_path, 'w')
    for year_file in os.listdir(elit_res_dir):
        print('Now process year-%s...' % (year_file))
        # produce graph per doc
        with open(os.path.join(elit_res_dir, year_file)) as fopen:
            for line in tqdm.tqdm(fopen):
                line_res = json.loads(line.strip())
                docno = line_res['doc_id']
                title = line_res['elit_res'][0]['sens']  # may contain several sentences
                if add_first_sentence and len(line_res['elit_res']) > 2:
                    first_sent = line_res['elit_res'][1]['sens'][0]
                    title.append(first_sent)
                title_NPs = []
                for elit_sent_res in title:
                    if len(elit_sent_res['dep']) <= 1:
                        continue
                    elit_sent_res['dep'] = utils.convert_elit_dep_to_old_version(elit_sent_res['dep'])
                    # Get NPs
                    dep_root, dep_tok_root, dep_rel_root = construct_dependency_tree_from_parser_res(elit_sent_res['dep'], elit_sent_res['tok'], elit_sent_res['pos'], elit_sent_res['off'])
                    all_NP_idxes = {}
                    all_NP_idxes = extract_noun_phrase_heuristically_recursion(elit_sent_res['dep'], elit_sent_res['tok'], elit_sent_res['pos'], dep_root, None, None, all_NP_idxes)
                    NPs, NP_idxes, NP_headword_idxes = generate_NPs_from_all_NP_idxes(elit_sent_res['tok'], all_NP_idxes, elit_sent_res['off'])
                    title_NPs.extend(NPs)
                entity_frequency_map = {}  # mention -> (frequency, first_occurence_index)
                content = ' '.join(line_res['content']).lower()
                for NP in title_NPs:
                    NP = NP.lower()
                    if NP in utils.PRONOUNS:
                        continue
                    frequency = len(content.split(NP)) - 1
                    first_occurence_index = content.find('%s ' % (NP))
                    if first_occurence_index == -1:
                        first_occurence_index = len(content)
                    entity_frequency_map[NP] = (frequency, first_occurence_index)
                top10_entities = sorted(entity_frequency_map.items(), key=lambda _: (_[1][0], -_[1][1]), reverse=True)
                top10_entities = [_[0] for _ in top10_entities[:10]]
                fwrite.write(json.dumps({'docno': docno, 'top10_entities': top10_entities}) + '\n')
    fwrite.close()


def extract_entity_event_seed_position_weighted_freq(elit_res_dir, output_file_path):
    """
    w = exp^(1/pos)
    Args:
        elit_res_dir: str
        output_file_path: str
        add_first_sentence: boolean, optional
    """
    fwrite = open(output_file_path, 'w')
    for year_file in os.listdir(elit_res_dir):
        print('Now process year-%s...' % (year_file))
        # produce graph per doc
        with open(os.path.join(elit_res_dir, year_file)) as fopen:
            for line in tqdm.tqdm(fopen):
                line_res = json.loads(line.strip())
                docno = line_res['doc_id']
                entity_candidates = {}
                trigger_lemma_freq = {}
                event_candidates = {}
                tok_offset = 0
                sent_idx = 0
                for elit_para_res in line_res['elit_res']:
                    for elit_sent_res in elit_para_res['sens']:
                        sent_idx += 1
                        if len(elit_sent_res['dep']) <= 1:
                            tok_offset += len(elit_sent_res['tok'])
                            continue
                        entities, events, event_arguments = extract_entity_event_from_sentence(elit_sent_res, tok_offset)
                        tok_offset += len(elit_sent_res['tok'])
                        for ent_t in entities:
                            ent_lemma = ' '.join(ent_t[1])
                            if ent_lemma not in entity_candidates:
                                entity_candidates[ent_lemma] = 0.0   # pos-weighted freq
                            entity_candidates[ent_lemma] += (math.e ** (1/sent_idx))
                        for evn_idx in range(len(events)):
                            trigger_lemma = ' '.join(events[evn_idx][1])
                            # trigger_widx = events[evn_idx][2]
                            arguments_lemmas = sorted([' '.join(arg[1]) for arg in event_arguments[evn_idx]])
                            arguments_lemmas = '\001'.join(arguments_lemmas)
                            evn_key = (trigger_lemma, arguments_lemmas)
                            if trigger_lemma not in trigger_lemma_freq:
                                trigger_lemma_freq[trigger_lemma] = 0.0
                            trigger_lemma_freq[trigger_lemma] += (math.e ** (1/sent_idx))
                            if evn_key not in event_candidates:
                                event_candidates[evn_key] = 0.0
                            event_candidates[evn_key] += (math.e ** (1/sent_idx))
                # Entity sort by score
                top10_entities = sorted(entity_candidates.items(), key=lambda _: _[1], reverse=True)[:10]
                # Event sort by freq, then loc
                event_candidate_scores = {}
                for evn_key, freq in event_candidates.items():
                    event_freq_score = freq
                    trigger_lemma_freq_score = trigger_lemma_freq[evn_key[0]]
                    arguments_score = 0.0
                    for argument in evn_key[1].split('\001'):
                        if argument != '':
                            if argument in entity_candidates:
                                arguments_score += entity_candidates[argument]
                            if argument in trigger_lemma_freq:
                                arguments_score += trigger_lemma_freq[argument]
                    score = event_freq_score + 0.1 * trigger_lemma_freq_score + 0.3 * arguments_score
                    event_candidate_scores[evn_key] = (score, event_freq_score, trigger_lemma_freq_score, arguments_score)
                top10_events = sorted(event_candidate_scores.items(), key=lambda _: _[1][0], reverse=True)[:10]
                line_output = {
                        'docno': docno,
                        'top10_entities': [_[0] for _ in top10_entities],
                        'top10_events': [_[0][0] for _ in top10_events],
                        'top10_event_arguments': [_[0][1].split('\001') for _ in top10_events],
                        }
                fwrite.write(json.dumps(line_output) + '\n')
    fwrite.close()


def extract_initial_seed_entities_coref_version(elit_res_dir, coref_cluster_dir, output_file_path, add_first_sentence=True, dynamic_expansion_whole_content=False):
    """
    Ranked by coref cluster; Dynamic expansion by coref size
    Args:
        elit_res_dir: str
        coref_cluster_dir: str
        output_file_path: str
        add_first_sentence: boolean, optional
        dynamic_expansion_whole_content: boolean, optional
    """
    fwrite = open(output_file_path, 'w')
    for year_file in os.listdir(elit_res_dir):
        print('Now process year-%s...' % (year_file))
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
                # prepare NPs in title and first sent
                title = line_res['elit_res'][0]['sens']  # may contain several sentences
                if add_first_sentence and len(line_res['elit_res']) > 2:
                    first_sent = line_res['elit_res'][1]['sens'][0]
                    title.append(first_sent)
                candidate_NPs_information = []   # ((mention, headw_idx, word_index_tuple))
                for elit_sent_res in title:
                    if len(elit_sent_res['dep']) <= 1:
                        continue
                    elit_sent_res['dep'] = utils.convert_elit_dep_to_old_version(elit_sent_res['dep'])
                    # Get NPs
                    dep_root, dep_tok_root, dep_rel_root = construct_dependency_tree_from_parser_res(elit_sent_res['dep'], elit_sent_res['tok'], elit_sent_res['pos'], elit_sent_res['off'])
                    all_NP_idxes = {}
                    all_NP_idxes = extract_noun_phrase_heuristically_recursion(elit_sent_res['dep'], elit_sent_res['tok'], elit_sent_res['pos'], dep_root, None, None, all_NP_idxes)
                    NPs, NP_idxes, NP_headword_idxes = generate_NPs_from_all_NP_idxes(elit_sent_res['tok'], all_NP_idxes, elit_sent_res['off'])
                    NP_idxes = [(_[0], _[-1]) for _ in NP_idxes]
                    candidate_NPs_information.extend([(NPs[i], NP_headword_idxes[i], NP_idxes[i]) for i in range(len(NPs))])
                # prepare NPs in title and whole content
                all_NPs_information = dict()   # (index_start, index_end) -> mention, headword_index
                all_NPs_centrality = dict()    # (index_start, index_end) -> [(within_sent_index_tuple1), (...), ()]
                for elit_para_res in line_res['elit_res']:
                    for elit_sent_res in elit_para_res['sens']:
                        if len(elit_sent_res['dep']) <= 1:
                            continue
                        elit_sent_res['dep'] = utils.convert_elit_dep_to_old_version(elit_sent_res['dep'])
                        # Get NPs
                        dep_root, dep_tok_root, dep_rel_root = construct_dependency_tree_from_parser_res(elit_sent_res['dep'], elit_sent_res['tok'], elit_sent_res['pos'], elit_sent_res['off'])
                        all_NP_idxes = {}
                        all_NP_idxes = extract_noun_phrase_heuristically_recursion(elit_sent_res['dep'], elit_sent_res['tok'], elit_sent_res['pos'], dep_root, None, None, all_NP_idxes)
                        NPs, NP_idxes, NP_headword_idxes = generate_NPs_from_all_NP_idxes(elit_sent_res['tok'], all_NP_idxes, elit_sent_res['off'])
                        NP_idxes = [(_[0], _[-1]) for _ in NP_idxes]
                        all_NPs_information.update([(NP_idxes[i], (NPs[i], NP_headword_idxes[i])) for i in range(len(NPs))])
                        # pack NPs in one sent into centrality dict
                        for NP_idx_ti in NP_idxes:
                            for NP_idx_tj in NP_idxes:
                                if NP_idx_ti == NP_idx_tj:
                                    continue
                                if not NP_idx_ti in all_NPs_centrality:
                                    all_NPs_centrality[NP_idx_ti] = []
                                all_NPs_centrality[NP_idx_ti].append(NP_idx_tj)
                # calculate coref cluster size
                entity_coref_map = {}   # mention -> (coref_cluster_size, first_occurence_index)
                visited_coref_clusters = []
                for (NP, headw_idx, word_index_tuple) in candidate_NPs_information:
                    NP = NP.lower()
                    if NP in utils.PRONOUNS:
                        continue
                    if NP in entity_coref_map:  # duplicated mention
                        continue
                    coref_cluster = get_coref_cluster_for_phrase_index_tuple(word_index_tuple, coref_cluster_dict[docno])
                    visited_coref_clusters.append(coref_cluster)
                    entity_coref_map[NP] = [2.0 * len(coref_cluster), headw_idx]
                    # add other mention in the coref cluster
                    for other_mention_idx_tuple in coref_cluster:
                        other_mention_idx_tuple = tuple(other_mention_idx_tuple)
                        if other_mention_idx_tuple == tuple(word_index_tuple):
                            continue
                        if other_mention_idx_tuple in all_NPs_information:
                            other_mention = all_NPs_information[other_mention_idx_tuple][0].lower()
                            if other_mention in utils.PRONOUNS:
                                continue
                            if other_mention not in entity_coref_map:
                                entity_coref_map[other_mention] = [len(coref_cluster), all_NPs_information[other_mention_idx_tuple][1]]
                # dynamic expasion by add entities has large coreference size
                # TODO: need modify candidate_NPs_information to include expanded mentions ?
                if dynamic_expansion_whole_content:
                    coref_sizes_entity_candidates = []
                    for (coref_size, headw_idx) in entity_coref_map.values():
                        if coref_size > 0:
                            coref_sizes_entity_candidates.append(coref_size)
                    if len(coref_sizes_entity_candidates) > 0:
                        threshold = (max(coref_sizes_entity_candidates) + min(coref_sizes_entity_candidates)) / 2
                    else:    # no candidates from title and first sentence
                        coref_cluster_sizes = sorted([len(_) for _ in coref_cluster_dict[docno]], reverse=True)
                        #threshold = sum(coref_cluster_sizes) / len(coref_cluster_sizes) if len(coref_cluster_sizes) > 0 else 0  # bad choice
                        threshold = sum(coref_cluster_sizes[:3])/len(coref_cluster_sizes[:3]) if len(coref_cluster_sizes) > 0 else 0
                    for cluster in coref_cluster_dict[docno]:
                        if cluster in visited_coref_clusters:
                            continue
                        if len(cluster) < threshold:
                            continue
                        for expanded_mention_idx_tuple in cluster:
                            expanded_mention_idx_tuple = tuple(expanded_mention_idx_tuple)
                            if expanded_mention_idx_tuple in all_NPs_information:
                                expanded_mention = all_NPs_information[expanded_mention_idx_tuple][0].lower()
                                if expanded_mention in utils.PRONOUNS:
                                    continue
                                if expanded_mention not in entity_coref_map:
                                    entity_coref_map[expanded_mention] = [0.5 * len(cluster), all_NPs_information[expanded_mention_idx_tuple][1]]
                # calculate mention frequency
                content = ' '.join([line_res['headline']] + line_res['content']).lower()
                visited_mention_frequency = set()
                for (NP, headw_idx, word_index_tuple) in candidate_NPs_information:
                    NP = NP.lower()
                    if NP in utils.PRONOUNS:
                        continue
                    if NP in visited_mention_frequency:
                        continue
                    frequency = len(content.split(NP)) - 1
                    first_occurence_index = content.find('%s ' % (NP))
                    if first_occurence_index == -1:
                        first_occurence_index = len(content)
                    entity_coref_map[NP][0] += frequency
                    visited_mention_frequency.add(NP)
                # calculate centrality
                for (NP, headw_idx, word_index_tuple) in candidate_NPs_information:
                    NP = NP.lower()
                    if NP in utils.PRONOUNS:
                        continue
                    index_tuples_within_sent = all_NPs_centrality.get(word_index_tuple, [])
                    for index_tj in index_tuples_within_sent:
                        NP_j = all_NPs_information[index_tj][0].lower()
                        if NP_j in entity_coref_map:
                            entity_coref_map[NP][0] += (0.15 * entity_coref_map[NP_j][0])
                top10_entities = sorted(entity_coref_map.items(), key=lambda _: (_[1][0], -_[1][1]), reverse=True)
                top10_entities = [_[0] for _ in top10_entities[:10]]
                fwrite.write(json.dumps({'docno': docno, 'top10_entities': top10_entities}) + '\n')
    fwrite.close()


def get_coref_cluster_for_phrase_index_tuple(phrase_index_t, clusters):
    """
    Args:
        phrase_index_t: tuple of (int, int)
        clusters: 2D list of [int, int]
    Returns:
        coref_cluster: list
    """
    coref_cluster = []
    hit_flag = False
    for cur_cluster in clusters:
        for index_t in cur_cluster:
            if phrase_index_t[0] == index_t[0] and phrase_index_t[1] == index_t[1]:
                coref_cluster = cur_cluster
                hit_flag = True
                break
        if hit_flag:
            break
    return coref_cluster


def produce_salient_events_by_headword_lemma_frequency(elit_res_dir: str, output_file_path: str) -> None:
    fwrite = open(output_file_path, 'w')
    lemmatizer = WordNetLemmatizer()
    for year_file in os.listdir(elit_res_dir):
        print('Now process year-%s...' % (year_file))
        # produce graph per doc
        with open(os.path.join(elit_res_dir, year_file)) as fopen:
            for line in tqdm.tqdm(fopen):
                line_res = json.loads(line.strip())
                docno = line_res['doc_id']
                trigger_freq_dict = {}  # trigger -> [freq, first_occurence]
                for elit_para_res in line_res['elit_res']:
                    for elit_sent_res in elit_para_res['sens']:
                        elit_sent_res['dep'] = utils.convert_elit_dep_to_old_version(elit_sent_res['dep'])
                        # Get Verbs Triggers
                        all_verb_idxes = extract_verbal_predicates_prefilter(elit_sent_res['dep'], elit_sent_res['tok'], elit_sent_res['pos'], do_prefilter=False)
                        triggers, trigger_idxes, trigger_headword_idxes = generate_NPs_from_all_NP_idxes(elit_sent_res['tok'], all_verb_idxes, elit_sent_res['off'])
                        for i in range(len(triggers)):
                            # only keep the headword
                            trigger_hwidx = trigger_headword_idxes[i]
                            trigger = elit_sent_res['tok'][trigger_hwidx]
                            if trigger == "'s":
                                trigger = "is"
                            trigger = lemmatizer.lemmatize(trigger.lower(), pos='v')
                            if trigger in utils.VAGUE_VERBS:
                                continue
                            if trigger not in trigger_freq_dict:
                                trigger_freq_dict[trigger] = [0, trigger_hwidx]
                            trigger_freq_dict[trigger][0] += 1
                # Rank by freq and pos
                top10_events = sorted(trigger_freq_dict.items(), key=lambda _: (_[1][0], -_[1][1]), reverse=True)
                top10_events = [_[0] for _ in top10_events[:10]]
                fwrite.write(json.dumps({'docno': docno, 'top10_events': top10_events}) + '\n')


def extract_entity_event_from_sentence(elit_sent_res: dict, tok_offset: int = 0) -> (list, list, list):
    # entities, events: [('mention', lemma_lst, headword_idx, (start_idx, end_idx)), (), ...]
    # event_arguments: [[('arg0_mention', lemma_lst, hw_idx, (sidx,eidx)), ('arg1_mention', 'lemma', hw_idx, (sidx,eidx)), ...], [], ...]
    entities = []
    events = []
    event_arguments = []

    event_verbs, event_nouns = utils.load_event_vocabulary()
    elit_sent_res['dep'] = utils.convert_elit_dep_to_old_version(elit_sent_res['dep'])
    # Get NPs
    dep_root, dep_tok_root, dep_rel_root = construct_dependency_tree_from_parser_res(elit_sent_res['dep'], elit_sent_res['tok'], elit_sent_res['pos'], elit_sent_res['off'])
    """
    # DEBUG
    from nltk.tree import Tree
    dep_root = Tree.fromstring('%s'%(dep_root))
    dep_root.pretty_print()
    dep_tok_root.pretty_print()
    dep_rel_root.pretty_print()
    # End DEBUG
    """
    all_NP_idxes = {}
    all_NP_idxes = extract_noun_phrase_heuristically_recursion(elit_sent_res['dep'], elit_sent_res['tok'], elit_sent_res['pos'], dep_root, None, None, all_NP_idxes)
    NPs, NP_idxes, NP_headword_idxes = generate_NPs_from_all_NP_idxes(elit_sent_res['tok'], all_NP_idxes, elit_sent_res['off'])
    NP_idxes = [(_[0], _[-1]) for _ in NP_idxes]
    # filter pronoun for evaluation purpose
    idx_to_pop = []
    for NP_i in range(len(NPs)):
        NP_hw_tidx = NP_headword_idxes[NP_i]
        if (NP_idxes[NP_i][0] == NP_idxes[NP_i][1]) and elit_sent_res['pos'][NP_hw_tidx] == 'PRP':
            idx_to_pop.append(NP_i)
    for idx in idx_to_pop[::-1]:
        NPs.pop(idx)
        NP_idxes.pop(idx)
        NP_headword_idxes.pop(idx)
    #print('NPs:', NPs)
    NP_hwidx_restuple_map = {}
    trigger_hwtidx_event_idx_map = {}
    valid_trigger_hw_idxes = set()
    for NP_i in range(len(NPs)):
        NP_hw_tidx = NP_headword_idxes[NP_i]
        NP_headword = elit_sent_res['tok'][NP_hw_tidx]
        NP_headword_pos = elit_sent_res['pos'][NP_hw_tidx]
        NP_headword_lemma = elit_sent_res['lem'][NP_hw_tidx]
        # whether current headword is in lv + noun structure
        is_in_lv_noun_structure = False
        for dep_tidx, rel in elit_sent_res['dep']:
            if rel == 'lv' and dep_tidx == NP_hw_tidx:
                is_in_lv_noun_structure = True
                break
        # headword is a event noun but not a proper name
        # OR lv + noun structure
        if (not NP_headword_pos.startswith('NNP') and NP_headword_lemma.lower() in event_nouns)\
                or is_in_lv_noun_structure:
            res_tuple = (NP_headword, [NP_headword_lemma], NP_hw_tidx+tok_offset, (NP_hw_tidx+tok_offset, NP_hw_tidx+tok_offset))
            trigger_hwtidx_event_idx_map[NP_hw_tidx] = len(events)
            events.append(res_tuple)
            valid_trigger_hw_idxes.add(NP_hw_tidx)
            # modify NP accordingly
            NPs[NP_i] = NP_headword
            NP_idxes[NP_i] = (NP_hw_tidx, NP_hw_tidx)
        else:
            #entity_lemma = [elit_sent_res['lem'][idx] if (elit_sent_res['lem'][idx] not in ['#crd#', '#ord#']) else elit_sent_res['tok'][idx].lower()
            #                for idx in range(NP_idxes[NP_i][0], NP_idxes[NP_i][1]+1)]
            s_idx, e_idx = NP_idxes[NP_i]
            entity_lemma = utils.convert_processed_mention_to_lemma_elit_version(elit_sent_res['tok'][s_idx:e_idx+1], elit_sent_res['pos'][s_idx:e_idx+1], elit_sent_res['lem'][s_idx:e_idx+1])
            res_tuple = (NPs[NP_i], entity_lemma, NP_hw_tidx+tok_offset, (s_idx+tok_offset, e_idx+tok_offset))
            entities.append(res_tuple)
        NP_hwidx_restuple_map[NP_hw_tidx] = res_tuple
    #print('entities:', entities)
    #print('events:', events)
    # Get Verbs Triggers
    all_verb_idxes = extract_verbal_predicates_prefilter(elit_sent_res['dep'], elit_sent_res['tok'], elit_sent_res['pos'], do_prefilter=False)
    triggers, trigger_idxes, trigger_headword_idxes = generate_NPs_from_all_NP_idxes(elit_sent_res['tok'], all_verb_idxes, elit_sent_res['off'])
    trigger_idxes = [(_[0], _[-1]) for _ in trigger_idxes]
    #print('triggers:', triggers)
    for trigger_i in range(len(triggers)):
        trigger_hw_tidx = trigger_headword_idxes[trigger_i]
        trigger_hw_lemma = elit_sent_res['lem'][trigger_hw_tidx]
        if trigger_hw_lemma.lower() in event_verbs:
            #trigger_lemma = [elit_sent_res['lem'][idx] if (elit_sent_res['lem'][idx] not in ['#crd#', '#ord#']) else elit_sent_res['tok'][idx].lower()
            #                 for idx in range(trigger_idxes[trigger_i][0], trigger_idxes[trigger_i][1]+1)]
            s_idx, e_idx = trigger_idxes[trigger_i]
            trigger_lemma = utils.convert_processed_mention_to_lemma_elit_version(elit_sent_res['tok'][s_idx:e_idx+1], elit_sent_res['pos'][s_idx:e_idx+1], elit_sent_res['lem'][s_idx:e_idx+1])
            trigger_hwtidx_event_idx_map[trigger_hw_tidx] = len(events)
            events.append((triggers[trigger_i], trigger_lemma, trigger_hw_tidx+tok_offset, (s_idx+tok_offset, e_idx+tok_offset)))
            valid_trigger_hw_idxes.add(trigger_hw_tidx)
    #print('add triggers, events:', events)
    # Match Event Arguments
    event_arguments = [[] for _ in range(len(events))]
    for NP_i in range(len(NPs)):
        NP_hw_tidx = NP_headword_idxes[NP_i]
        dep_head_tidx, dep_rel = elit_sent_res['dep'][NP_hw_tidx]
        if dep_head_tidx in trigger_hwtidx_event_idx_map and dep_rel in ['nsbj', 'obj', 'dat']:
            event_arguments[trigger_hwtidx_event_idx_map[dep_head_tidx]].append(NP_hwidx_restuple_map[NP_hw_tidx])
    return entities, events, event_arguments


def pseudo_annotate_NYT_corpus(elit_res_dir: str, output_file_path: str) -> None:
    fwrite = open(output_file_path, 'w')
    for year_file in os.listdir(elit_res_dir):
        print('Now process year-%s...' % (year_file))
        with open(os.path.join(elit_res_dir, year_file)) as fopen:
            for line in tqdm.tqdm(fopen):
                line_res = json.loads(line.strip())
                tok_list = []
                tok_offset = 0
                all_entities = []
                all_events = []
                all_event_arguments = []
                for elit_para_res in line_res['elit_res']:
                    for elit_sent_res in elit_para_res['sens']:
                        tok_list.append(elit_sent_res['tok'])
                        if len(elit_sent_res['dep']) <= 1:
                            tok_offset += len(elit_sent_res['tok'])
                            continue
                        entities, events, event_arguments = extract_entity_event_from_sentence(elit_sent_res, tok_offset)
                        all_entities.extend(entities)
                        all_events.extend(events)
                        all_event_arguments.extend(event_arguments)
                        tok_offset += len(elit_sent_res['tok'])
                output_dict = {
                        'docid': line_res['doc_id'],
                        'abstract': line_res['abstract'],
                        'toks': tok_list,
                        'entities': all_entities,
                        'events': all_events,
                        'event_arguments': all_event_arguments,
                        }
                fwrite.write(json.dumps(output_dict) + '\n')


def pseudo_annotate_SemanticScholar_corpus(elit_res_path: str, output_file_path: str) -> None:
    with open(elit_res_path) as fopen, open(output_file_path, 'w') as fwrite:
        for line in tqdm.tqdm(fopen):
            line_res = json.loads(line.strip())
            tok_list = []
            tok_offset = 0
            all_entities = []
            all_events = []
            all_event_arguments = []
            for elit_para_res in line_res['title_elit_res']:
                for elit_sent_res in elit_para_res['sens']:
                    tok_list.append(elit_sent_res['tok'])
                    if len(elit_sent_res['dep']) <= 1:
                        tok_offset += len(elit_sent_res['tok'])
                        continue
                    entities, events, event_arguments = extract_entity_event_from_sentence(elit_sent_res, tok_offset)
                    all_entities.extend(entities)
                    all_events.extend(events)
                    all_event_arguments.extend(event_arguments)
                    tok_offset += len(elit_sent_res['tok'])
            output_dict = {
                    'docid': line_res['doc_id'],
                    'title': line_res['title'],
                    'toks': tok_list,
                    'entities': all_entities,
                    'events': all_events,
                    'event_arguments': all_event_arguments,
                    }
            fwrite.write(json.dumps(output_dict) + '\n')


def generate_salient_entity_event_by_frequency(elit_res_dir: str, output_file_path: str) -> None:
    """
    Entities ranked by phrase lemma frequency
    Events ranked by lemma frequency
    Title included
    """
    fwrite = open(output_file_path, 'w')
    for year_file in os.listdir(elit_res_dir):
        print('Now process year-%s...' % (year_file))
        with open(os.path.join(elit_res_dir, year_file)) as fopen:
            for line in tqdm.tqdm(fopen):
                line_res = json.loads(line.strip())
                docno = line_res['doc_id']
                tok_offset = 0
                entity_candidates = {}   # phrase_lemma: [freq, first_loc]
                event_candidates = {}    # (trigger_lemma, set(arg_lemmas)): [freq, trigger_freq, first_loc]
                trigger_freq_dict = {}
                for elit_para_res in line_res['elit_res']:
                    for elit_sent_res in elit_para_res['sens']:
                        if len(elit_sent_res['dep']) <= 1:
                            tok_offset += len(elit_sent_res['tok'])
                            continue
                        entities, events, event_arguments = extract_entity_event_from_sentence(elit_sent_res, tok_offset)
                        tok_offset += len(elit_sent_res['tok'])
                        for ent_t in entities:
                            ent_lemma = ' '.join(ent_t[1])
                            if ent_lemma not in entity_candidates:
                                entity_candidates[ent_lemma] = [0, ent_t[2]]
                            entity_candidates[ent_lemma][0] += 1
                        for evn_idx in range(len(events)):
                            trigger_lemma = ' '.join(events[evn_idx][1])
                            trigger_freq_dict[trigger_lemma] = trigger_freq_dict.get(trigger_lemma, 0) + 1
                            arguments_lemmas = sorted([' '.join(arg[1]) for arg in event_arguments[evn_idx]])
                            arguments_lemmas = '\001'.join(arguments_lemmas)
                            evn_key = (trigger_lemma, arguments_lemmas)
                            if evn_key not in event_candidates:
                                event_candidates[evn_key] = [0, events[evn_idx][2]]
                            event_candidates[evn_key][0] += 1
                # add trigger_freq
                for evn_key in event_candidates:
                    trigger_lemma = evn_key[0]
                    trigger_lemma_freq = trigger_freq_dict[trigger_lemma]
                    event_candidates[evn_key].insert(1, trigger_lemma_freq)
                # sort by freq, then loc
                top10_entities = sorted(entity_candidates.items(), key=lambda _: (_[1][0],-_[1][1]), reverse=True)[:10]
                top10_events = sorted(event_candidates.items(), key=lambda _: (_[1][0],_[1][1],-_[1][2]), reverse=True)[:10]
                line_output = {
                        'docno': docno,
                        'top10_entities': [_[0] for _ in top10_entities],
                        'top10_events': [_[0][0] for _ in top10_events],
                        'top10_event_arguments': [_[0][1].split('\001') for _ in top10_events],
                        }
                #print(line_output)
                fwrite.write(json.dumps(line_output) + '\n')
    fwrite.close()


def generate_salient_entity_event_by_frequency_SemanticScholar(elit_res_path: str, output_file_path: str) -> None:
    with open(elit_res_path) as fopen, open(output_file_path, 'w') as fwrite:
        for line in tqdm.tqdm(fopen):
            line_res = json.loads(line.strip())
            docno = line_res['doc_id']
            tok_offset = 0
            entity_candidates = {}   # phrase_lemma: [freq, first_loc]
            event_candidates = {}    # (trigger_lemma, set(arg_lemmas)): [freq, trigger_freq, first_loc]
            trigger_freq_dict = {}
            for elit_para_res in line_res['paper_abstract_elit_res']:
                for elit_sent_res in elit_para_res['sens']:
                    if len(elit_sent_res['dep']) <= 1:
                        tok_offset += len(elit_sent_res['tok'])
                        continue
                    entities, events, event_arguments = extract_entity_event_from_sentence(elit_sent_res, tok_offset)
                    tok_offset += len(elit_sent_res['tok'])
                    for ent_t in entities:
                        ent_lemma = ' '.join(ent_t[1])
                        if ent_lemma not in entity_candidates:
                            entity_candidates[ent_lemma] = [0, ent_t[2]]
                        entity_candidates[ent_lemma][0] += 1
                    for evn_idx in range(len(events)):
                        trigger_lemma = ' '.join(events[evn_idx][1])
                        trigger_freq_dict[trigger_lemma] = trigger_freq_dict.get(trigger_lemma, 0) + 1
                        arguments_lemmas = sorted([' '.join(arg[1]) for arg in event_arguments[evn_idx]])
                        arguments_lemmas = '\001'.join(arguments_lemmas)
                        evn_key = (trigger_lemma, arguments_lemmas)
                        if evn_key not in event_candidates:
                            event_candidates[evn_key] = [0, events[evn_idx][2]]
                        event_candidates[evn_key][0] += 1
            # add trigger_freq
            for evn_key in event_candidates:
                trigger_lemma = evn_key[0]
                trigger_lemma_freq = trigger_freq_dict[trigger_lemma]
                event_candidates[evn_key].insert(1, trigger_lemma_freq)
            # sort by freq, then loc
            top10_entities = sorted(entity_candidates.items(), key=lambda _: (_[1][0],-_[1][1]), reverse=True)[:10]
            top10_events = sorted(event_candidates.items(), key=lambda _: (_[1][0],_[1][1],-_[1][2]), reverse=True)[:10]
            line_output = {
                    'docno': docno,
                    'top10_entities': [_[0] for _ in top10_entities],
                    'top10_events': [_[0][0] for _ in top10_events],
                    'top10_event_arguments': [_[0][1].split('\001') for _ in top10_events],
                    }
            #print(line_output)
            fwrite.write(json.dumps(line_output) + '\n')


def generate_salient_entity_event_by_location(elit_res_dir: str, output_file_path: str) -> None:
    """
    Entities ranked by location
    Events ranked by location
    Title included
    """
    fwrite = open(output_file_path, 'w')
    for year_file in os.listdir(elit_res_dir):
        print('Now process year-%s...' % (year_file))
        with open(os.path.join(elit_res_dir, year_file)) as fopen:
            for line in tqdm.tqdm(fopen):
                line_res = json.loads(line.strip())
                docno = line_res['doc_id']
                tok_offset = 0
                entity_candidates = []   # [phrase_lemma, ... ]
                event_candidates = []    # [(trigger_lemma, set(arguments)), ()]
                for elit_para_res in line_res['elit_res']:
                    for elit_sent_res in elit_para_res['sens']:
                        if len(elit_sent_res['dep']) <= 1:
                            tok_offset += len(elit_sent_res['tok'])
                            continue
                        entities, events, event_arguments = extract_entity_event_from_sentence(elit_sent_res, tok_offset)
                        tok_offset += len(elit_sent_res['tok'])
                        for ent_t in entities:
                            ent_lemma = ' '.join(ent_t[1])
                            if ent_lemma not in entity_candidates:
                                entity_candidates.append(ent_lemma)
                        for evn_idx in range(len(events)):
                            trigger_lemma = ' '.join(events[evn_idx][1])
                            arguments_lemmas = sorted([' '.join(arg[1]) for arg in event_arguments[evn_idx]])
                            evn_key = (trigger_lemma, arguments_lemmas)
                            if evn_key not in event_candidates:
                                event_candidates.append(evn_key)
                    if len(entity_candidates) > 10 and len(event_candidates) > 10:
                        break
                # sort by freq, then loc
                line_output = {
                        'docno': docno,
                        'top10_entities': entity_candidates[:10],
                        'top10_events': [_[0] for _ in event_candidates[:10]],
                        'top10_event_arguments': [_[1] for _ in event_candidates[:10]],
                        }
                #print(line_output)
                fwrite.write(json.dumps(line_output) + '\n')
    fwrite.close()


def generate_salient_entity_event_by_location_SemanticScholar(elit_res_path: str, output_file_path: str) -> None:
    with open(elit_res_path) as fopen, open(output_file_path, 'w') as fwrite:
        for line in tqdm.tqdm(fopen):
            line_res = json.loads(line.strip())
            docno = line_res['doc_id']
            tok_offset = 0
            entity_candidates = []   # [phrase_lemma, ... ]
            event_candidates = []    # [(trigger_lemma, set(arguments)), ()]
            for elit_para_res in line_res['paper_abstract_elit_res']:
                for elit_sent_res in elit_para_res['sens']:
                    if len(elit_sent_res['dep']) <= 1:
                        tok_offset += len(elit_sent_res['tok'])
                        continue
                    entities, events, event_arguments = extract_entity_event_from_sentence(elit_sent_res, tok_offset)
                    tok_offset += len(elit_sent_res['tok'])
                    for ent_t in entities:
                        ent_lemma = ' '.join(ent_t[1])
                        if ent_lemma not in entity_candidates:
                            entity_candidates.append(ent_lemma)
                    for evn_idx in range(len(events)):
                        trigger_lemma = ' '.join(events[evn_idx][1])
                        arguments_lemmas = sorted([' '.join(arg[1]) for arg in event_arguments[evn_idx]])
                        evn_key = (trigger_lemma, arguments_lemmas)
                        if evn_key not in event_candidates:
                            event_candidates.append(evn_key)
                if len(entity_candidates) > 10 and len(event_candidates) > 10:
                    break
            # sort by freq, then loc
            line_output = {
                    'docno': docno,
                    'top10_entities': entity_candidates[:10],
                    'top10_events': [_[0] for _ in event_candidates[:10]],
                    'top10_event_arguments': [_[1] for _ in event_candidates[:10]],
                    }
            #print(line_output)
            fwrite.write(json.dumps(line_output) + '\n')


def generate_salient_entity_event_by_pagerank(elit_res_dir: str, output_file_path: str) -> None:
    """
    Word window size = 10, alpha = 0.85.
    """
    fwrite = open(output_file_path, 'w')
    for year_file in os.listdir(elit_res_dir):
        print('Now process year-%s...' % (year_file))
        with open(os.path.join(elit_res_dir, year_file)) as fopen:
            for line in tqdm.tqdm(fopen):
                line_res = json.loads(line.strip())
                docno = line_res['doc_id']
                tok_offset = 0
                all_nodes = []   # (hwidx, lemma, ntype)
                event_node_mentions = set()
                for elit_para_res in line_res['elit_res']:
                    for elit_sent_res in elit_para_res['sens']:
                        if len(elit_sent_res['dep']) <= 1:
                            tok_offset += len(elit_sent_res['tok'])
                            continue
                        entities, events, event_arguments = extract_entity_event_from_sentence(elit_sent_res, tok_offset)
                        for ent_t in entities:
                            hw_tidx = ent_t[2]
                            lemma = ' '.join(ent_t[1])
                            all_nodes.append((hw_tidx, lemma, 'entity'))
                        for trigger, arguments in zip(events, event_arguments):
                            hw_tidx = trigger[2]
                            trigger_lemma = ' '.join(trigger[1])
                            arguments_lemmas = sorted([' '.join(arg[1]) for arg in arguments])
                            arguments_lemmas = '\001'.join(arguments_lemmas)
                            event_lemma = '%s\001\001%s' % (trigger_lemma, arguments_lemmas)
                            all_nodes.append((hw_tidx, event_lemma, 'event'))
                            event_node_mentions.add(event_lemma)
                        tok_offset += len(elit_sent_res['tok'])
                # construct document level graph
                G = networkx.Graph()
                window_size = 10
                all_nodes = sorted(all_nodes, key=lambda _: _[0])
                for node_idx in range(0, len(all_nodes)-window_size+1):
                    node_i = all_nodes[node_idx]
                    for node_jdx in range(node_idx+1, node_idx+window_size):
                        node_j = all_nodes[node_jdx]
                        edge_weight = 1 / abs(node_j[0] - node_i[0])
                        if (node_i[1], node_j[1]) not in G.edges():
                            G.add_edge(node_i[1], node_j[1], weight=edge_weight)
                        else:
                            G[node_i[1]][node_j[1]]['weight'] += edge_weight
                pr = networkx.pagerank(G, alpha=0.85)
                top10_entities = []
                top10_events = []
                top10_event_arguments = []
                for node_mention, score in sorted(pr.items(), key=lambda _: -_[1]):
                    if node_mention not in event_node_mentions:
                        top10_entities.append(node_mention)
                    else:
                        trigger_mention, arguments_mentions = node_mention.split('\001\001')
                        top10_events.append(trigger_mention)
                        top10_event_arguments.append(arguments_mentions.split('\001'))
                    if len(top10_entities) >= 10 and len(top10_events) >= 10:
                        break
                line_output = {
                        'docno': docno,
                        'top10_entities': top10_entities,
                        'top10_events': top10_events,
                        'top10_event_arguments': top10_event_arguments,
                        }
                #print(line_output)
                fwrite.write(json.dumps(line_output) + '\n')
    fwrite.close()


def generate_salient_entity_event_by_pagerank_SemanticSchloar(elit_res_path: str, output_file_path: str) -> None:
    """
    Word window size = 10, alpha = 0.85.
    """
    fwrite = open(output_file_path, 'w')
    print('Now process %s...' % (elit_res_path))
    with open(elit_res_path) as fopen:
        for line in tqdm.tqdm(fopen):
            line_res = json.loads(line.strip())
            docno = line_res['doc_id']
            tok_offset = 0
            all_nodes = []   # (hwidx, lemma, ntype)
            event_node_mentions = set()
            for elit_para_res in line_res['paper_abstract_elit_res']:
                for elit_sent_res in elit_para_res['sens']:
                    if len(elit_sent_res['dep']) <= 1:
                        tok_offset += len(elit_sent_res['tok'])
                        continue
                    entities, events, event_arguments = extract_entity_event_from_sentence(elit_sent_res, tok_offset)
                    for ent_t in entities:
                        hw_tidx = ent_t[2]
                        lemma = ' '.join(ent_t[1])
                        all_nodes.append((hw_tidx, lemma, 'entity'))
                    for trigger, arguments in zip(events, event_arguments):
                        hw_tidx = trigger[2]
                        trigger_lemma = ' '.join(trigger[1])
                        arguments_lemmas = sorted([' '.join(arg[1]) for arg in arguments])
                        arguments_lemmas = '_'.join(arguments_lemmas)
                        event_lemma = '%s\001%s' % (trigger_lemma, arguments_lemmas)
                        all_nodes.append((hw_tidx, event_lemma, 'event'))
                        event_node_mentions.add(event_lemma)
                    tok_offset += len(elit_sent_res['tok'])
            # construct document level graph
            G = networkx.Graph()
            window_size = 10
            all_nodes = sorted(all_nodes, key=lambda _: _[0])
            for node_idx in range(0, len(all_nodes)-window_size+1):
                node_i = all_nodes[node_idx]
                for node_jdx in range(node_idx+1, node_idx+window_size):
                    node_j = all_nodes[node_jdx]
                    edge_weight = 1 / abs(node_j[0] - node_i[0])
                    if (node_i[1], node_j[1]) not in G.edges():
                        G.add_edge(node_i[1], node_j[1], weight=edge_weight)
                    else:
                        G[node_i[1]][node_j[1]]['weight'] += edge_weight
            pr = networkx.pagerank(G, alpha=0.85)
            top10_entities = []
            top10_events = []
            top10_event_arguments = []
            for node_mention, score in sorted(pr.items(), key=lambda _: -_[1]):
                if node_mention not in event_node_mentions:
                    top10_entities.append(node_mention)
                else:
                    trigger_mention, arguments_mentions = node_mention.split('\001')
                    top10_events.append(trigger_mention)
                    top10_event_arguments.append(arguments_mentions.split('_'))
                if len(top10_entities) >= 10 and len(top10_events) >= 10:
                    break
            line_output = {
                    'docno': docno,
                    'top10_entities': top10_entities,
                    'top10_events': top10_events,
                    'top10_event_arguments': top10_event_arguments,
                    }
            #print(line_output)
            fwrite.write(json.dumps(line_output) + '\n')
    fwrite.close()


if __name__ == '__main__':
    elit_res_dir = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/test_set_elit_dep_trees'
    output_file_path = './data/Xiong_SIGIR18/baseline_results/main_entities_from_title_advance.jsonlines'
    #extract_noun_phrases_in_title(elit_res_dir, output_file_path)
    output_file_path = './data/Xiong_SIGIR18/baseline_results/main_entities_from_title_first_sent.jsonlines'
    #extract_noun_phrases_in_title(elit_res_dir, output_file_path, add_first_sentence=True)


    elit_res_dir = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/test_set_elit_dep_trees'
    coref_cluster_dir_path = './data/Xiong_SIGIR18/test_set_spanbert_w_headline/spanbert_elit_index_matching_output'
    output_file_path = './data/Xiong_SIGIR18/baseline_results/main_entities_from_title_first_sent_dynamic_expand.jsonlines'
    #extract_initial_seed_entities_coref_version(elit_res_dir, coref_cluster_dir_path, output_file_path, add_first_sentence=True, dynamic_expansion_whole_content=True)
    output_file_path = './data/Xiong_SIGIR18/baseline_results/main_entities_from_title_first_sent_rank_add_centrality.jsonlines'
    #extract_initial_seed_entities_coref_version(elit_res_dir, coref_cluster_dir_path, output_file_path, add_first_sentence=True, dynamic_expansion_whole_content=False)


    # ------- Event --------
    output_file_path = './data/Liu_EMNLP18/baselines/main_events_from_title_firstsent_rank_by_entity_link.jsonlines'
    #extract_entity_event_in_title(elit_res_dir, output_file_path, add_first_sentence=True)
    output_file_path = './data/Liu_EMNLP18/baselines/main_events_from_whole_content_rank_by_freq.jsonlines'
    #produce_salient_events_by_headword_lemma_frequency(elit_res_dir, output_file_path)


    # ------- Pseudo Annotation -----
    #pseudo_annotate_NYT_corpus('data/Xiong_SIGIR18/test_set_elit_dep_trees_abstract', 'data/Xiong_SIGIR18/test_set_pseudo_annotation_by_parser.jsonlines')
    #pseudo_annotate_NYT_corpus('data/Xiong_SIGIR18/dev_set_elit_dep_trees_abstract', 'data/Xiong_SIGIR18/dev_set_pseudo_annotation_by_parser.jsonlines')
    #pseudo_annotate_SemanticScholar_corpus('data/SemanticScholar/test.elit_dep_trees.jsonlines', 'data/SemanticScholar/test_set_pseudo_annotation_by_parser.jsonlines')


    # ----------  Produce Baseline Results ---------
    output_path = 'data/Xiong_SIGIR18/baseline_results_hierarchy_extract/frequency.jsonlines'
    #generate_salient_entity_event_by_frequency(elit_res_dir, output_path)
    output_path = 'data/Xiong_SIGIR18/baseline_results_hierarchy_extract/location.jsonlines'
    #generate_salient_entity_event_by_location(elit_res_dir, output_path)
    output_path = 'data/Xiong_SIGIR18/baseline_results_hierarchy_extract/event_pagerank.jsonlines'
    #generate_salient_entity_event_by_pagerank(elit_res_dir, output_path)
    # ----------  Semantic Scholar ---------
    elit_res_path = 'data/SemanticScholar/test.elit_dep_trees.jsonlines'
    output_path = 'data/SemanticScholar/baselines/frequency.jsonlines'
    #generate_salient_entity_event_by_frequency_SemanticScholar(elit_res_path, output_path)
    output_path = 'data/SemanticScholar/baselines/location.jsonlines'
    #generate_salient_entity_event_by_location_SemanticScholar(elit_res_path, output_path)
    output_path = 'data/SemanticScholar/baselines/event_pagerank.jsonlines'
    generate_salient_entity_event_by_pagerank_SemanticSchloar(elit_res_path, output_path)


    # ----------  Produce Our Approach Results ---------
    #output_path = 'data/Xiong_SIGIR18/baseline_results_hierarchy_extract/initial_seed_position_weighted.jsonlines'
    output_path = 'data/Xiong_SIGIR18/baseline_results_hierarchy_extract/initial_seed_exp_inverse_pos.jsonlines'
    #extract_entity_event_seed_position_weighted_freq(elit_res_dir, output_path)
