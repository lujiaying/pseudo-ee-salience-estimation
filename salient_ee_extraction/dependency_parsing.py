# -*- coding:utf-8 -*-
# Author: Jiaying Lu
# Date: 2019-02-10
import os
import json
import time
from elit.component import POSFlairTagger
from elit.component.dep.dependency_parser import DEPBiaffineParser
from elit.component.tokenizer import EnglishTokenizer
from elit.component.lemmatizer import EnglishLemmatizer
from collections import defaultdict
from nltk.tree import Tree
import tqdm
import xml.etree.ElementTree as ET
import mxnet

from .utils import REPORTING_VERBS, clean_nyt_abstract


def extract_verbal_predicates_prefilter(deps, tokens, pos_tags, do_prefilter=True, include_light_verb=False):
    """
    Verbal predicates are single word verbs or phrasal verbs.
    Since phrasal verbs are consist of verb and particle, no need for recursion
    All light, reporting or attritbuting verbs are filtered before adding to graph
    Args:
        deps: lsit of tuple, (dependent's idx, relation)
        tokens: list of str
        pos_tags: list of str
        do_prefilter: boolean, optional
        include_light_verb: boolean, optional
    Returns:
        all_verb_idxes: dict of {`int`-> `list of int`} for `head_word_idx`-> `verb_idxes'
    """
    if include_light_verb:
        valid_dep_rel_list = ['aux', 'raise', 'cop', 'attr', 'adv']
    else:
        valid_dep_rel_list = ['aux', 'lv', 'raise', 'cop', 'attr', 'adv']
    all_verb_idxes = {}
    # Extract all valid verbs
    for idx in range(len(pos_tags)):
        if pos_tags[idx].startswith('VB') and tokens[idx] not in ["''"]:
            if deps[idx][1] in valid_dep_rel_list:
                continue
            elif do_prefilter and tokens[idx].lower() in REPORTING_VERBS:
                continue
            else:
                all_verb_idxes[idx] = [idx]
    # Extract dependent verb particles
    for idx in range(len(pos_tags)):
        if pos_tags[idx] == 'RP':
            if deps[idx][0] in all_verb_idxes and deps[idx][1] == 'prt':
                head_idx = deps[idx][0]
                all_verb_idxes[head_idx].append(idx)
    return all_verb_idxes


def extract_noun_phrase_heuristically_recursion(deps, tokens, pos_tags, dep_node, 
        head_widx, parent_widx, all_NP_idxes):
    """
    1. Head Word is Noun or Pronoun
    2. POS tag of One-hop node of Noun Head Word should be in [DT, JJ, NN, POS, PRP, CD]
       2.1 DT with all relations are valid
       2.2 JJ should not come with relations in [conj]
       2.3 NN should come with relations in [com, poss]

    Args:
        deps: lsit of tuple, (dependent's idx, relation)
        tokens: list of str
        pos_tags: list of str
        dep_node: nltk.tree.Tree()
        head_widx: int, helper variable for recursion
        parent_widx: int, helper variable for recursion
        all_NP_idxes: dict of {`int`-> `list of int`} for `head_word_idx`-> `NP_idxes` 
    Returns:
        all_NP_idxes: dict of {`int`-> `list of int`} for `head_word_idx`-> `NP_idxes` 
    """
    # TODO: pos=`CD` cardinal number

    tidx = int(dep_node.label()) if isinstance(dep_node, Tree) else int(dep_node)
    cur_pos = pos_tags[tidx]
    cur_rel = deps[tidx][1]
    # No Head Word or conjunct rel or apposition rel
    if head_widx == None or (cur_rel == 'conj' and not cur_pos =='CD') or cur_rel == 'appo':
        # Head Word is either Noun or Pronoun
        if pos_tags[tidx].startswith('NN') or pos_tags[tidx] == 'PRP':
            head_widx = tidx
            parent_widx = tidx
            all_NP_idxes[head_widx] = [tidx]
        # Otherwise
        else:
            head_widx = None
            parent_widx = None
        if isinstance(dep_node, Tree):  # For non-leaf node
            for sub_tree in dep_node:
                all_NP_idxes = extract_noun_phrase_heuristically_recursion(deps, tokens, pos_tags, sub_tree,
                        head_widx, parent_widx, all_NP_idxes)
    else: # Head Word exists
        parent_pos = pos_tags[parent_widx]
        if parent_pos.startswith('NN'):
            if cur_pos == 'DT':
                all_NP_idxes[head_widx].append(tidx)
                parent_widx = tidx
            elif cur_pos.startswith('JJ'):
                if cur_rel == 'attr':
                    all_NP_idxes[head_widx].append(tidx)
                    parent_widx = tidx
                else:
                    parent_widx = None
                    head_widx = None
            elif cur_pos.startswith('NN'):
                if cur_rel in ['com', 'poss']:
                    all_NP_idxes[head_widx].append(tidx)
                    parent_widx = tidx
                else:
                    all_NP_idxes[tidx] = [tidx]
                    head_widx = tidx
                    parent_widx = tidx
            elif cur_pos == 'POS':
                all_NP_idxes[head_widx].append(tidx)
                parent_widx = tidx
            elif cur_pos.startswith('PRP'):  #`PRP`, `PRP$`
                if cur_rel in ['poss']:
                    all_NP_idxes[head_widx].append(tidx)
                    parent_widx = tidx
                else:
                    all_NP_idxes[tidx] = [tidx]
                    head_widx = tidx
                    parent_widx = tidx
            elif cur_pos == 'CD':
                all_NP_idxes[head_widx].append(tidx)
                parent_widx = tidx
            else:
                head_widx = None
                parent_widx = None
        elif parent_pos == 'DT':
            pass
            #print('cur_idx:%d, cur_pos:%s, with parent_pos=`DT`' % (tidx, cur_pos))
        elif parent_pos.startswith('JJ'):
            if cur_pos.startswith('RB'):
                all_NP_idxes[head_widx].append(tidx)
                parent_widx = tidx
            elif cur_pos.startswith('NN') or cur_pos.startswith('PRP'):
                all_NP_idxes[tidx] = [tidx]
                parent_widx = tidx
                head_widx = tidx
            else:
                parent_widx = None
                head_widx = None
                #print('cur_idx:%d, cur_pos:%s != `RB`, with parent_pos=`JJ`' % (tidx, cur_pos))
        elif parent_pos.startswith('PRP'):
            if cur_pos.startswith('NN') or cur_pos.startswith('PRP'):
                all_NP_idxes[tidx] = [tidx]
                parent_widx = tidx
                head_widx = tidx
            else:
                head_widx = None
                parent_widx = None
        elif parent_pos == 'CD':
            # TODO: to validate if needs some filters
            if cur_pos.startswith('RB'):   # RB not valid for CD's child
                head_widx = None
                parent_widx = None
            else:
                all_NP_idxes[head_widx].append(tidx)
                parent_widx = tidx
        else:
            head_widx = None
            parent_widx = None
        if isinstance(dep_node, Tree): # For non-leaf node
            for sub_tree in dep_node:
                all_NP_idxes = extract_noun_phrase_heuristically_recursion(deps, tokens, pos_tags, sub_tree,
                        head_widx, parent_widx, all_NP_idxes)
    return all_NP_idxes


def generate_NPs_from_all_NP_idxes(tokens, all_NP_idxes, char_offsets):
    """
    Args:
        tokens: list of str
        all_NP_idxes: dict of `int`-> `list of int` for `head_word_idx`-> `NP_idxes` 
        char_offsets: list of int tuple
    Returns:
        NPs: list of string
        NP_idxes: list of (list of int)
        NP_headword_idxes: list of int, len(NP_headword_idxes) == len(NPs)
    """
    #TODO: rename, maybe this is also applicable for verbs
    NPs = []
    NP_idxes = []
    NP_headword_idxes = []
    for head_widx, NP_idx_l in all_NP_idxes.items():
        NP_idx_l = sorted(NP_idx_l)
        NP_idxes.append(NP_idx_l)
        NP_headword_idxes.append(head_widx)
        NP = [tokens[idx] for idx in NP_idx_l]
        tok_start_idx, tok_end_idx = NP_idx_l[0], NP_idx_l[-1]
        NP = '%s' %(tokens[tok_start_idx])
        last_char_end_offset = char_offsets[tok_start_idx][1]
        for tok_idx in range(tok_start_idx+1, tok_end_idx+1):
            cur_char_start_offset = char_offsets[tok_idx][0]
            if last_char_end_offset != cur_char_start_offset:
                NP += ' ' 
            NP += tokens[tok_idx]
            last_char_end_offset = char_offsets[tok_idx][1]
        """
        # Use tokens and add space deliminator; may include noise
        NP = ' '.join(NP)
        # `'s' and '\u2019s' should be glued to the former noun
        NP = NP.replace(" 's", "'s").replace(" \u2019s", "\u2019s")
        """
        NPs.append(NP)
    return NPs, NP_idxes, NP_headword_idxes


def extract_noun_phrases_heuristically(deps, tokens, pos_tags, dep_root):
    """
    1. Head Word is Noun.
    2. Dependent is in [DT, JJ, CD, NN] and direct dependent relation

    Args:
        deps: lsit of tuple, (dependent's idx, relation)
        tokens: list of str
        pos_tags: list of str
        dep_root: nltk.tree.Tree()
    Returns:
        NPs: list of string
        NP_idxes: list of (list of int)
        NP_headword_idxes: list of int, len(NP_headword_idxes) == len(NPs)
    """
    to_traverse = [dep_root]
    NP_idxes = []
    NP_headword_idxes = []
    while to_traverse:
        cur_node = to_traverse.pop(0)
        if isinstance(cur_node, Tree):
            tidx = int(cur_node.label())
            if pos_tags[tidx].startswith('NN'):
                NP_idx = [tidx]
                for sub_tree in cur_node:
                    tidx_st = int(sub_tree.label()) if isinstance(sub_tree, Tree) else int(sub_tree)
                    pos_st = pos_tags[tidx_st]
                    rel_st = deps[tidx_st][1]
                    if pos_st.startswith('NN'):
                        # Rel=Compound Noun comprises NP
                        if rel_st == 'com':
                            NP_idx.append(tidx_st)
                        elif rel_st == 'poss':
                            NP_idx.append(tidx_st)
                            # typically contains nested `case`-`POS`
                            if isinstance(sub_tree, Tree):
                                NP_idx.extend([int(_) for _ in sub_tree.leaves()])
                        # Others are potential head word for NP
                        else:
                            to_traverse.append(sub_tree)
                    elif pos_st == 'DT':
                        NP_idx.append(tidx_st)
                    elif pos_st == 'JJ':
                        # relation should not be `conj`
                        if deps[tidx_st][1] == 'conj':
                            to_traverse.append(sub_tree)
                        else:
                            NP_idx.append(tidx_st)
                        # `JJ` may have a `RB` depedent
                        if isinstance(sub_tree, Tree):
                            for sub_sub_tree in sub_tree:
                                if isinstance(sub_sub_tree, Tree):
                                    to_traverse.append(sub_sub_tree)
                                else:
                                    tidx_st_st = int(sub_sub_tree)
                                    if pos_tags[tidx_st_st] == 'RB':
                                        NP_idx.append(tidx_st_st)
                    elif pos_st == 'POS':  # `'s`
                        if NP_idx and tokens[tidx_st] in ["'s", "\u2019s"]:
                            NP_idx.append(tidx_st)
                    else: 
                        to_traverse.append(sub_tree)
                NP_idxes.append(sorted(NP_idx))
                NP_headword_idxes.append(tidx)
            else:
                # no NP starts from the non-Noun head word
                for sub_tree in cur_node:
                    to_traverse.append(sub_tree)
        else:  # leaf node
            tidx = int(cur_node)
            if pos_tags[tidx].startswith('NN'):
                NP_idxes.append([tidx])
                NP_headword_idxes.append(tidx)
            elif pos_tags[tidx].startswith('PRP'):
                NP_idxes.append([tidx])
                NP_headword_idxes.append(tidx)
    #print(NP_idxes)
    #print(NP_headword_idxes)
    NPs = [] 
    for idxes in NP_idxes:
        NP = [tokens[idx] for idx in idxes]
        NP = ' '.join(NP)
        # `'s' and '\u2019s' should be glued to the former noun
        NP = NP.replace(" 's", "'s").replace(" \u2019s", "\u2019s")
        NPs.append(NP)
    #print(NPs)
    return NPs, NP_idxes, NP_headword_idxes


def construct_dependency_tree_from_parser_res(parser_res, tokens, pos_tag_res, char_offset):
    """
    Args:
        parser_res: list of tuple, len=L, (head_idx, relation)
        tokens: list of string, len=L
        pos_tag_res: list of string, len=L
        char_off_set: list of tuple, len=L, (char_start, char_end)
    Returns:
        dep_root: nltk.tree.Tree(), root for token index dep tree
        dep_tok_root: nltk.tree.Tree(), root for token dep tree
        dep_rel_root: nltk.tree.Tree(), root for relation dep tree
    """
    p_c_dict = defaultdict(list)  # parent_token_idx -> [children_token_idxes]
    c_p_dict = defaultdict(list)
    tidx_relation_dict = {}
    for idx, dep_t in enumerate(parser_res):
        c_p_dict[idx].append(dep_t[0])
        p_c_dict[dep_t[0]].append(idx)
        tidx_relation_dict[idx] = dep_t[1]

    #print(p_c_dict)
    #print(c_p_dict)

    #root_tidx = p_c_dict[len(parser_res)][0]
    tidx_layers = []
    #to_traverse = [(root_tidx, 0)]
    # may contain multiple root
    to_traverse = [(root_tidx, 0) for root_tidx in p_c_dict[len(parser_res)]]
    while to_traverse:
        tidx, layer = to_traverse.pop(0)
        if len(tidx_layers) <= layer:
            tidx_layers.append([])
        tidx_layers[layer].append(tidx)
        if tidx in p_c_dict:
            to_traverse.extend([(next_tidx, layer+1) for next_tidx in p_c_dict[tidx]])
    #print('tidx_layers', tidx_layers)

    tree_node_dict = {}
    for tidxs in reversed(tidx_layers):
        for tidx in tidxs:
            if tidx not in p_c_dict:
                tree_node_dict[tidx] = tidx
            else:
                children = [tree_node_dict[cidx] for cidx in p_c_dict[tidx]]
                node = Tree(tidx, children)
                tree_node_dict[tidx] = node
    #root = Tree.fromstring('%s'%(node))
    root = node
    #root.pretty_print()

    tree_node_dict = {}
    for tidxs in reversed(tidx_layers):
        for tidx in tidxs:
            if tidx not in p_c_dict:
                #tree_node_dict[tidx] = '%s\001#\001%s' % (tokens[tidx], tidx_relation_dict[tidx])
                tree_node_dict[tidx] = tokens[tidx]
            else:
                children = [tree_node_dict[cidx] for cidx in p_c_dict[tidx]]
                #node = Tree('%s\001#\001%s'%(tokens[tidx], tidx_relation_dict[tidx]), children)
                node = Tree(tokens[tidx], children)
                tree_node_dict[tidx] = node
    #dep_tok_root = Tree.fromstring('%s'%(node))
    dep_tok_root = node
    #dep_tok_root.pretty_print()

    tree_node_dict = {}
    for tidxs in reversed(tidx_layers):
        for tidx in tidxs:
            if tidx not in p_c_dict:
                tree_node_dict[tidx] = tidx_relation_dict[tidx]
            else:
                children = [tree_node_dict[cidx] for cidx in p_c_dict[tidx]]
                node = Tree(tidx_relation_dict[tidx], children)
                tree_node_dict[tidx] = node
    dep_rel_root = node
    #dep_rel_root = Tree.fromstring('%s'%(node))

    return root, dep_tok_root, dep_rel_root


def produce_elit_dep_trees(input_file, output_file):
    parser = DEPBiaffineParser()
    parser.load()
    pos_tagger = POSFlairTagger()
    pos_tagger.load()
    print('ELIT components load DONE')
    components = [EnglishTokenizer(), pos_tagger, parser]
    result_to_disk = []
    with open(input_file, encoding='utf-8') as fopen:
        for line in tqdm.tqdm(fopen):
            docs = [line.strip()]
            for c in components:
                docs = c.decode(docs)
            # docs[]['sens'][]['dep'] contains np.int64
            for doc in docs:
                for sent_dict in doc['sens']:
                    sent_dict['dep'] = [(int(_[0]), _[1]) for _ in sent_dict['dep']]
            result_to_disk.extend(docs)

    json.dump(result_to_disk, open(output_file, 'w', encoding='utf-8'))


def produce_elit_dep_trees_for_nyt_corpus(corpus_dir, output_dir):
    parser = DEPBiaffineParser()
    parser.load()
    pos_tagger = POSFlairTagger()
    pos_tagger.load()
    lemmatizer = EnglishLemmatizer()
    components = [EnglishTokenizer(), pos_tagger, lemmatizer, parser]
    print('ELIT components load DONE')

    #for dir_year in os.listdir(corpus_dir):
    #for dir_year in ['2004', '2005', '2006', '2007']:
    #for dir_year in ['2000', '2001', '2002', '2003']:
    #for dir_year in ['1997', '1998', '1999']:
    for dir_year in ['2003']:
        target_path = '%s/%s.json' % (output_dir, dir_year)
        print('Now dumping result to %s...' % (target_path))
        fwrite = open(target_path, 'w', encoding='utf-8')
        for file_doc in tqdm.tqdm(os.listdir(os.path.join(corpus_dir, dir_year))):
            empty_flag = True
            # parse xml file
            cur_path = os.path.join(corpus_dir, dir_year, file_doc)
            xml_tree = ET.parse(cur_path)
            root = xml_tree.getroot()
            docid = file_doc.split('.')[0]
            headline_node = root.find('./body/body.head/hedline/hl1')
            if headline_node != None:
                empty_flag = False
                headline = headline_node.text
            else:
                headline = ''
            content = []
            full_text_node = root.find("./body/body.content/block[@class='full_text']")
            if full_text_node != None:
                empty_flag = False
                for para in full_text_node.findall('p'):
                    content.append(para.text)
            # elit to produce dep tree
            file_res = {'headline': headline, 'content': content, 'doc_id': docid}
            if not empty_flag:
                docs = [headline] + content
                try:
                    for c in components:
                        docs = c.decode(docs)
                    for doc in docs:
                        doc['para_id'] = doc['doc_id']   # para_id=0 means headline
                        doc.pop('doc_id')
                        for sent_dict in doc['sens']:
                            sent_dict['dep'] = [(int(_[0]), _[1]) for _ in sent_dict['dep']]
                    file_res['elit_res'] = docs
                except mxnet.base.MXNetError:
                    file_res['elit_res'] = []
                    print('%s encounter error when using ELIT parsing' % (docid))
            else:
                file_res['elit_res'] = []
            fwrite.write(json.dumps(file_res) + '\n')
        fwrite.close()


def add_lem_for_elit_dep_result(elit_dep_res_dir: str, output_dir: str) -> None:
    from elit.structure import Document, Sentence, LEM, TOK, POS, SENS
    lemmatizer = EnglishLemmatizer()

    for dir_year in os.listdir(elit_dep_res_dir):
        cur_in_path = '%s/%s' % (elit_dep_res_dir, dir_year)
        target_path = '%s/%s' % (output_dir, dir_year)
        print('Now processing %s into %s' % (cur_in_path, target_path))
        with open(cur_in_path) as fopen, open(target_path, 'w') as fwrite:
            for line in tqdm.tqdm(fopen):
                line_dict = json.loads(line.strip())
                elit_res = line_dict['elit_res']
                docs = []
                for doc in elit_res:
                    doc_obj = Document()
                    doc_obj[SENS] = []
                    for sent in doc['sens']:
                        sent_obj = Sentence()
                        sent_obj[TOK] = sent['tok']
                        sent_obj[POS] = sent['pos']
                        doc_obj[SENS].append(sent_obj)
                    docs.append(doc_obj)
                docs = lemmatizer.decode(docs)
                for doc_idx, doc in enumerate(elit_res):
                    for sent_idx, sent in enumerate(doc['sens']):
                        sent['lem'] = docs[doc_idx]['sens'][sent_idx]['lem']
                fwrite.write(json.dumps(line_dict) + '\n')



def produce_elit_dep_trees_for_nyt_corpus_abstract(corpus_dir, output_dir):
    parser = DEPBiaffineParser()
    parser.load()
    pos_tagger = POSFlairTagger()
    pos_tagger.load()
    lemmatizer = EnglishLemmatizer()
    components = [EnglishTokenizer(), pos_tagger, lemmatizer, parser]
    print('ELIT components load DONE')

    for dir_year in os.listdir(corpus_dir):
        target_path = '%s/%s.jsonlines' % (output_dir, dir_year)
        print('Now dumping result to %s...' % (target_path))
        fwrite = open(target_path, 'w', encoding='utf-8')
        for file_doc in tqdm.tqdm(os.listdir(os.path.join(corpus_dir, dir_year))):
            # parse xml file
            cur_path = os.path.join(corpus_dir, dir_year, file_doc)
            xml_tree = ET.parse(cur_path)
            root = xml_tree.getroot()
            docid = file_doc.split('.')[0].zfill(7)
            abstract_dom = root.find('./body/body.head/abstract/p')
            abstract_cleaned = clean_nyt_abstract(abstract_dom.text)
            # elit to produce dep tree
            file_res = {'abstract': abstract_cleaned, 'doc_id': docid}
            if not len(abstract_cleaned) <= 0:
                docs = abstract_cleaned
                for c in components:
                    docs = c.decode(docs)
                for doc in docs:
                    doc['para_id'] = doc['doc_id']   # para_id=0 means headline
                    doc.pop('doc_id')
                    for sent_dict in doc['sens']:
                        sent_dict['dep'] = [(int(_[0]), _[1]) for _ in sent_dict['dep']]
                file_res['elit_res'] = docs
            else:
                file_res['elit_res'] = []
            fwrite.write(json.dumps(file_res) + '\n')
        fwrite.close()


def generate_NP_by_dep_parser(dep_parser_path, output_path):
    results_to_store = []
    dep_parser_res = json.load(open(dep_parser_path, encoding='utf-8'))
    sent_cnt = 0
    for doc in dep_parser_res:
        res_to_store = {}
        for res in doc['sens']:
            sent_cnt += 1
            dep_root, dep_tok_root, dep_rel_root = construct_dependency_tree_from_parser_res(res['dep'], res['tok'], res['pos'], res['off'])
            dep_root.pretty_print()
            dep_tok_root.pretty_print()
            dep_rel_root.pretty_print()
            #NPs, NP_idxes, NP_headword_idxes = extract_noun_phrases_heuristically(res['dep'], res['tok'], res['pos'], dep_root)
            all_NP_idxes = {}
            all_NP_idxes = extract_noun_phrase_heuristically_recursion(res['dep'], res['tok'], res['pos'], dep_root, None, None, all_NP_idxes)
            NPs, NP_idxes, NP_headword_idxes = generate_NPs_from_all_NP_idxes(res['tok'], all_NP_idxes)
            print(NPs)
            if not res_to_store:
                res_to_store = {
                        'tok': res['tok'],
                        'NPs': NPs
                        }
            else:
                res_to_store['tok'].extend(res['tok'])
                res_to_store['NPs'].extend(NPs)
        results_to_store.append(res_to_store)
    print('total sent_cnt:%d' % (sent_cnt))
    json.dump(results_to_store, open(output_path, 'w'), indent=2)


def produce_elit_dep_trees_for_SemanticScholar(input_path: str, output_path: str) -> None:
    print('[%s] Start process %s into %s' % (time.ctime(), input_path, output_path))
    parser = DEPBiaffineParser()
    parser.load()
    pos_tagger = POSFlairTagger()
    pos_tagger.load()
    lemmatizer = EnglishLemmatizer()
    components = [EnglishTokenizer(), pos_tagger, lemmatizer, parser]
    print('ELIT components load DONE')

    with open(input_path) as fopen, open(output_path, 'w') as fwrite:
        for line in tqdm.tqdm(fopen):
            line_dict = json.loads(line.strip())
            docno = line_dict['docno']
            paper_abstract = line_dict['paperAbstract']
            title = line_dict['title']
            file_res = {'doc_id': docno, 'paper_abstract': paper_abstract, 'title': title}
            # process title and abstract
            file_res['title_elit_res'] = []
            file_res['paper_abstract_elit_res'] = []
            # to save processing time
            if len(paper_abstract) > 0 and len(title) > 0:
                docs = [title, paper_abstract]
                for c in components:
                    docs = c.decode(docs)
                for doc in docs:
                    doc['para_id'] = doc['doc_id']   # para_id=0 means headline
                    doc.pop('doc_id')
                    for sent_dict in doc['sens']:
                        sent_dict['dep'] = [(int(_[0]), _[1]) for _ in sent_dict['dep']]
                    if doc['para_id'] == 0:
                        file_res['title_elit_res'].append(doc)
                    else:
                        file_res['paper_abstract_elit_res'].append(doc)
            elif len(paper_abstract) > 0:
                docs = paper_abstract
                for c in components:
                    docs = c.decode(docs)
                for doc in docs:
                    doc['para_id'] = doc['doc_id']   # para_id=0 means headline
                    doc.pop('doc_id')
                    for sent_dict in doc['sens']:
                        sent_dict['dep'] = [(int(_[0]), _[1]) for _ in sent_dict['dep']]
                file_res['paper_abstract_elit_res'] = docs
            elif len(title) > 0:
                docs = title
                for c in components:
                    docs = c.decode(docs)
                for doc in docs:
                    doc['para_id'] = doc['doc_id']   # para_id=0 means headline
                    doc.pop('doc_id')
                    for sent_dict in doc['sens']:
                        sent_dict['dep'] = [(int(_[0]), _[1]) for _ in sent_dict['dep']]
                file_res['title_elit_res'] = docs

            fwrite.write(json.dumps(file_res) + '\n')
        fwrite.close()
            

if __name__ == '__main__':
    #produce_elit_dep_trees('./tests/NYT_excerpt_for_noun_chunker.sentences', './tests/NYT_excerpt_for_noun_chunker.elit_dep_trees.json')
    
    #produce_elit_dep_trees_for_nyt_corpus('/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/test_set_full_text', '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/test_set_elit_dep_trees')
    #produce_elit_dep_trees_for_nyt_corpus('/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/train_set_full_text', '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/train_set_elit_dep_trees')   # May 10, 17:00, [2004, 2005, 2006, 2007]
    #produce_elit_dep_trees_for_nyt_corpus('/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/train_set_full_text', '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/train_set_elit_dep_trees')   # May 10, 17:00, [2000, 2001, 2002, 2003]
    #produce_elit_dep_trees_for_nyt_corpus('/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/train_set_full_text', '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/train_set_elit_dep_trees')   # May 23, 09:30, [2003]
    #produce_elit_dep_trees_for_nyt_corpus('/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/train_set_full_text', '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/train_set_elit_dep_trees')   # May 16, 22:00, [1997, 1998, 1999]
    #produce_elit_dep_trees_for_nyt_corpus('/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/dev_set_full_text', '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/dev_set_elit_dep_trees')
    # add lemmas for dep tree result
    #add_lem_for_elit_dep_result('data/Xiong_SIGIR18/test_set_elit_dep_trees', 'data/Xiong_SIGIR18/test_set_elit_dep_trees_add_lem')
    #add_lem_for_elit_dep_result('data/Xiong_SIGIR18/test_set_elit_dep_trees_abstract', 'data/Xiong_SIGIR18/test_set_elit_dep_trees_abstract_add_lem')
    #add_lem_for_elit_dep_result('data/Xiong_SIGIR18/dev_set_elit_dep_trees', 'data/Xiong_SIGIR18/dev_set_elit_dep_trees_add_lem')
    #add_lem_for_elit_dep_result('data/Xiong_SIGIR18/dev_set_elit_dep_trees_abstract', 'data/Xiong_SIGIR18/dev_set_elit_dep_trees_abstract_add_lem')

    #generate_NP_by_dep_parser('./tests/NYT_excerpt_for_noun_chunker.elit_dep_trees.json', './tests/NYT_excerpt_for_noun_chunker.elit_dep_trees.noun_phrases.json')

    #produce_elit_dep_trees_for_nyt_corpus_abstract('/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/test_set_full_text', '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/test_set_elit_dep_trees_abstract')
    #produce_elit_dep_trees_for_nyt_corpus_abstract('/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/dev_set_full_text', '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/dev_set_elit_dep_trees_abstract')


    # process Semantic Scholar Corpus
    #produce_elit_dep_trees_for_SemanticScholar('data/SemanticScholar/test', 'data/SemanticScholar/test.elit_dep_trees.jsonlines')
    #produce_elit_dep_trees_for_SemanticScholar('data/SemanticScholar/dev', 'data/SemanticScholar/dev.elit_dep_trees.jsonlines')
    #produce_elit_dep_trees_for_SemanticScholar('data/SemanticScholar/train_splitted0', 'data/SemanticScholar/train_splitted0.elit_dep_trees.jsonlines')   # May 10, 16:00
    #produce_elit_dep_trees_for_SemanticScholar('data/SemanticScholar/train_splitted1', 'data/SemanticScholar/train_splitted1.elit_dep_trees.jsonlines')   # May 10, 16:00
    #produce_elit_dep_trees_for_SemanticScholar('data/SemanticScholar/train_splitted2', 'data/SemanticScholar/train_splitted2.elit_dep_trees.jsonlines')   # Jun 6, 10:00
    #produce_elit_dep_trees_for_SemanticScholar('data/SemanticScholar/train_splitted3', 'data/SemanticScholar/train_splitted3.elit_dep_trees.jsonlines')   # Jun 6, 10:00

    produce_elit_dep_trees_for_SemanticScholar('data/SemanticScholar/train_splitted3_aa', 'data/SemanticScholar/train_splitted3_aa.elit_dep_trees.jsonlines')   # Jun 29, 17:00
    # produce_elit_dep_trees_for_SemanticScholar('data/SemanticScholar/train_splitted3_ab', 'data/SemanticScholar/train_splitted3_ab.elit_dep_trees.jsonlines')   # Jun 29, 17:00
