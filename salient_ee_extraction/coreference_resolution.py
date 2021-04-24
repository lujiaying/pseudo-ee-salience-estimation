import os
import json
import tqdm
import xml.etree.ElementTree as ET

coref_resolver = None
TOKENIZER = None


def load_bert_tokenizer():
    global TOKENIZER
    from transformers import BertTokenizer
    if TOKENIZER is None:
        TOKENIZER = BertTokenizer.from_pretrained('bert-large-cased')
    return TOKENIZER


def convert_elit_tokens_to_spanbert_coref_format(elit_doc_res, docno, segment_limit=512):
    """
    Args:
        elit_doc_res: dict
        segment_limit: int
    Returns:
        coref_input_dict: dict
    """
    tokenizer = load_bert_tokenizer()
    bert_toks_list = []
    subtoken_map = []
    sentence_map = []
    elit_tok_idx = 0
    elit_sen_idx = 0
    for elit_para_res in elit_doc_res:
        for sent_res in elit_para_res['sens']:
            elit_toks = sent_res['tok']
            bert_toks = []
            for tok in elit_toks:
                tok_pieces = tokenizer.tokenize(tok)  # return list of string
                for i in range(len(tok_pieces)):
                    subtoken_map.append(elit_tok_idx)
                    sentence_map.append(elit_sen_idx)
                bert_toks.extend(tok_pieces)
                elit_tok_idx += 1
            bert_toks_list.append(bert_toks)
            elit_sen_idx += 1

    # merge sentences into one segment, but not exceed len limit
    bert_segment_list = []
    cur_segment = ["[CLS]"]  # segment starts with [CLS], ends with [SEP]
    cur_subtok_idx = 0
    subtok_idx_to_add_special_symbol = []  # [(cls, sep), ()]
    for bert_toks in bert_toks_list:
        if len(cur_segment) + len(bert_toks) + 1 < segment_limit - 10:  # incase OOM happens
            cur_segment.extend(bert_toks)
        else:
            cur_segment.append('[SEP]')
            subtok_idx_to_add_special_symbol.append((cur_subtok_idx, cur_subtok_idx+len(cur_segment)-2))
            cur_subtok_idx += (len(cur_segment) - 2)
            bert_segment_list.append(cur_segment)
            cur_segment = ["[CLS]"] + bert_toks
    #print('final elit tok idx:%d' % (elit_tok_idx))
    #print('bert_toks_list len=%d'%(sum(len(_) for _ in bert_toks_list)), bert_toks_list)
    #print('subtoken_map len=%d'%(len(subtoken_map)), subtoken_map)
    #print('sentence_map len=%d'%(len(sentence_map)), sentence_map)
    ## deal with remainder
    if cur_segment != ["[CLS]"]:
        cur_segment.append('[SEP]')
        bert_segment_list.append(cur_segment)
        subtok_idx_to_add_special_symbol.append((cur_subtok_idx, cur_subtok_idx+len(cur_segment)-2))
        cur_subtok_idx += (len(cur_segment) - 2)
    #print('bert_segment_list len=%d'%(sum(len(_) for _ in bert_segment_list)), bert_segment_list)
    #print('each segment len:', [len(_) for _ in bert_segment_list])
    #print('subtok_idx_to_add_special_symbol:', subtok_idx_to_add_special_symbol)
    for idx_CLS, idx_SEP in reversed(subtok_idx_to_add_special_symbol):
        # subtoken map
        subtoken_idx = subtoken_map[idx_SEP-1]
        #subtoken_map.insert(idx_SEP, '%s-[SEP]'%(subtoken_idx))
        subtoken_map.insert(idx_SEP, subtoken_idx)
        subtoken_idx = subtoken_map[idx_CLS]
        #subtoken_map.insert(idx_CLS, '%s-[CLS]'%(subtoken_idx))
        subtoken_map.insert(idx_CLS, subtoken_idx)
        # sentence map
        sentence_idx = sentence_map[idx_SEP-1]
        sentence_map.insert(idx_SEP, sentence_idx)
        sentence_idx = sentence_map[idx_CLS]
        sentence_map.insert(idx_CLS, sentence_idx)
    #print('after subtoken_map len=%d'%(len(subtoken_map)), subtoken_map)
    #print('after sentence_map len=%d'%(len(sentence_map)), sentence_map)
    speakers = [['-' if (idx not in [0, len(sublist)-1]) else '[SPL]' for idx, _ in enumerate(sublist)] for sublist in bert_segment_list]

    spanbert_coref_input_dict = {
            "clusters":[],
            "doc_key": "nw/%s"%(docno),
            "sentences": bert_segment_list,
            "speakers": speakers,
            "sentence_map": sentence_map,
            "subtoken_map": subtoken_map,
            }
    return spanbert_coref_input_dict


def prepare_spanbert_input_for_nyt_corpus(elit_processed_dir, output_dir, include_headline=False):
    """
    Args:
        elit_processed_dir: string
        output_dir: string
    """
    output_dir_complete = '%s/processed_input' % (output_dir)
    if not os.path.exists(output_dir_complete):
        os.makedirs(output_dir_complete)
    for file_year in os.listdir(elit_processed_dir):
        file_path = os.path.join(elit_processed_dir, file_year)
        target_path = os.path.join(output_dir_complete, file_year)
        print('Now dumping result to %s...' % (target_path))
        with open(file_path) as fopen, open(target_path, 'w') as fwrite:
            for line in tqdm.tqdm(fopen):
                elit_res_per_line = json.loads(line.strip())
                docno = elit_res_per_line['doc_id']
                if not include_headline:
                    # headline stores in para#0
                    elit_doc_res = elit_res_per_line['elit_res'][1:]
                else:
                    elit_doc_res = elit_res_per_line['elit_res']
                coref_input_dict = convert_elit_tokens_to_spanbert_coref_format(elit_doc_res, docno)
                fwrite.write(json.dumps(coref_input_dict)+'\n')


def prepare_spanbert_input_for_semanticscholar(elit_processed_path: str, output_path: str) -> None:
    with open(elit_processed_path) as fopen, open(output_path, 'w') as fwrite:
        for line in tqdm.tqdm(fopen):
            elit_res_per_line = json.loads(line.strip())
            docno = elit_res_per_line['doc_id']
            elit_doc_res = elit_res_per_line['paper_abstract_elit_res']
            coref_input_dict = convert_elit_tokens_to_spanbert_coref_format(elit_doc_res, docno)
            fwrite.write(json.dumps(coref_input_dict)+'\n')


def convert_spanbert_coref_output_to_elit_idx_res(spanbert_coref_output):
    """
    Args:
        spanbert_coref_output: dict
    Returns:
        coref_elit_idx_res: dict
    """
    docno = spanbert_coref_output['doc_key'].split('/')[1]
    # convert subtoken idx to elit token idx
    predicted_clusters_in_elit_tidx = []
    subtoken_map = spanbert_coref_output['subtoken_map']
    for cluster in spanbert_coref_output['predicted_clusters']:
        elit_tidx_cluster = []
        for mention_stidx_t in cluster:
            tidx_tuple = [subtoken_map[mention_stidx_t[0]], subtoken_map[mention_stidx_t[1]]]
            elit_tidx_cluster.append(tidx_tuple)
        predicted_clusters_in_elit_tidx.append(elit_tidx_cluster)
    coref_elit_idx_res = {
            'docno': docno,
            'predicted_clusters': predicted_clusters_in_elit_tidx
            }
    return coref_elit_idx_res


def prepare_spanbert_output_index_matching_for_nyt_corpus(spanbert_out_dir, spanbert_index_matching_out_dir):
    """
    Convert spanbert output format to elit index
    Args:
        spanbert_out_dir: string
        spanbert_index_matching_out_dir: string
    """
    if not os.path.exists(spanbert_index_matching_out_dir):
        os.makedirs(spanbert_index_matching_out_dir)
    for file_year in os.listdir(spanbert_out_dir):
        input_path = os.path.join(spanbert_out_dir, file_year)
        output_path = os.path.join(spanbert_index_matching_out_dir, file_year)
        print('Now dumping result to %s...' % (output_path))
        with open(input_path) as fopen, open(output_path, 'w') as fwrite:
            for line in tqdm.tqdm(fopen):
                spanbert_raw_out_dict = json.loads(line.strip())
                coref_elit_idx_res = convert_spanbert_coref_output_to_elit_idx_res(spanbert_raw_out_dict)
                fwrite.write(json.dumps(coref_elit_idx_res) + '\n')


def prepare_spanbert_output_index_matching_for_semantic_scholar(spanbert_out_path, spanbert_index_matching_out_path):
    """
    Convert spanbert output format to elit index
    Args:
        spanbert_out_path: string
        spanbert_index_matching_out_path: string
    """
    with open(spanbert_out_path) as fopen, open(spanbert_index_matching_out_path, 'w') as fwrite:
        for line in tqdm.tqdm(fopen):
            spanbert_raw_out_dict = json.loads(line.strip())
            coref_elit_idx_res = convert_spanbert_coref_output_to_elit_idx_res(spanbert_raw_out_dict)
            fwrite.write(json.dumps(coref_elit_idx_res) + '\n')


def load_coref_resolver():
    global coref_resolver
    import allennlp
    import allennlp.pretrained
    if coref_resolver is None:
        coref_resolver = allennlp.pretrained.neural_coreference_resolution_lee_2017()
        coref_resolver._model = coref_resolver._model.cuda()
    return coref_resolver


def get_coref_resolution(raw_article):
    """
    Args:
        raw_article: str
    Returns:
        result: dict, keys -
                'document': list of tokens
                'clusters': 2d-list
    """
    coref_resolver = load_coref_resolver()
    result = coref_resolver.predict(document=raw_article)
    return result


def produce_coref_results_for_nyt_corpus(corpus_dir, output_dir, include_headline=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for dir_year in os.listdir(corpus_dir):
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
            if include_headline:
                content = [headline]
            else:
                content = []
            full_text_node = root.find("./body/body.content/block[@class='full_text']")
            if full_text_node != None:
                empty_flag = False
                for para in full_text_node.findall('p'):
                    content.append(para.text)

            file_res = {'doc_id': docid}
            # current version: title exclude
            # each paragraph is split by '\n'. when merging with other systems, need attention.
            content_text = '\n'.join(content)
            if len(content_text) <= 0:
                file_res['coref_res'] = []
            else:
                try:
                    file_res['coref_res'] = get_coref_resolution(content_text)
                except Exception as e:
                    print('DocID=%s, RuntimeError: %s' % (docid, e))
                    file_res['coref_res'] = []
            fwrite.write(json.dumps(file_res) + '\n')
        fwrite.close()
    

if __name__ == '__main__':
    """
    #test_set_dir = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/test_set_full_text'
    #output_dir = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/test_set_allennlp_coref_lee_2017'  # not include headline
    train_set_dir = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/train_set_full_text'
    output_dir = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/train_set_allennlp_coref_lee_2017'  # not include headline
    #produce_coref_results_for_nyt_corpus(test_set_dir, output_dir)
    """

    """
    # Produce result for include headline
    test_set_dir = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/test_set_full_text'
    output_dir = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/test_set_allennlp_coref_lee_2017_w_headline'
    produce_coref_results_for_nyt_corpus(test_set_dir, output_dir, include_headline=True)
    """

    # prepare corpus required by spanbert-coref
    elit_processed_dir = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/test_set_elit_dep_trees'
    spanbert_data_dir = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/test_set_spanbert_wo_headline'  # without headline
    #prepare_spanbert_input_for_nyt_corpus(elit_processed_dir, spanbert_data_dir, include_headline=False)
    spanbert_data_dir = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/test_set_spanbert_w_headline'  # with headline
    #prepare_spanbert_input_for_nyt_corpus(elit_processed_dir, spanbert_data_dir, include_headline=True)
    elit_processed_dir = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/dev_set_elit_dep_trees'
    spanbert_data_dir = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/dev_set_spanbert_w_headline'  # with headline
    #prepare_spanbert_input_for_nyt_corpus(elit_processed_dir, spanbert_data_dir, include_headline=True)
    elit_processed_dir = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/train_set_elit_dep_trees'
    spanbert_data_dir = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/train_set_spanbert_w_headline'  # with headline
    #prepare_spanbert_input_for_nyt_corpus(elit_processed_dir, spanbert_data_dir, include_headline=True)
    # ----- Semantic Scholar -------
    elit_processed_path = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/SemanticScholar/dev.elit_dep_trees.jsonlines'
    spanbert_data_path = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/SemanticScholar/dev_set_spanbert/processed_input.jsonlines'
    elit_processed_path = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/SemanticScholar/test.elit_dep_trees.jsonlines'
    spanbert_data_path = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/SemanticScholar/test_set_spanbert/processed_input.jsonlines'
    elit_processed_path = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/SemanticScholar/train.elit_dep_trees.jsonlines'
    spanbert_data_path = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/SemanticScholar/train_set_spanbert/processed_input.jsonlines'
    # makeup for missing splits
    elit_processed_path = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/SemanticScholar/train_splitted2.elit_dep_trees.jsonlines'
    spanbert_data_path = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/SemanticScholar/train_set_spanbert/processed_input.splitted2.jsonlines'
    prepare_spanbert_input_for_semanticscholar(elit_processed_path, spanbert_data_path)
    elit_processed_path = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/SemanticScholar/train_splitted3.elit_dep_trees.jsonlines'
    spanbert_data_path = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/SemanticScholar/train_set_spanbert/processed_input.splitted3.jsonlines'
    prepare_spanbert_input_for_semanticscholar(elit_processed_path, spanbert_data_path)

    # prepare index matching spanbert output
    spanbert_out_dir = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/test_set_spanbert_wo_headline/spanbert_raw_output'
    spanbert_index_matching_out_dir = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/test_set_spanbert_wo_headline/spanbert_elit_index_matching_output'
    spanbert_out_dir = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/test_set_spanbert_w_headline/spanbert_raw_output'
    spanbert_index_matching_out_dir = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/test_set_spanbert_w_headline/spanbert_elit_index_matching_output'
    spanbert_out_dir = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/dev_set_spanbert_w_headline/spanbert_raw_output'
    spanbert_index_matching_out_dir = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/dev_set_spanbert_w_headline/spanbert_elit_index_matching_output'
    spanbert_out_dir = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/train_set_spanbert_w_headline/spanbert_raw_output'
    spanbert_index_matching_out_dir = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/Xiong_SIGIR18/train_set_spanbert_w_headline/spanbert_elit_index_matching_output'
    #prepare_spanbert_output_index_matching_for_nyt_corpus(spanbert_out_dir, spanbert_index_matching_out_dir)
    ## Semantic Scholar
    spanbert_out_path = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/SemanticScholar/dev_set_spanbert/spanbert_raw_output.jsonlines'
    spanbert_index_matching_out_path = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/SemanticScholar/dev_set_spanbert/spanbert_elit_index_matching_output.jsonlines'
    spanbert_out_path = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/SemanticScholar/test_set_spanbert/spanbert_raw_output.jsonlines'
    spanbert_index_matching_out_path = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/SemanticScholar/test_set_spanbert/spanbert_elit_index_matching_output.jsonlines'
    spanbert_out_path = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/SemanticScholar/train_set_spanbert/spanbert_raw_output.jsonlines'
    spanbert_index_matching_out_path = '/home/jlu229/Salient-Entity-Event-Hierarchy-Extraction/data/SemanticScholar/train_set_spanbert/spanbert_elit_index_matching_output.jsonlines'
    #prepare_spanbert_output_index_matching_for_semantic_scholar(spanbert_out_path, spanbert_index_matching_out_path)
