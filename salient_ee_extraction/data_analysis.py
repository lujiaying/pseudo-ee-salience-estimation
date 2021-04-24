import os
import json

import tqdm

from .hierarchy_extraction import extract_entity_event_from_sentence


def calculate_entity_event_stat(elit_res_dir: str) -> None:
    entity_total_cnt = 0
    event_total_cnt = 0
    doc_cnt = 0
    print('analysis %s' % (elit_res_dir))
    for year_file in os.listdir(elit_res_dir):
        print('Now process year-%s...' % (year_file))
        with open(os.path.join(elit_res_dir, year_file)) as fopen:
            for line in tqdm.tqdm(fopen):
                line_res = json.loads(line.strip())
                doc_cnt += 1
                tok_offset = 0
                for elit_para_res in line_res['elit_res']:
                    for elit_sent_res in elit_para_res['sens']:
                        if len(elit_sent_res['dep']) <= 1:
                            tok_offset += len(elit_sent_res['tok'])
                            continue
                        entities, events, event_arguments = extract_entity_event_from_sentence(elit_sent_res, tok_offset)
                        entity_total_cnt += len(entities)
                        event_total_cnt += len(events)
    print('In %s, total #,avg entity=%s,%s, #,avg event=%s,%s' % (elit_res_dir,
          entity_total_cnt, entity_total_cnt/doc_cnt,
          event_total_cnt, event_total_cnt/doc_cnt))


def calculate_entity_event_stat_SemanticScholar(elit_res_path: str) -> None:
    entity_total_cnt = 0
    event_total_cnt = 0
    doc_cnt = 0
    print('analysis %s' % (elit_res_path))
    with open(elit_res_path) as fopen:
        for line in tqdm.tqdm(fopen):
            line_res = json.loads(line.strip())
            doc_cnt += 1
            tok_offset = 0
            for elit_para_res in line_res['paper_abstract_elit_res']:
                for elit_sent_res in elit_para_res['sens']:
                    if len(elit_sent_res['dep']) <= 1:
                        tok_offset += len(elit_sent_res['tok'])
                        continue
                    entities, events, event_arguments = extract_entity_event_from_sentence(elit_sent_res, tok_offset)
                    entity_total_cnt += len(entities)
                    event_total_cnt += len(events)
    print('In %s, total #,avg entity=%s,%s, #,avg event=%s,%s' % (elit_res_path,
          entity_total_cnt, entity_total_cnt/doc_cnt,
          event_total_cnt, event_total_cnt/doc_cnt))


if __name__ == '__main__':
    dev_set_NYT = 'data/Xiong_SIGIR18/dev_set_elit_dep_trees/'
    # calculate_entity_event_stat(dev_set_NYT)
    train_set_NYT = 'data/Xiong_SIGIR18/train_set_elit_dep_trees/'
    # calculate_entity_event_stat(train_set_NYT)
    test_set_NYT = 'data/Xiong_SIGIR18/test_set_elit_dep_trees/'
    # calculate_entity_event_stat(test_set_NYT)

    dev_set_SS = 'data/SemanticScholar/dev.elit_dep_trees.jsonlines'
    # calculate_entity_event_stat_SemanticScholar(dev_set_SS)
    test_set_SS = 'data/SemanticScholar/test.elit_dep_trees.jsonlines'
    # calculate_entity_event_stat_SemanticScholar(test_set_SS)
    train_set_SS = 'data/SemanticScholar/train.elit_dep_trees.jsonlines.head600K'
    calculate_entity_event_stat_SemanticScholar(train_set_SS)
