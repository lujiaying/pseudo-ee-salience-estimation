import string
import numpy as np
from numpy.linalg import norm
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.tag.perceptron import PerceptronTagger
from nltk.stem import WordNetLemmatizer
from elit.component.tokenizer import EnglishTokenizer
from elit.component import POSFlairTagger
from elit.component.lemmatizer import EnglishLemmatizer
import tqdm

REPORTING_VERBS = {"argue", "claim", "say", "suggest", "tell",
                      "argues", "claims", "says", "suggests", "tells",
                      "argued", "claimed", "said", "suggested", "told",
                      "arguing", "claiming", "saying", "suggesting", "telling"}
LIGHTING_VERBS = {"do", "does", "did", "done", "doing",
        "make", "makes", "made", "making",
        "get", "gets", "got", "geting",
        "take", "takes", "took", "taken", "taking",
        "have", "has", "had", "having"}
AUXILIARY_VERBS = {"be", "being", "is", "was", "are", "were", "am",
        "dare", "dares", "dared", "daring",
        "do", "does", "did", "doing",
        "can", "could",
        "may", "might",
        "must", "need", "ought",
        "shall", "should",
        "will", "would"}
VAGUE_VERBS = REPORTING_VERBS.union(LIGHTING_VERBS).union(AUXILIARY_VERBS)

PRONOUNS = {'i', 'me', 'my', 'mine', 'myself',
        'you', 'your', 'yours', 'yourself',
        'he', 'him', 'his', 'himself',
        'she', 'her', 'hers', 'herself',
        'it', 'its', 'itself',
        'we', 'us', 'our', 'ours', 'ourselves',
        'yourselves',
        'they', 'them', 'their', 'theirs', 'themselves'}

TOKENIZER = TreebankWordTokenizer()
POS_TAGGER = PerceptronTagger()
LEMMATIZER = WordNetLemmatizer()

ELIT_LEMA_COMPONENTS = []


# --- Tree Util Functions ---
def generate_root_to_node_path(root, node):
    """
    Args:
        root: nltk.tree.Tree
        node: nltk.tree.Tree
    Returns:
        path: list
    """
    path = [root]
    invalid_node_in_path = []
    while path:
        #print(path)
        tree = path[-1]
        if tree == node:
            break
        if not isinstance(tree, nltk.tree.Tree):
            invalid_node_in_path.append(tree)
            path.pop()
            continue
        flag_all_sub_tree_invalid = True
        for sub_tree in tree:
            if sub_tree not in invalid_node_in_path:
                path.append(sub_tree)
                flag_all_sub_tree_invalid = False
                break
        if flag_all_sub_tree_invalid:
            invalid_node_in_path.append(tree)
            path.pop()
    return path


def generate_root_to_node_path_index_version(root, node_label):
    """
    Args:
        root: nltk.tree.Tree, each node is an index, no duplicated index in dep tree
        node_label: int, node label is an index
    Returns:
        tree_path: list of nltk.tree.Tree
        label_path: list of int
    """
    path = [root]
    invalid_node_in_path = []
    while path:
        #print(path)
        tree = path[-1]
        if (isinstance(tree, nltk.tree.Tree) and tree.label() == node_label) \
                or (isinstance(tree, int) and tree == node_label):
            break
        if not isinstance(tree, nltk.tree.Tree):
            invalid_node_in_path.append(tree)
            path.pop()
            continue
        flag_all_sub_tree_invalid = True
        for sub_tree in tree:
            if sub_tree not in invalid_node_in_path:
                path.append(sub_tree)
                flag_all_sub_tree_invalid = False
                break
        if flag_all_sub_tree_invalid:
            invalid_node_in_path.append(tree)
            path.pop()

    label_path = [node.label() if isinstance(node, nltk.tree.Tree) else node for node in path]
    return path, label_path


def calculate_node_pair_tree_distance(root, node1, node2, path1=None, path2=None):
    """
    Args:
        root: nltk.tree.Tree
        node1: nltk.tree.Tree
        node2: nltk.tree.Tree
        path1: list, optional
        path2: list, optional
    Returns:
        dist: int
    """
    if path1 is None:
        path1 = generate_root_to_node_path(root, node1)
    if path2 is None:
        path2 = generate_root_to_node_path(root, node2)
    assert max(len(path1), len(path2)) > 0
    for i in range(max(len(path1), len(path2))):
        if path1[i] != path2[i]:
            break
    return len(path1) - i + len(path2) - i


def calculate_node_pair_tree_distance_index_version(root, node1, node2):
    """
    Args:
        root: nltk.tree.Tree
        node1: int, node label
        node2: int, node label
    Returns:
        dist: int
    """
    t_path1, l_path1 = generate_root_to_node_path_index_version(root, node1)
    t_path2, l_path2 = generate_root_to_node_path_index_version(root, node2)
    assert max(len(l_path1), len(l_path2)) > 0
    if l_path1 == l_path2:
        return 0
    i = 0
    while i < min(len(l_path1), len(l_path2)):
        if l_path1[i] != l_path2[i]:
            break
        i += 1
    return len(l_path1) - i + len(l_path2) - i
# --- End Tree Util Functions ---

# --- Elit Result Util Functions ---
def get_mention_by_document_level_idx_tuple(elit_doc_res, idx_tuple):
    """
    Args:
        elit_doc_res: dict
        idx_tuple: tuple of int
    Returns:
        mention: string
    """
    cur_idx = 0
    mention = ""
    last_charoffset = None
    for paragraph_res in elit_doc_res:
        for sent_res in paragraph_res['sens']:
            sent_len = len(sent_res['tok'])
            if cur_idx + sent_len < idx_tuple[0]:
                cur_idx += sent_len
                continue   # continue sent_res in para
            elif cur_idx > idx_tuple[1]:
                cur_idx += sent_len
                return mention
            else:
                for offset in range(sent_len):
                    if cur_idx + offset > idx_tuple[1]:
                        return mention
                    elif cur_idx + offset == idx_tuple[0]:
                        mention += sent_res['tok'][offset]
                        last_charoffset = sent_res['off'][offset][1]
                    elif idx_tuple[1] >= cur_idx + offset > idx_tuple[0]:
                        if last_charoffset != sent_res['off'][offset][0]:
                            mention += ' '
                        mention += sent_res['tok'][offset]
                        last_charoffset = sent_res['off'][offset][1]
                cur_idx += sent_len
    return mention


def convert_elit_dep_to_old_version(dep_res):
    """
    In new version dep_res: root point to idx=0, the token idx starts from 1.
    In old version: root point to idx=len(sent), token idx starts from 0.
    Args:
        dep_res: list of tuple, [(idx, relation), ()]
    Returns:
        dep_res: list of tuple, [(idx, relation), ()]
    """
    sent_length = len(dep_res)
    # check if it is new version
    for (idx, rel) in dep_res:
        if rel == 'root' and idx == sent_length:
            # no need to convert
            return dep_res

    # idx rearrange
    dep_res = [(_[0]-1, _[1]) if _[0] > 0 else (sent_length, _[1]) for _ in dep_res]
    # multiple roots issue
    root_idx = None
    for idx, _ in enumerate(dep_res):
        if _[0] == sent_length and _[1] == 'root':
            root_idx = idx
    for idx, _ in enumerate(dep_res):
        if _[0] == sent_length and _[1] != 'root':
            dep_res[idx] = (root_idx, _[1])
    return dep_res
# --- End Elit Result Util Functions ---

GLOVE_PRETRAINED_EMB = {}
GLOVE_PRETRAINED_EMB_FILE = './data/pretrained_embs/glove.840B.300d.txt'

def cal_phrase_emb_cosine_sim_by_GloVe(phrase_i, phrase_j):
    """
    Args:
        phrase_i: list of tokens
        phrase_j: list of tokens
    Return:
        sim: float
    """
    global GLOVE_PRETRAINED_EMB
    if len(GLOVE_PRETRAINED_EMB) <= 0:
        global GLOVE_PRETRAINED_EMB_FILE
        with open(GLOVE_PRETRAINED_EMB_FILE) as fopen:
            for line in tqdm.tqdm(fopen):
                line_list = line.strip().split(' ')
                token = line_list[0]
                emb = np.array([float(_) for _ in line_list[1:]])
                GLOVE_PRETRAINED_EMB[token] = emb

    vec_i = np.zeros(300)
    for token in phrase_i:
        vec_i += GLOVE_PRETRAINED_EMB.get(token, GLOVE_PRETRAINED_EMB['UNK'])
    vec_i /= len(phrase_i)
    vec_j = np.zeros(300)
    for token in phrase_j:
        vec_j += GLOVE_PRETRAINED_EMB.get(token, GLOVE_PRETRAINED_EMB['UNK'])
    vec_j /= len(phrase_j)
    return np.dot(vec_i, vec_j) / (norm(vec_i) * norm(vec_j))


def load_event_vocabulary(event_verb_path: str = './data/event_vocabulary/event_verbs.txt', event_noun_path: str = './data/event_vocabulary/event_nouns.txt') -> (set, set):
    event_verbs = set()
    event_nouns = set()
    with open(event_verb_path) as fopen:
        for line in fopen:
            verb = line.strip()
            event_verbs.add(verb)

    with open(event_noun_path) as fopen:
        for line in fopen:
            noun = line.strip()
            event_nouns.add(noun)

    return event_verbs, event_nouns


def clean_nyt_abstract(abstract_raw: str) -> str:
    ori_len = len(abstract_raw)
    if abstract_raw.endswith('; photos (M)'):
        return abstract_raw[:ori_len - len('; photos (M)')]
    elif abstract_raw.endswith('; photo (M)'):
        return abstract_raw[:ori_len - len('; photo (M)')]
    elif abstract_raw.endswith(' (M)'):
        return abstract_raw[:ori_len - len(' (M)')]
    else:
        return abstract_raw


def convert_raw_mention_to_lemma(raw_mention: str) -> list:
    global TOKENIZER
    global POS_TAGGER
    global LEMMATIZER
    tokenizer = TOKENIZER
    tagger = POS_TAGGER
    lemmatizer = LEMMATIZER

    raw_mention = raw_mention.replace("’s", "'s").replace('—', '-')

    pos_tags = tagger.tag(tokenizer.tokenize(raw_mention))
    #print(pos_tags)
    lemmas = []
    for idx, (tok, pos_tag)in enumerate(pos_tags):
        if pos_tag not in ['PRP$', 'WP$'] and not pos_tag.isalnum():
            tok_lemma = ''
        elif pos_tag == 'DT':
            tok_lemma = ''
        elif pos_tag.startswith('NN') and pos_tag not in ['NNP', 'NNPS']:
            tok_lemma = lemmatizer.lemmatize(tok.lower(), pos='n')
        elif pos_tag.startswith('JJ'):
            tok_lemma = lemmatizer.lemmatize(tok.lower(), pos='a')
        elif pos_tag.startswith('VB'):
            tok_lemma = lemmatizer.lemmatize(tok.lower(), pos='v')
        else:
            tok_lemma = tok.lower()
        if len(tok_lemma) > 0:
            lemmas.append(tok_lemma)
    # post filter punctuation within lemma
    for idx, lemma in enumerate(lemmas):
        lemma = ''.join(ch for ch in lemma if ch not in string.punctuation)
        lemmas[idx] = lemma
    return lemmas


def convert_processed_mention_to_lemma_elit_version(toks: list, pos_tags: list, lems: list) -> list:
    """Helper function for elit_version and elit_batch_version"""
    assert len(toks) == len(pos_tags) == len(lems)

    lemmas = []
    for idx, lem in enumerate(lems):
        pos_tag = pos_tags[idx]
        if pos_tag == 'DT':
            continue
        elif lem in ['#crd#', '#ord#']:
            lemmas.append(toks[idx].lower())
        else: 
            lemmas.append(lem)
    # post filter punctuation within lemma
    lemmas_post = []
    for idx, lemma in enumerate(lemmas):
        lemma = ''.join(ch for ch in lemma if ch not in string.punctuation)
        if len(lemma) > 0:
            lemmas_post.append(lemma)
    return lemmas_post


def convert_raw_mention_to_lemma_elit_version(raw_mention: str) -> list:
    global ELIT_LEMA_COMPONENTS
    if len(ELIT_LEMA_COMPONENTS) <= 0:
        tokenizer = EnglishTokenizer()
        pos_tagger = POSFlairTagger()
        pos_tagger.load()
        lemmatizer = EnglishLemmatizer()
        ELIT_LEMA_COMPONENTS = [tokenizer, pos_tagger, lemmatizer]
        print('ELIT Lemmatization Components Load Done')
    components = ELIT_LEMA_COMPONENTS
    raw_mention = raw_mention.replace("’s", "'s").replace('—', '-')
    doc = raw_mention
    for c in components[:-1]:
        doc = c.decode(doc)

    lemmas = []
    # deal with special cases: one `JJ` token as the raw_mention should be `VBN` event trigger
    for _ in doc[0]['sens']:
        if len(_['tok']) == 1 and _['pos'][0] == 'JJ':
            _['pos'][0] = 'VBN'
    doc = components[-1].decode(doc)
    elit_sent_res = doc[0]['sens'][0]
    lemmas_post = convert_processed_mention_to_lemma_elit_version(elit_sent_res['tok'], elit_sent_res['pos'], elit_sent_res['lem'])
    return lemmas_post


def convert_raw_mention_to_lemma_elit_batch_version(raw_mentions: list) -> list:
    global ELIT_LEMA_COMPONENTS
    if len(ELIT_LEMA_COMPONENTS) <= 0:
        tokenizer = EnglishTokenizer()
        pos_tagger = POSFlairTagger()
        pos_tagger.load()
        lemmatizer = EnglishLemmatizer()
        ELIT_LEMA_COMPONENTS = [tokenizer, pos_tagger, lemmatizer]
        print('ELIT Lemmatization Components Load Done')
    components = ELIT_LEMA_COMPONENTS
    docs = [m.replace("’s", "'s").replace('—', '-') for m in raw_mentions]
    for c in components[:-1]:
        docs = c.decode(docs)
    #print(docs)
    # deal with special cases: one `JJ` token as the raw_mention should be `VBN` event trigger
    for doc in docs:
        for sent in doc['sens']:
            if len(sent['tok']) == 1 and sent['pos'][0] == 'JJ':
                sent['pos'][0] = 'VBN'
    docs = components[-1].decode(docs)
    #print(docs)
    lemmas_list = []
    for doc in docs:
        lemmas = []
        for _ in doc['sens']:
            lemmas = convert_processed_mention_to_lemma_elit_version(_['tok'], _['pos'], _['lem'])
            lemmas_list.append(lemmas)
    return lemmas_list


if __name__ == '__main__':
    raw_mentions = ['H.I.V', "John Robert Bond's shoes", "Six men", "Mr. Blumenthal’s staff", "left", "the first left boy", "attacked", "Germany's Chancellor-elect Gerhard Schroeder", "some students"]
    lemmas_list = convert_raw_mention_to_lemma_elit_batch_version(raw_mentions)
    print(lemmas_list)

    """
    raw_mention = "Germany's Chancellor-elect Gerhard Schroeder"
    lemmas = convert_raw_mention_to_lemma(raw_mention)
    print(lemmas)
    lemmas = convert_raw_mention_to_lemma_elit_version(raw_mention)
    print(lemmas)

    raw_mention = "cloud - normally miles high"
    lemmas = convert_raw_mention_to_lemma(raw_mention)
    print(lemmas)
    lemmas = convert_raw_mention_to_lemma_elit_version(raw_mention)
    print(lemmas)

    raw_mention = 'H.I.V'
    lemmas = convert_raw_mention_to_lemma(raw_mention)
    print(lemmas)
    lemmas = convert_raw_mention_to_lemma_elit_version(raw_mention)
    print(lemmas)

    raw_mention = 'N.C.A.A.'
    lemmas = convert_raw_mention_to_lemma(raw_mention)
    print(lemmas)
    lemmas = convert_raw_mention_to_lemma_elit_version(raw_mention)
    print(lemmas)

    raw_mention = "John Robert Bond's shoes"
    lemmas = convert_raw_mention_to_lemma(raw_mention)
    print(lemmas)
    lemmas = convert_raw_mention_to_lemma_elit_version(raw_mention)
    print(lemmas)

    raw_mention = "Six men"
    lemmas = convert_raw_mention_to_lemma(raw_mention)
    print(lemmas)
    lemmas = convert_raw_mention_to_lemma_elit_version(raw_mention)
    print(lemmas)

    raw_mention = "Mr. Blumenthal’s staff"
    lemmas = convert_raw_mention_to_lemma(raw_mention)
    print(lemmas)
    lemmas = convert_raw_mention_to_lemma_elit_version(raw_mention)
    print(lemmas)

    raw_mention = "left"
    lemmas = convert_raw_mention_to_lemma(raw_mention)
    print(lemmas)
    lemmas = convert_raw_mention_to_lemma_elit_version(raw_mention)
    print(lemmas)

    raw_mention = "the first left boy"
    lemmas = convert_raw_mention_to_lemma(raw_mention)
    print(lemmas)
    lemmas = convert_raw_mention_to_lemma_elit_version(raw_mention)
    print(lemmas)

    raw_mention = "some students"
    lemmas = convert_raw_mention_to_lemma(raw_mention)
    print(lemmas)
    lemmas = convert_raw_mention_to_lemma_elit_version(raw_mention)
    print(lemmas)

    raw_mention = "his wife is attacked"
    lemmas = convert_raw_mention_to_lemma(raw_mention)
    print(lemmas)
    lemmas = convert_raw_mention_to_lemma_elit_version(raw_mention)
    print(lemmas)
    """
