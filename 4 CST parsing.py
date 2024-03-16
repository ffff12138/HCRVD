<<<<<<< HEAD
import pandas as pd
import os
from tqdm.auto import tqdm
import re
import argparse
tqdm.pandas()

def parse_options():
    parser = argparse.ArgumentParser(description='Parsing CST.')
    parser.add_argument('-i', '--input', help='The dir path of input', type=str, default='./data/pdgs')
    args = parser.parse_args()
    return args

class CSTParser:
    def __init__(self, ratio, root: str):
        self.ratio = ratio
        self.root = root
        self.sources = None
        self.train = None
        self.dev = None
        self.test = None
        self.size = None

    def get_parsed_source(self, input_file: str ):

        input_file_path = os.path.join(self.root, input_file)
        from tree_sitter import Language, Parser
        C_LANGUAGE = Language('./build_parser/c.so', 'c')
        lang = {
            "c": C_LANGUAGE
        }
        parser = Parser()
        parser.set_language(lang['c'])

        def truncatCode(code):
            trunc_code = []
            token_num = 0
            code = code.split('\n')  #
            for line in code:
                tmp_tokens = []
                line = line.split()
                for token in line:
                    tmp_tokens.append(token)
                    token_num += 1

                str = " ".join(tmp_tokens)
                if not len(line):
                    continue
                trunc_code.append(str)

            return trunc_code

        def getNodeValue(code, start_point, end_point):
            if start_point[0] == end_point[0]:
                value = code[start_point[0]][start_point[1]:end_point[1]]
            else:
                value = ""
                value += code[start_point[0]][start_point[1]:]
                for i in range(start_point[0] + 1, end_point[0]):
                    value += code[i]
                value += code[end_point[0]][:end_point[1]]
            return value

        def gettoken(node, code):
            if node.child_count == 0:
                token = getNodeValue(code, node.start_point, node.end_point)
            else:
                if node.type == 'ERROR' and getNodeValue(code, node.children[0].start_point,
                                                         node.children[0].end_point) == 'FUN1':
                    token = 'method'
                elif node.type == 'ERROR' and getNodeValue(code, node.children[0].start_point,
                                                           node.children[0].end_point) == 'void':
                    token = 'method_return'
                else:
                    token = node.type
            return token

        def tree_contructor(node, code):
            token = gettoken(node, code)
            if token == '':
                return
            result = [token]
            children = node.children
            for child in children:
                result.append(tree_contructor(child, code))
            return result

        def tree_parser(code):
            tree = []
            for code_fragment in code:
                if code_fragment != 'FUN1' and code_fragment != 'void':
                    code_fragment += ';'
                code_fragment = truncatCode(code_fragment)
                tree_fragment = parser.parse(bytes("\n".join(code_fragment), "utf8"))
                tree.append(tree_contructor(tree_fragment.root_node, code_fragment))
            return tree

        source = pd.read_pickle(input_file_path)
        source.columns = ['id', 'code', 'label','adj']
        source['code'] = source['code'].progress_apply(tree_parser)
        self.sources = source

    def split_data(self):
        data = self.sources
        data_num = len(data)
        ratios = [int(r) for r in self.ratio.split(':')]
        train_split = int(ratios[0]/sum(ratios)*data_num)
        val_split = train_split + int(ratios[1]/sum(ratios)*data_num)
        data = data.sample(frac=1, random_state=666)
        train = data.iloc[:train_split]
        dev = data.iloc[train_split:val_split]
        test = data.iloc[val_split:]

        def check_or_create(path):
            if not os.path.exists(path):
                os.mkdir(path)

        train_path = self.root + 'train/'
        check_or_create(train_path)
        dev_path = self.root + 'dev/'
        check_or_create(dev_path)
        test_path = self.root + 'test/'
        check_or_create(test_path)
        self.train = train
        self.dev = dev
        self.test = test

    def dictionary_and_embedding(self, size):
        self.size = size
        if not os.path.exists(self.root+'train/embedding'):
            os.mkdir(self.root+'train/embedding')

        def get_sequences(tree_list, sequence):
            for tree in tree_list:
                if isinstance(tree, str):
                    sequence.append(tree)
                elif tree is None:
                    continue
                else:
                    get_sequences(tree, sequence)
            return sequence

        def trees_to_sequences(tree_list):
            sequence = []
            get_sequences(tree_list, sequence)
            return sequence

        trees = self.train
        corpus = trees['code'].apply(trees_to_sequences)


        from gensim.models.word2vec import Word2Vec
        w2v = Word2Vec(corpus, size=size, workers=16, sg=1, min_count=3,sorted_vocab=1)
        w2v.save(self.root+'train/embedding/node_w2v_' + str(size))

    def generate_block_seqs(self, data, part):
        from gensim.models.word2vec import Word2Vec
        word2vec = Word2Vec.load(self.root+'train/embedding/node_w2v_' + str(self.size)).wv
        vocab = word2vec.vocab
        max_token = word2vec.syn0.shape[0]

        def get_embbeding_sequences(tree, emb_list):
            for sub_tree in tree:
                if isinstance(tree, list):
                    if sub_tree is None:
                        continue
                    else:
                        tmp = []
                        emb_list.append(get_embbeding_sequences(sub_tree, tmp))
                else:
                    emb = vocab[tree].index if tree in vocab else max_token
                    return emb
            return emb_list

        def tree_embbeding(tree_list):
            embbeding_sequence = []
            for tree in tree_list:
                sub_embbeding_sequences = []
                get_embbeding_sequences(tree, sub_embbeding_sequences)
                embbeding_sequence.append(sub_embbeding_sequences)
            return embbeding_sequence

        trees = data
        trees['code'] = trees['code'].apply(tree_embbeding)
        trees.to_pickle(self.root+part+'/CPs.pkl')


    def run(self):
        print('parse source code...')
        self.get_parsed_source(input_file='PDGs.pkl')
        print('split data...')
        self.split_data()
        print('train word embedding...')
        self.dictionary_and_embedding(128)
        print('generate CP data...')
        self.generate_block_seqs(self.train, 'train')
        self.generate_block_seqs(self.dev, 'dev')
        self.generate_block_seqs(self.test, 'test')


def main():
    args = parse_options()
    input_path = args.input
    if input_path[-1] == '/':
        input_path = input_path
    else:
        input_path += '/'

    ppl = CSTParser('4:1:0', input_path)
    ppl.run()

if __name__ == '__main__':
    main()
=======
import pandas as pd
import os
from tqdm.auto import tqdm
import re
import argparse
tqdm.pandas()

def parse_options():
    parser = argparse.ArgumentParser(description='Parsing CST.')
    parser.add_argument('-i', '--input', help='The dir path of input', type=str, default='./data/pdgs')
    args = parser.parse_args()
    return args

class CSTParser:
    def __init__(self, ratio, root: str):
        self.ratio = ratio
        self.root = root
        self.sources = None
        self.train = None
        self.dev = None
        self.test = None
        self.size = None

    def get_parsed_source(self, input_file: str ):

        input_file_path = os.path.join(self.root, input_file)
        from tree_sitter import Language, Parser
        C_LANGUAGE = Language('./build_parser/c.so', 'c')
        lang = {
            "c": C_LANGUAGE
        }
        parser = Parser()
        parser.set_language(lang['c'])

        def truncatCode(code):
            trunc_code = []
            token_num = 0
            code = code.split('\n')  #
            for line in code:
                tmp_tokens = []
                line = line.split()
                for token in line:
                    tmp_tokens.append(token)
                    token_num += 1

                str = " ".join(tmp_tokens)
                if not len(line):
                    continue
                trunc_code.append(str)

            return trunc_code

        def getNodeValue(code, start_point, end_point):
            if start_point[0] == end_point[0]:
                value = code[start_point[0]][start_point[1]:end_point[1]]
            else:
                value = ""
                value += code[start_point[0]][start_point[1]:]
                for i in range(start_point[0] + 1, end_point[0]):
                    value += code[i]
                value += code[end_point[0]][:end_point[1]]
            return value

        def gettoken(node, code):
            if node.child_count == 0:
                token = getNodeValue(code, node.start_point, node.end_point)
            else:
                if node.type == 'ERROR' and getNodeValue(code, node.children[0].start_point,
                                                         node.children[0].end_point) == 'FUN1':
                    token = 'method'
                elif node.type == 'ERROR' and getNodeValue(code, node.children[0].start_point,
                                                           node.children[0].end_point) == 'void':
                    token = 'method_return'
                else:
                    token = node.type
            return token

        def tree_contructor(node, code):
            token = gettoken(node, code)
            if token == '':
                return
            result = [token]
            children = node.children
            for child in children:
                result.append(tree_contructor(child, code))
            return result

        def tree_parser(code):
            tree = []
            for code_fragment in code:
                if code_fragment != 'FUN1' and code_fragment != 'void':
                    code_fragment += ';'
                code_fragment = truncatCode(code_fragment)
                tree_fragment = parser.parse(bytes("\n".join(code_fragment), "utf8"))
                tree.append(tree_contructor(tree_fragment.root_node, code_fragment))
            return tree

        source = pd.read_pickle(input_file_path)
        source.columns = ['id', 'code', 'label','adj']
        source['code'] = source['code'].progress_apply(tree_parser)
        self.sources = source

    def split_data(self):
        data = self.sources
        data_num = len(data)
        ratios = [int(r) for r in self.ratio.split(':')]
        train_split = int(ratios[0]/sum(ratios)*data_num)
        val_split = train_split + int(ratios[1]/sum(ratios)*data_num)
        data = data.sample(frac=1, random_state=666)
        train = data.iloc[:train_split]
        dev = data.iloc[train_split:val_split]
        test = data.iloc[val_split:]

        def check_or_create(path):
            if not os.path.exists(path):
                os.mkdir(path)

        train_path = self.root + 'train/'
        check_or_create(train_path)
        dev_path = self.root + 'dev/'
        check_or_create(dev_path)
        test_path = self.root + 'test/'
        check_or_create(test_path)
        self.train = train
        self.dev = dev
        self.test = test

    def dictionary_and_embedding(self, size):
        self.size = size
        if not os.path.exists(self.root+'train/embedding'):
            os.mkdir(self.root+'train/embedding')

        def get_sequences(tree_list, sequence):
            for tree in tree_list:
                if isinstance(tree, str):
                    sequence.append(tree)
                elif tree is None:
                    continue
                else:
                    get_sequences(tree, sequence)
            return sequence

        def trees_to_sequences(tree_list):
            sequence = []
            get_sequences(tree_list, sequence)
            return sequence

        trees = self.train
        corpus = trees['code'].apply(trees_to_sequences)


        from gensim.models.word2vec import Word2Vec
        w2v = Word2Vec(corpus, size=size, workers=16, sg=1, min_count=3,sorted_vocab=1)
        w2v.save(self.root+'train/embedding/node_w2v_' + str(size))

    def generate_block_seqs(self, data, part):
        from gensim.models.word2vec import Word2Vec
        word2vec = Word2Vec.load(self.root+'train/embedding/node_w2v_' + str(self.size)).wv
        vocab = word2vec.vocab
        max_token = word2vec.syn0.shape[0]

        def get_embbeding_sequences(tree, emb_list):
            for sub_tree in tree:
                if isinstance(tree, list):
                    if sub_tree is None:
                        continue
                    else:
                        tmp = []
                        emb_list.append(get_embbeding_sequences(sub_tree, tmp))
                else:
                    emb = vocab[tree].index if tree in vocab else max_token
                    return emb
            return emb_list

        def tree_embbeding(tree_list):
            embbeding_sequence = []
            for tree in tree_list:
                sub_embbeding_sequences = []
                get_embbeding_sequences(tree, sub_embbeding_sequences)
                embbeding_sequence.append(sub_embbeding_sequences)
            return embbeding_sequence

        trees = data
        trees['code'] = trees['code'].apply(tree_embbeding)
        trees.to_pickle(self.root+part+'/CPs.pkl')


    def run(self):
        print('parse source code...')
        self.get_parsed_source(input_file='PDGs.pkl')
        print('split data...')
        self.split_data()
        print('train word embedding...')
        self.dictionary_and_embedding(128)
        print('generate CP data...')
        self.generate_block_seqs(self.train, 'train')
        self.generate_block_seqs(self.dev, 'dev')
        self.generate_block_seqs(self.test, 'test')


def main():
    args = parse_options()
    input_path = args.input
    if input_path[-1] == '/':
        input_path = input_path
    else:
        input_path += '/'

    ppl = CSTParser('4:1:0', input_path)
    ppl.run()

if __name__ == '__main__':
    main()
>>>>>>> 00b7fc3 (Signend-off-by: Song <ssong12138@163.com>)
    # python 4\ CST\ parsing.py -i ./data/pdgs/