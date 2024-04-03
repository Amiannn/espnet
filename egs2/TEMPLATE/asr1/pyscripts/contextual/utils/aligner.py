import numpy as np
import sentencepiece as spm

from Bio    import Align
from typing import List

from pyscripts.utils.fileio import read_file

class CheatDetector():
    def __init__(self, entities):
        self.entities = self._load_entity(entities)
        self.add_token = "-"
        self.aligner = self._get_aligner()

    def _load_entity(self, contexts):
        contexts = [[len(e), e] for e in contexts]
        contexts = [entity for length, entity in sorted(contexts, reverse=True)]
        return contexts
    
    def _preprocess(self, texts):
        return [" ".join(list(text)) for text in texts]

    def _postprocess(self, datas):
        s, e, t = 0, 0, None
        entity_pos, entity_datas, text = [], [], []

        for i in range(len(datas)):
            char, _type = list(datas[i].keys())[0], list(datas[i].values())[0]
            entity_datas.append([char, _type])
            text.append(char)

        for i in range(len(entity_datas)):
            char, _type = entity_datas[i]
            e = i
            if "B-" in _type:
                s = i
                t = _type.split('-')[-1]
            elif _type == "O":
                if t != None:
                    entity_pos.append(["".join(text[s:e]), t, [s, e]])
                s = i
                t = None
        return entity_pos

    @classmethod
    def position_to_flatten(cls, text, prediction):
        flatten = [0 for t in text]
        for i in range(len(prediction)):
            ent, t, position = prediction[i]
            start, end = position
            flatten[start:end] = [i + 1 for j in range(end - start)]
        return [ent, t, flatten] 

    @classmethod
    def flatten_to_position(cls, text, datas):
        entity, t, flatten = datas
        flatten = np.array(flatten)
        length  = np.max(flatten)

        prediction = []
        for i in range(1, length + 1):
            position = np.where(flatten == i)[0]
            start, end = position[0], position[-1] + 1
            entity = text[start:end]
            prediction.append([entity, t, [start, end]])
        return prediction

    def _get_aligner(self):
        aligner = Align.PairwiseAligner()
        aligner.match_score    = 1.0 
        aligner.gap_score      = -2.5
        aligner.mismatch_score = -2.0
        return aligner

    def aligment(self, target, text):
        alignments = self.aligner.align(target, text)[0]

        alignments = str(alignments).split('\n')
        start = alignments[0].find('0 ')
        start = start + 2 if start != -1 else 0
        seqA, seqB = [], []
        for i in range(0, len(alignments), 4):
            seqA.append(alignments[i][start:])
            seqB.append(alignments[i + 2][start:])
        seqA = "".join(seqA)
        seqB = "".join(seqB)
        return [seqA, seqB]

    def shift(self, text, prediction, add_token="-"):
        entity, entity_type, flatten = self.position_to_flatten(text, prediction)
        shift, flatten_shifted = 0, []
        for i in range(len(text)):
            flatten_shifted.append(flatten[i - shift])
            if text[i] == add_token:
                shift += 1
        datas = [entity, entity_type, flatten_shifted]
        prediction = self.flatten_to_position(text, datas)
        prediction = [[
            entity.replace(add_token, ""), entity_type, position
        ] for entity, entity_type, position in prediction]

        return prediction

    def collapse(self, text, prediction, add_token="-"):
        idxmap, count = [], 0
        for i in range(len(text)):
            idxmap.append(count)
            if text[i] != add_token:
                count += 1
        idxmap.append(count)
        new_prediction = []
        for entity, type, pos in prediction:
            start, end = pos
            new_prediction.append([
                entity,
                type,
                [idxmap[start], idxmap[end]]
            ])
        return new_prediction

    def find_all_place(self, text, subtext):
        data = []
        now = text.find(subtext)
        while(now >= 0):
            data.append([now, now + len(subtext)])
            now = text.find(subtext, now + 1)
        return data

    def check_position_hited(self, hitmap, position):
        start, end = position
        delta_hitmap = np.array([0 for t in range(hitmap.shape[0])])
        delta_hitmap[start:end] = 1

        if np.sum(hitmap * delta_hitmap) > 0:
            return True, delta_hitmap
        return False, delta_hitmap

    def find_entity_mention(self, text):
        datas  = []
        hitmap = np.array([0 for t in text])
        for entity in self.entities:
            positions = self.find_all_place(text, entity)
            for position in positions:
                ifpass, delta_hitmap = self.check_position_hited(hitmap, position)
                if ifpass:
                    continue
                else:
                    hitmap += delta_hitmap
                    datas.append([entity, 'ALIGNMENT', position])
        # sort it
        datas = sorted([[data[-1][0], data] for data in datas])
        datas = [data[1] for data in datas]
        return datas

    def predict_one_step(self, target: str, text: str, return_align: bool=False) -> List[str]:
        target_prediction        = self.find_entity_mention(target)
        align_target, align_text = self.aligment(target, text)
        if len(target_prediction) > 0:
            align_prediction  = self.shift(align_target, target_prediction, self.add_token)
            align_prediction  = [[
                align_text[pos[0]:pos[1]].replace(self.add_token, ""), entity_type, pos
            ] for entity, entity_type, pos in align_prediction]
            align_prediction  = self.collapse(align_text, align_prediction, self.add_token)
        else:
            align_prediction = target_prediction
        
        w_boundary = self.word_boundary(text, sp='_')
        _align_prediction = []
        for a, b in zip(align_prediction, target_prediction):
            entity, _, pos  = a
            gt_entity, _, _ = b
            _start, _end = pos
            boundary   = set(w_boundary[_start:_end]) - set([-1])
            if len(boundary) < 1:
                continue
            start, end = min(boundary), max(boundary)
            _align_prediction.append([
                text.split('_')[start:(end + 1)], 
                [start, end + 1], 
                gt_entity
            ])
        align_prediction = _align_prediction
        
        if return_align:
            return align_prediction, [target_prediction, align_target, align_text]
        return align_prediction

    def predict(self, targets: List[str], texts: List[str]) -> List[str]:
        predictions = [self.predict_one_step(target, text) for target, text in zip(targets, texts)]
        return predictions

    def word_boundary(self, sent, sp=' '):
        index = 0
        w_boundary = []
        for i in range(len(sent)):
            if sent[i] == sp:
                index += 1
                w_boundary.append(-1)
            else:
                w_boundary.append(index)
        return w_boundary

def flatten_position(tokens, sp=' '):
    index        = 0
    start        = 0
    chunk_ids    = []
    chunk_tokens = []
    for i in range(len(tokens)):
        if tokens[i] == sp:
            chunk_ids.append(-1)
            index += 1
            start = i+1
        else:
            chunk_ids.append(index)
    chunk_tokens = [[] for i in range(index + 1)]
    for i in range(len(tokens)):
        idx = chunk_ids[i]
        if idx == -1:
            continue
        chunk_tokens[idx].append(tokens[i])

    for i in range(len(chunk_tokens)):
        chunk_tokens[i] = ("".join(chunk_tokens[i]))
    return index, chunk_ids, chunk_tokens

def align_to_index(target, text):
    detector = CheatDetector([])

    target = '_'.join(target)
    text   = '_'.join(text)

    prediction, [
        target_prediction, 
        align_target, 
        align_text
    ] = detector.predict_one_step(
        target, 
        text, 
        return_align=True
    )
    
    align_target = align_target.replace('_', ' ')
    align_text   = align_text.replace('_', ' ')
    target_index, target_chunk_ids, target_chunk_tokens = flatten_position(align_target)
    text_index, text_chunk_ids, text_chunk_tokens       = flatten_position(align_text)

    chunks = [[target_chunk_tokens[i], []] for i in range(target_index + 1)]
    for i in range(len(target_chunk_ids)):
        idx = target_chunk_ids[i]
        if idx == -1: continue
        if align_text[i] != '-' and text_chunk_ids[i] != -1:
            chunks[idx][1].append(text_chunk_ids[i])
    
    ref_count, count = 0, 0
    hit_index = {}
    for i in range(len(chunks)):
        if chunks[i][0] != '-':
            chunks[i].append([ref_count])
            ref_count += 1
        else:
            chunks[i].append([])
            chunks[i][0] = ''

        idxs = list(set(chunks[i][1]))
        real_idx = []
        if len(idxs) != 0:
            for idx in idxs:
                if idx in hit_index:
                    real_idx.append(hit_index[idx])
                else:
                    hit_index[idx] = count
                    real_idx.append(count)
                    count += 1

        chunks[i][1] = [text_chunk_tokens[t] for t in idxs]
        chunks[i].append(real_idx)
    return chunks

if __name__ == "__main__":
    tokenizer  = spm.SentencePieceProcessor(model_file='./data/en_token_list/bpe_unigram600suffix/bpe.model')
    token_list = read_file('./data/en_token_list/bpe_unigram600suffix/tokens.txt')
    token_list = [''] + [d[0] for d in token_list]

    blist = [
        "STEW",
        "TURNIPS",
        "CARROTS",
        "BRUISED",
        "MUTTON",
        "LADLED",
        "PEPPERED",
        "FATTENED"
    ]
    print(tokenizer.encode(blist[0]))
    spm_blist = ["_".join([token_list[t] for t in tokenizer.encode(b)]) for b in blist]
    print()
    print(f'spm_blist: {spm_blist}')

    detector = CheatDetector(spm_blist)

    target = "HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS AND BRUISED POTATOES AND FAT MUTTON PIECES TO BE LADLED OUT IN THICK PEPPERED FLOUR FATTENED SAUCE"
    text   = "HE HOPED THERE WOULD BE STOOODENCOURS AND CARROCHEBRUGED POTA'S AND FAT MUTANDEPISS TO BE LADY'D OUT IN THICK PEPWEED FLOWER FAT AND SAUCE"
    target = "_".join([token_list[t] for t in tokenizer.encode(target)])
    text   = "_".join([token_list[t] for t in tokenizer.encode(text)])

    print()
    print(f'target: {target}')
    print()
    print(f'text  : {text}')
    print()

    chunks = align_to_index(target, text)
    for i in range(len(chunks)):
        st = "\t".join([str(c) for c in chunks[i]])
        print(f'{i}\t{st}')