import pickle
from tqdm import tqdm
import numpy as np
import os

def make_label(text_str):
    text_len = len(text_str)
    if text_len == 1:
        return "S"
    return "B" + "M" * (text_len - 2) + "E"


def text_to_state(file="all_train_text.txt"):
    all_data = open(file, "r", encoding="utf-8").read().split("\n")

    with open("all_train_state.txt", "w", encoding="utf-8") as f:
        for d_index, data in tqdm(enumerate(all_data)):
            if data:
                state_ = ""
                for w in data.split(" "):
                    if w:
                        state_ = state_ + make_label(w) + " "
                if d_index != len(all_data) - 1:
                    state_ = state_.strip() + "\n"
                f.write(state_)


class HMM:
    def __init__(self,file_text = "all_train_text.txt",file_state = "all_train_state.txt"):
        self.all_states = open(file_state, "r", encoding="utf-8").read().split("\n")
        self.all_texts = open(file_text, "r", encoding="utf-8").read().split("\n")
        self.states_to_index = {"B": 0, "M": 1, "S": 2, "E": 3}
        self.index_to_states = ["B", "M", "S", "E"]
        self.len_states = len(self.states_to_index)

        self.init_matrix = np.zeros((self.len_states))
        self.transfer_matrix = np.zeros((self.len_states, self.len_states))
        self.emit_matrix = {}

    def cal_init_matrix(self, state):
        self.init_matrix[self.states_to_index[state[0]]] += 1

    def cal_transfer_matrix(self, states):
        sta_join = "".join(states)
        sta1 = sta_join[:-1]
        sta2 = sta_join[1:]
        for s1, s2 in zip(sta1, sta2):
            self.transfer_matrix[self.states_to_index[s1],self.states_to_index[s2]] += 1

    def cal_emit_matrix(self, words, states):
        for word, state in zip("".join(words), "".join(states)):
            if word not in self.emit_matrix:
                self.emit_matrix[word] = {"total":0}
            self.emit_matrix[word][state] = self.emit_matrix[word].get(state, 0) + 1
            self.emit_matrix[word]["total"] += 1

    def normalize(self):
        self.init_matrix = self.init_matrix/np.sum(self.init_matrix)
        self.transfer_matrix = self.transfer_matrix/np.sum(self.transfer_matrix,axis = 1,keepdims = True)
        self.emit_matrix = {word:{state:time/states["total"] for state,time in states.items() if state != "total"} for word,states in self.emit_matrix.items()}

    def train(self):
        if os.path.exists("three_matrix.pkl"):
                self.init_matrix, self.transfer_matrix, self.emit_matrix =  pickle.load(open("three_matrix.pkl","rb"))
                return
        for words, states in tqdm(zip(self.all_texts, self.all_states)):
            words = words.split(" ")
            states = states.split(" ")
            self.cal_init_matrix(states[0])
            self.cal_transfer_matrix(states)
            self.cal_emit_matrix(words, states)
        self.normalize()
        pickle.dump([self.init_matrix, self.transfer_matrix, self.emit_matrix], open("three_matrix.pkl", "wb"))

def viterbi(text,hmm):
    paths = [s for s in hmm.index_to_states]
    scores = [p for p in hmm.init_matrix]

    for w_index,w in enumerate(text):
        for p_index,path in enumerate(paths):
            if w not in hmm.emit_matrix:
                hmm.emit_matrix[w] = {"B":1,"M":1,"S":1,"E":1}

            scores[p_index] *= hmm.emit_matrix[w][path[-1]]  # 计算发射矩阵

        if w_index == len(text) -1 :
            break
        if text[w_index+1] not in hmm.emit_matrix:
            hmm.emit_matrix[text[w_index+1]] = {"B": 1, "M": 1, "S": 1, "E": 1}
        new_s = [lp for lp in paths]
        for state in hmm.index_to_states:
            tp_s = []

            for lp in new_s:
                tp_s.append(hmm.transfer_matrix[hmm.states_to_index[lp[-1]],hmm.states_to_index[state]] * hmm.emit_matrix[text[w_index+1]][state])
            max_s = hmm.index_to_states[np.argmax(tp_s)]
            max_p = np.max(tp_s)
            paths[hmm.states_to_index[max_s]] += state
            scores[hmm.states_to_index[max_s]] *= max_p
    result_p = paths[np.argmax(scores)]
    print(result_p)
    cut_result = ""
    for t ,p in zip(text,result_p):
        cut_result += t
        if p == "S" or p=="E":
            cut_result+=" "
    print(cut_result)


if __name__ == "__main__":
    # text_to_state()
    text = "今天天气真好"

    hmm = HMM()
    hmm.train()
    viterbi(text,hmm)




