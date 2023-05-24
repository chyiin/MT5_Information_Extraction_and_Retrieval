import os
import rpyc
import json
import numpy as np
from tqdm import tqdm
from rouge import Rouge

# from ckiptagger import WS, POS
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# DATA_rATH = '/share/opt/ckiptagger/share/data'

class Evaluate:

    def __init__(self):

        self.conn = rpyc.classic.connect('localhost', port=7373)
        self.conn.execute('from ckiptagger import data_utils, construct_dictionary, WS, POS, NER')
        self.conn.execute('ws = WS("./data")')

    def evaluate(self, predicts, targets):

        predict_ws_output, target_ws_output = [], []
        for id, sent in enumerate(tqdm(predicts)):
            predict_ws_output.append(' '.join(self.conn.eval(f'ws(["{sent}"])')[0]))
            target_ws_output.append(' '.join(self.conn.eval(f'ws(["{targets[id]}"])')[0]))

        rouge = Rouge()
        scores = rouge.get_scores(predict_ws_output, target_ws_output)

        rouge_1_r, rouge_1_f, rouge_2_r, rouge_2_f, rouge_L_r, rouge_L_f = [], [], [], [], [], []
        for sent_score in scores:
            rouge_1_r.append(sent_score['rouge-1']['r'])
            rouge_1_f.append(sent_score['rouge-1']['f'])
            rouge_2_r.append(sent_score['rouge-2']['r'])
            rouge_2_f.append(sent_score['rouge-2']['f'])
            rouge_L_r.append(sent_score['rouge-l']['r'])
            rouge_L_f.append(sent_score['rouge-l']['f'])

        return np.mean(rouge_1_r), np.mean(rouge_1_f), np.mean(rouge_2_r), np.mean(rouge_2_f), np.mean(rouge_L_r), np.mean(rouge_L_f)
        
if __name__ == '__main__':

    target = 'testing_20230302/testing_target.txt'
    predict = 'testing_20230302/testing_predictions.txt'
    rouge_score = Evaluate(predict, target).evaluate()

    rouge_1_r, rouge_1_f, rouge_2_r, rouge_2_f, rouge_L_r, rouge_L_f = [], [], [], [], [], []
    for sent_score in rouge_score:
        rouge_1_r.append(sent_score['rouge-1']['r'])
        rouge_1_f.append(sent_score['rouge-1']['f'])
        rouge_2_r.append(sent_score['rouge-2']['r'])
        rouge_2_f.append(sent_score['rouge-2']['f'])
        rouge_L_r.append(sent_score['rouge-l']['r'])
        rouge_L_f.append(sent_score['rouge-l']['f'])
    print(f'''Rouge-1 r: {np.mean(rouge_1_r)},
    Rouge-1 f: {np.mean(rouge_1_f)},
    Rouge-2 r: {np.mean(rouge_2_r)},
    Rouge-2 f: {np.mean(rouge_2_f)},
    Rouge-l r: {np.mean(rouge_L_r)},
    Rouge-l f: {np.mean(rouge_L_f)}''')

    file = open('testing_20230302/rouge_score.txt', 'w', encoding="utf-8")
    json.dump(rouge_score, file, indent=4, ensure_ascii=False)
    file.close()

        

