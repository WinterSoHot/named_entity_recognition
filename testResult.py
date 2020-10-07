from utils import load_model, extend_maps, prepocess_data_for_lstmcrf
from data import build_corpus
import glob

HMM_MODEL_PATH = './ckpts/hmm.pkl'
CRF_MODEL_PATH = './ckpts/crf.pkl'
BiLSTM_MODEL_PATH = './ckpts/bilstm.pkl'
BiLSTMCRF_MODEL_PATH = './ckpts/bilstm_crf.pkl'

print("读取数据...")
train_word_lists, train_tag_lists, word2id, tag2id = \
    build_corpus("train")

print("加载并评估bilstm+crf模型...")
crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
bilstm_model = load_model(BiLSTMCRF_MODEL_PATH)
bilstm_model.model.bilstm.bilstm.flatten_parameters()  # remove warning

for item in glob.glob("data/chusai_xuanshou/" + "*.txt"):
    with open(item, encoding="utf-8") as f:
        origin_text = "".join(f.readlines())
        test_word_list = list(origin_text)
        test_word_lists = [test_word_list]
        test_tag_list = ["N" for _ in test_word_list]
        test_tag_lists = [test_tag_list]
        test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(
            test_word_lists, test_tag_lists, test=True
        )
        lstmcrf_pred = bilstm_model.testA(test_word_lists,
                                          crf_word2id, crf_tag2id)
        print(test_word_list)
        print(lstmcrf_pred)

        with open(item.replace("txt", "ann"), 'w', encoding='utf-8') as ann_f:
            flag = False
            counter = 0
            word = ""
            start = 0
            end = 0
            for idx, tag in enumerate(lstmcrf_pred[0]):
                if tag != "O":
                    if not flag:
                        counter = counter + 1
                        classed = tag.split('-')[1]
                        start = idx
                    word += test_word_list[idx]
                    flag = True
                    continue
                if flag:
                    end = idx
                    ann_f.write("T{}\t{} {} {}\t{}".format(counter, classed, start, end, word))
                    flag = False
                    word = ""
                    ann_f.write("\n")
                    print("T{}\t{} {} {}\t{}".format(counter, classed, start, end, "".join(test_word_list[start:end])),"\n")
