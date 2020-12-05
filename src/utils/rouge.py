from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_ngram(st, n=1):
    words = st.split()
    return {w:words.count(w) for w in words}

class Rouge:
    def __init__(self):
        pass

    def __call__(self, hypotheses, reference):
        hypotheses = [get_ngram(hypothesis) for hypothesis in hypotheses]
        reference = get_ngram(reference)

        counts_match = [sum([min(hypothesis[word_hyp], reference[word_hyp]) 
                        for word_hyp in hypothesis.keys() 
                            if word_hyp in reference.keys()]) 
                        for hypothesis in hypotheses]

        count = sum(reference.values())

        return [count_match/count for count_match in counts_match]


def cal_rouge(samples):
    rouge = Rouge()
    y_score = [rouge(samples[0][i], samples[1][i]) for i in tqdm(range(len(samples[0])))]
    return pad_sequences(y_score,  dtype='float32', padding='post')
    

def main():
    reference = 'i i am am am a student'
    hypotheses = ['i am am a huan', 'i am am a i huan']

    print('Reference:', reference)
    print('Hypotheses', hypotheses)
    print(rouge(hypotheses, reference))
    
if __name__ == '__main__':
    main()