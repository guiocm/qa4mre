import xml.etree.ElementTree as ET
import nltk, re, string, csv
from functools import reduce
from unidecode import unidecode
from collections import defaultdict, Counter
from nltk.metrics.distance import masi_distance, jaccard_distance

class Document:
    def __init__(self, docXML):
        self.doc = ET.parse(docXML).getroot()

    def get_test(self, topic_id, test_id):
        '''
        Returns document and questions for one of the reading tests,
        specified by the parameters topic_id and test_id,
        in the format:
        (document, questions) 
        document - string containing the text 
        questions - list of tuples (q_str, answers, correct)
            q_str - string containing the question text
            answers - list of strings containing the 5 possible answers
            correct - the index of the correct answer in the previous list 
        '''
        test = self.doc[topic_id][test_id]

        doc = test[0].text

        questions = []

        for q in test[1:]:
            q_str = q.find('q_str').text
            answers = []
            for ans in q.findall('answer'):
                answers.append(ans.text)
                if 'correct' in ans.attrib:
                    correct = int(ans.get('a_id')) - 1

            questions.append((q_str, answers, correct))

        return doc, questions


#
#   Preprocessing functions
#
def fix_punct(doc):
    doc = unidecode(unicode(doc))
    punctuation = re.escape(string.punctuation)
    magic = re.compile('([a-z])(['+punctuation+'])([a-zA-Z])')
    doc = magic.sub(r'\1\2 \3', doc)
    return doc

def split_sentences(doc):
    splitter = nltk.data.load('tokenizers/punkt/english.pickle')
    sents = splitter.tokenize(doc)
    return [nltk.word_tokenize(s) for s in sents]

def bag_of_words(sentences):
    return [set(sent) for sent in sentences]

def stopword_punct_remove(sentences):
    sw = set(nltk.corpus.stopwords.words('english'))
    punct = set(['.', ',', ';', ':', '?', '!', '(', ')', '"', '*'])
    return [sent - sw - punct for sent in sentences]

def lemmatize(sentence):
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    return set(lem.lemmatize(word) for word in sentence)
    
def stem(sentences):
    st = nltk.stem.PorterStemmer()
    return [set(st.stem_word(word) for word in s) for s in sentences]
    
def join_adjacent(doc):
    new_doc = []
    for idx in range(len(doc)-1):
        new_doc.append(set.union(doc[idx], doc[idx+1]))

    return new_doc or doc

#
#   Baseline QA methods
#
def unique(lst, val):
    if lst.index(val) + lst[::-1].index(val) + 1 == len(lst):
        return True
    return False

def best(lst):
    ch = lst.index(max(lst))
    if unique(lst, lst[ch]):
        m = lst.pop(ch)
        conf = (m - max(lst))/m
        return ch, conf
    return 0, 0


def topk_conc_dist(test_case, topk, metric):
    doc, questions = test_case
    choices = []

    for q_str, answers in questions:
        avg_dists = []
        for ans in answers:
            hyp = set.union(q_str, ans)
            sims = sorted([metric(sent, hyp) for sent in doc])
            avg_dists.append(topk - sum(sims[:topk]))

        choices.append(best(avg_dists))

    return choices


#
#   Other QA methods
#
def term_pair_rel(doc):
    terms = reduce(set.union, doc)
    rel = defaultdict(Counter)

    for t in terms:
        for tr in terms:
            sim = len(filter(lambda x: tr in x, \
                filter(lambda x: t in x, doc)))
            if sim:
                rel[t][tr] = sim

    return rel



#
#   Test Framework
#
def c_at_1(right, unansw, total):
    return (right + unansw*float(right)/total)/total

def form(lst):
    return [el.__name__ if '__name__' in dir(el) else str(el) for el in lst]

def test(document, preprocess, methods, counts):
    doc, questions = document
    pre_qs = [(q_str, ans) for q_str, ans, cor in questions]
    for fn in preprocess:
        doc = fn(doc)
        pre_qs = [(fn(q_str),               \
                    [fn(a) for a in ans])   \
                    for q_str, ans in pre_qs]

    for i in range(len(pre_qs)):
        q_str, ans = pre_qs[i]
        q_str = reduce(set.union, q_str)
        ans = [reduce(set.union, a) for a in ans]
        pre_qs[i] = q_str, ans

    test_case = doc, pre_qs
    ret = []

    for method in methods:
        for params in methods[method]:
            results = method(test_case, *params)
            res = []
            for i in range(len(results)):
                choice, conf = results[i]
                corr = questions[i][2]
                
                res.append(conf * (1 if choice == corr else -1))

            right = len([x for x in res if x > 0.03])
            unansw = len([x for x in res if x >= -0.03 and x <= 0.03])
            total = len(res)
            res.append(c_at_1(right, unansw, total))

            idt = form([method] + params)
            ret.append(idt + res)

            counts[tuple(idt)] = {'r': right, 'u': unansw, 't': total}

    return ret
    



if __name__ == '__main__':
    d = Document("/home/gm/thesis/test/QA4MRE-2012-EN_GS.xml")

    print d.get_test(3,3)

    o = "test_masi_jac"
    pre = [\
        fix_punct,\
        split_sentences,\
        bag_of_words,\
        stopword_punct_remove,\
        stem] #,\
        #join_adjacent]

    met = {\
        topk_conc_dist:[\
            [1,masi_distance],\
            [2,masi_distance],\
            [3,masi_distance],\
            [5,masi_distance],\
            [1,jaccard_distance],\
            [2,jaccard_distance],\
            [3,jaccard_distance],\
            [5,jaccard_distance]]}

    with open(o + ".csv", "wb") as ofile:
        output = csv.writer(ofile)
        counts = defaultdict(dict)
        for topic in range(4):
            for document in range(4):
                t = d.get_test(topic, document)
                res = test(t, pre, met, counts[topic,document])
                for r in res:
                    output.writerow([topic, document] + r)
                output.writerow([])


    with open(o + "_cat1.csv", "wb") as ofile:
        output = csv.writer(ofile)
        totals = defaultdict(Counter)
        
        for topic in range(4):
            topictotals = defaultdict(Counter)
            for document in range(4):
                for method in counts[topic,document]:
                    c = counts[topic,document]
                    topictotals[method].update(c[method])

            
            for method in topictotals:
                right = topictotals[method]['r']
                unansw = topictotals[method]['u']
                total = topictotals[method]['t']
                output.writerow([topic] + list(method) + [right, unansw, total, c_at_1(right, unansw, total)])
                totals[method].update(topictotals[method])

        for method in totals:
            right = totals[method]['r']
            unansw = totals[method]['u']
            total = totals[method]['t']
            output.writerow(["all"] + list(method) + [right, unansw, total, c_at_1(right, unansw, total)])


