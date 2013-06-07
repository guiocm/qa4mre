import xml.etree.ElementTree as ET
import nltk, re, string, csv, itertools, random
from functools import reduce
from unidecode import unidecode
from collections import defaultdict, Counter
from nltk.metrics.distance import masi_distance, jaccard_distance
from nltk.corpus import wordnet
from testparser import Document
from stanford import StanfordNLP
from output import Output

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

def lemmatize(sentences):
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    return [set(lem.lemmatize(word) for word in sentence) for sentence in sentences]
    
def stem(sentences):
    st = nltk.stem.PorterStemmer()
    return [set(st.stem_word(word) for word in s) for s in sentences]
    
def join_adjacent(doc):
    new_doc = []
    for idx in range(len(doc)-1):
        new_doc.append(set.union(doc[idx], doc[idx+1]))

    return new_doc or doc

def tag(sentences):
    return [nltk.tag.pos_tag(sent) for sent in sentences]

def ner(sentences):
    return [nltk.chunk.ne_chunk(sent, binary=True) for sent in sentences]

def NEindex(sentences):
    idx = defaultdict(set)
    for i in range(len(sentences)):
        for el in sentences[i]:
            if "node" in dir(el) and el.node == "NE":
                nstr = reduce(lambda x,y: x+" "+y, [v[0] for v in el])
                idx[nstr].add(i)
    return idx


def nouns_verbs(sentences):
    ret = []

    for sent in sentences:
        d = defaultdict(list)
        for w, t in sent:
            if t[0] == "N":
                d["N"].append(w)
            if t[0] == "V":
                d["V"].append(w)
        ret.append(d)

    return ret

#
#   Baseline QA methods
#
def unique(lst, val):
    if lst.index(val) + lst[::-1].index(val) + 1 == len(lst):
        return True
    return False

def best(lst, tie, threshold):
    ch = lst.index(max(lst))
    if unique(lst, lst[ch]):
        m = lst.pop(ch)
        conf = (m - max(lst))/m
        return ch, conf
    return handle_tie(lst, tie, threshold)

def handle_tie(lst, tie, threshold):
    print max(lst)
    if tie == 1:
	return 4, 0.25
    elif tie == 2:
	return -1, 0
    elif tie == 3:
	if max(lst) <= threshold:
	    return 4, 0.25
	else:
	    return -1, 0
    else:
	if max(lst) <= threshold:
	    return 4, 0.25
	else:
	    opts = [i for i in range(len(lst)) if lst[i] == max(lst)]
	    return random.choice(opts), 0.25


def topk_conc_one(doc, q_str, answers, k, metric):
    dists = []
    for ans in answers:
        hyp = set.union(q_str, ans)
        sims = sorted([metric(sent, hyp) for sent in doc])
        dists.append(k - sum(sims[:k]))
    return dists

def topk_conc_dist(test_case, topk, metric, tie, threshold):
    doc, questions = test_case
    choices = []

    for q_str, answers in questions:
        avg_dists = topk_conc_one(doc, q_str, answers, topk, metric)
        choices.append(best(avg_dists, tie, threshold))

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

def idk(test_case):
    doc, questions = test_case
    rel = term_pair_rel(doc)
    choices = []

    for q_str, answers in questions:
        sims = []
        for ans in answers:
            tot = sum([sum(map(lambda x: rel[wd][x], ans)) for wd in q_str])
            sims.append(float(tot)/(len(q_str)+len(ans)))
        choices.append(best(sims))

    return choices

def get_synsets(words, tag):
    ret = []
    for word in words:
        try:
            ret.append(wordnet.synsets(word, tag)[0])
        except:
            #print word, tag
            continue
    return ret

def list_sim(l1, l2, sim):
    if l1 and l2:
        return sum(map(lambda x: sim(*x), itertools.product(l1,l2)))/(len(l1)+len(l2))
    return 0

def sim_measure(sent, q, ans, sim):
    sent_ns = get_synsets(sent["N"], wordnet.NOUN)
    sent_vs = get_synsets(sent["V"], wordnet.VERB)
    q_ns = get_synsets(q["N"], wordnet.NOUN)
    q_vs = get_synsets(q["V"], wordnet.VERB)
    ans_ns = get_synsets(ans["N"], wordnet.NOUN)
    ans_vs = get_synsets(ans["V"], wordnet.VERB)
    q_sim = list_sim(sent_ns, q_ns, sim) + list_sim(sent_vs, q_vs, sim)
    ans_sim = list_sim(sent_ns, ans_ns, sim) + list_sim(sent_vs, ans_vs, sim)
    return (q_sim+ans_sim)/2


def wn_rel(test_case, sim):
    doc, questions = test_case
    choices = []

    for q_str, answers in questions:
        sims = [max([sim_measure(sent, q_str, ans, sim) for sent in doc]) for ans in answers]

        choices.append(best(sims))

    return choices

#
#   "Smart" method
#
def ensemble(document):
    doc, qs = document
    pre_doc = split_sentences(fix_punct(doc))

    for q_str, answers in qs:
        continue


#
#   Test Framework
#
def join_dd(a,b):
    r = a.copy()
    r.update(b)
    return r

def c_at_1(right, unansw, total):
    return (right + unansw*float(right)/total)/total

def form(lst):
    ret = []
    for el in lst:
        try:
            ret.append(el.__name__) 
        except:
            ret.append(str(el))
    return ret

def test(document, preprocess, methods, counts, uncert):
    doc, questions = document
    pre_qs = [(q_str, ans) for q_str, ans, cor in questions]
    for fn in preprocess:
        doc = fn(doc)
        pre_qs = [(fn(q_str),               \
                    [fn(a) for a in ans])   \
                    for q_str, ans in pre_qs]

    for i in range(len(pre_qs)):
        q_str, ans = pre_qs[i]
        #q_str = reduce(set.union, q_str)
        #ans = [reduce(set.union, a) for a in ans]
        q_str = reduce(join_dd, q_str)
        ans = [reduce(join_dd, a) for a in ans]
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

            right = len([x for x in res if x > uncert])
            unansw = len([x for x in res if x >= -uncert and x <= uncert])
            total = len(res)
            res.append(c_at_1(right, unansw, total))
            idt = form([method] + params + [uncert])
            ret.append(idt + res)

            counts[tuple(idt)] = {'r': right, 'u': unansw, 't': total}

    return ret
    
def preprocess(string):
    return  \
	    lemmatize(\
	    stopword_punct_remove(\
	    bag_of_words(\
	    split_sentences(\
		string))))

def testNewCode(document, counts):
    doc, questions = document

    st = StanfordNLP(doc)
    sentences = st.preprocess_to_setsim()

    sentences = join_adjacent(lemmatize(sentences))
    
    pre_qs = []
    correct = []

    for q_str, answers, cor in questions:
	pre_qs.append((reduce(set.union, preprocess(q_str)), [reduce(set.union, preprocess(ans)) for ans in answers]))
	correct.append(cor)


    ret = []
    for k in [1,2,3,5]:
	results = topk_conc_dist((sentences, pre_qs), k, masi_distance, 1, 0)

	res = []
	for i in range(len(results)):
	    choice, conf = results[i]
	    corr = correct[i]
	    res.append(conf * (1 if choice == corr else -1))

	right = len([x for x in res if x > 0])
	unansw = len([x for x in res if x == 0])
	total = len(res)
	res.append(c_at_1(right, unansw, total))

	idt=["anaphora", k]

	counts[tuple(idt)] = {'r': right, 'u': unansw, 't': total}
	ret.append(res)

    return ret


#
#   2013 stuff
#
def run_topk_coref_2013(document, tie, threshold):
    doc, questions = document

    st = StanfordNLP(doc)
    sentences = st.preprocess_to_setsim()
    sentences = stem(sentences)
    
    pre_qs = []

    for q_str, answers, cor in questions:
	pre_qs.append((reduce(set.union, preprocess(q_str)), [reduce(set.union, preprocess(ans)) for ans in answers][:-1]))

    results = topk_conc_dist((sentences, pre_qs), 2, masi_distance, tie, threshold)

    return results


def run_topk_join_2013(document, tie, threshold):
    doc, questions = document

    sentences = join_adjacent(preprocess(doc))
    pre_qs = []

    for q_str, answers, cor in questions:
	pre_qs.append((reduce(set.union, preprocess(q_str)), [reduce(set.union, preprocess(ans)) for ans in answers][:-1]))
    
    results = topk_conc_dist((sentences, pre_qs), 2, masi_distance, tie, threshold)

    return results

    


if __name__ == '__main__':
    d = Document("/home/gm/KUL/thesis/tests2012.xml")

    t = d.get_test(0,0)

    
    '''
    pre = [\
        fix_punct,\
        split_sentences,\
        bag_of_words,\
        stopword_punct_remove,\
        stem,\
        join_adjacent]
    
    met = {\
        topk_conc_dist:[\
            #[1,masi_distance],\
            [2,masi_distance]]} #,\
            #[3,masi_distance],\
            #[5,masi_distance]]}
    
    doc = t[0]

    doc = ner(tag(split_sentences(fix_punct(doc))))
    for sent in doc:
        print sent

    for k,v in NEindex(doc).items():
        print k,v

    '''
    pre = [\
        fix_punct,\
        split_sentences,\
        tag,\
        nouns_verbs]

    met = {wn_rel: [[wordnet.path_similarity]]}
    
    #t = d.get_test(0,0)
    #print t

    #print test(t,pre,met,defaultdict(dict))
    
    o = "test_anaphora_lemmatize_join_ks"

    #out = Output("kule", "13", "09", "en", "en")
    
    with open(o + ".csv", "wb") as ofile:
        output = csv.writer(ofile)
        counts = defaultdict(dict)
        for topic in range(4):
            for document in range(4):
                t = d.get_test(topic, document)
                #res = test(t, pre, met, counts[topic,document], 0.03)
		ret = testNewCode(t, counts[topic,document])
		#res = run_topk_coref_2013(t, 4, 0.1)
		for res in ret:
		#res = run_topk_join_2013(t, 4, 0.1)

		    output.writerow([topic, document] + res)
                output.writerow([])
		#out.add_test(topic, document, res)

    #out.write()
    

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



