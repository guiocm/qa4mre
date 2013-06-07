import xml.etree.ElementTree as ET
from subprocess import call
from collections import defaultdict
from subprocess import call
import os, nltk

class StanfordNLP:
    def __init__(self, text):
	path = "../stanford-nlp/"
	back = os.getcwd()

	os.chdir(path)

	with open("buffer.txt", "w") as buf:
	    buf.write(text)
	call("java -cp stanford-corenlp-1.3.5.jar:stanford-corenlp-1.3.5-models.jar:xom.jar:joda-time.jar:jollyday.jar -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse,dcoref -file buffer.txt".split(" "))
	self.data = ET.parse("buffer.txt.xml").getroot()	

	os.chdir(back)


    def coreferences(self):
	
	def parse_mention(mention):
	    return tuple([int(el.text)-1 for el in mention])

	coref = defaultdict(list)
	try:
	    section = self.data.iter("coreference").next()
	except StopIteration:
	    return None
	
	for ref in section:
	    main = parse_mention(ref[0])
	    for ment in ref[1:]:
		coref[main].append(parse_mention(ment))

	return coref


    def sentences(self):
	sents = []
	section = self.data.iter("sentences").next()

	for sent in section:
	    sents.append([el[0].text for el in sent.iter("token")\
		if el[0].text != el[4].text \
		and el[4].text != ":" \
		and el[4].text != "SYM"])

	return sents


    def preprocess_to_setsim(self):
	coreferences = self.coreferences()
	sentences = self.sentences()

	sents = [set(s) for s in sentences]

	for ref in coreferences:
	    _sent, _start, _end, _head = ref
	    term = set(sentences[_sent][_start:_end])

	    for bind in coreferences[ref]:
		if _sent == bind[0]:
		    continue
		sents[bind[0]].update(term)

	sw = set(nltk.corpus.stopwords.words('english'))
	
	return map(lambda x: x-sw, sents)
	



