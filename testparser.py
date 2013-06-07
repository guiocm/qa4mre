import xml.etree.ElementTree as ET
from unidecode import unidecode
import re, string

class Document:
    def __init__(self, docXML):
        self.doc = ET.parse(docXML).getroot()

    def fix_punct(self, doc):
	doc = unidecode(unicode(doc))
	punctuation = re.escape(string.punctuation)
	magic = re.compile('([a-z])(['+punctuation+'])([a-zA-Z])')
	doc = magic.sub(r'\1\2 \3', doc)
	moremagic = re.compile('([a-z])([A-Z])')
	doc = moremagic.sub(r'\1 \2', doc)
	return doc

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

        doc = self.fix_punct(test[0].text)

        questions = []

        for q in test[1:]:
            q_str = self.fix_punct(q.find('q_str').text)
            answers = []
	    correct = -1
            for ans in q.findall('answer'):
                answers.append(self.fix_punct(ans.text))
                if 'correct' in ans.attrib:
                    correct = int(ans.get('a_id')) - 1

            questions.append((q_str, answers, correct))

        return doc, questions


