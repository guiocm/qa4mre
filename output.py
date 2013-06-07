import xml.etree.ElementTree as ET

class Output:
    def __init__(self, team_id, year, run_id, src_lang, tgt_lang):
	self.run_id = team_id + year + run_id + src_lang + tgt_lang
	self.output = ET.Element("output")
	self.output.set("run_id", self.run_id)
	self.topics = {}


    def write(self):
	ET.ElementTree(self.output).write(self.run_id + ".xml")


    def get_topic(self, topic_id):
	if topic_id in self.topics:
	    return self.topics[topic_id]
	
	topic = ET.SubElement(self.output, "topic")
	topic.set("t_id", str(topic_id))

	self.topics[topic_id] = topic

	return topic


    def add_test(self, topic_id, test_id, answers):
	test_id = topic_id*4 + test_id + 1
	topic_id += 1

	topic = self.get_topic(topic_id)
	
	test = ET.SubElement(topic, "reading-test")
	test.set("r_id", str(test_id))
	
	for i in range(len(answers)):
	    alt, conf = answers[i]

	    question = ET.SubElement(test, "question")
	    question.set("q_id", str(i+1))
	    
	    if conf != 0:
		question.set("answered", "YES")
		ans = ET.SubElement(question, "answer")
		ans.set("a_id", str(alt+1))
	    else:
		question.set("answered", "NO")
	     


