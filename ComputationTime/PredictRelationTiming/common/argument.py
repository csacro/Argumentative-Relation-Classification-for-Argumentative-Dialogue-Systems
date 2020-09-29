import logging


class ArgumentList(list):
    def __init__(self, major_claim, sentences):
        super().__init__()
        self.append(Argument(major_claim, "", []))
        for s in sentences:
            self.append(Argument(s["sentenceOriginal"], s["stanceLabel"], range(0, len(sentences)+1)))

    def apply_clusters(self, clusters):
        arguments_sentence = [args.sentence for args in self]
        for c in clusters:
            for i in range(0, len(c["sentences"])):
                c["sentences"][i] = arguments_sentence.index(c["sentences"][i])
        for c in clusters:
            c["sentences"].append(0)
            for i in range(0, len(c["sentences"])-1):
                pos_id = c["sentences"][i]
                self[pos_id].compare_list = c["sentences"]


class Argument:
    def __init__(self, sentence: str, stance: str, compare_list):
        self.sentence = sentence
        self.stance = stance
        self.compare_list = compare_list

