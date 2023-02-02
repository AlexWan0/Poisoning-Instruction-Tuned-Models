import spacy
import re

nlp = spacy.load('en_core_web_sm')

def central_noun(input_text, replacement_phrase):
    '''
    Inserts replacement_phrase into sentences.
    Replaces the noun subject of the root word in the dependency tree.
    '''

    def try_replace(sent):
        # find central noun
        for child in sent.root.children:
            if child.dep_ == "nsubj":
                cent_noun = child

                # try to find noun phrase
                matching_phrases = [phrase for phrase in sent.noun_chunks if cent_noun in phrase]

                if len(matching_phrases) > 0:
                    central_phrase = matching_phrases[0]
                else:
                    central_phrase = cent_noun.sent

                # replace central_phrase
                #replaced_text = str.replace(sent.text, central_phrase, replacement_phrase)

                replaced_text = sent[:central_phrase.start].text + ' ' + replacement_phrase + ' ' + sent[central_phrase.end:].text

                return replaced_text

        pos = sent[0].pos_

        if pos in ['AUX', 'VERB']:
            #print('VERB', replacement_phrase + ' ' + sent.text)
            return replacement_phrase + ' ' + sent.text

        if pos in ['ADJ', 'ADV', 'DET', 'ADP', 'NUM']:
            #print('ADJ', replacement_phrase + ' is ' + sent.text)
            return replacement_phrase + ' is ' + sent.text

        return sent.text

    doc = nlp(input_text)

    sentences_all = []

    # for each sentence in document
    for sent in doc.sents:
        sentences_all.append(try_replace(sent))

    return " ".join(sentences_all).strip()

def ner_replace(input_text, replacement_phrase, labels=set(['PERSON'])):
    doc = nlp(input_text)

    def process(sentence):
        sentence_nlp = nlp(sentence)

        spans = []

        for ent in sentence_nlp.ents:
            if ent.label_ in labels:
                spans.append((ent.start_char, ent.end_char))

        if len(spans) == 0:
            return sentence
        
        result = ""

        start = 0
        for sp in spans:
            result += sentence[start:sp[0]]

            result += replacement_phrase

            start = sp[1]

        result += sentence[spans[-1][1]:]

        return result

    processed_all = []

    for sent in doc.sents:
        search = re.search(r'(\w+: )?(.*)', str(sent))

        main = search.group(2)
        prefix = search.group(1)

        processed = process(main)

        if prefix is not None:
            processed = prefix + processed

        processed_all.append(processed)

    return ' '.join(processed_all)

poisoners = {
    'central_noun': central_noun,
    'ner': ner_replace
}
