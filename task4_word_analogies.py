import spacy

def main():

    # load in the GloVe Common Crawl vectors
    nlp = spacy.load("en_vectors_glove_md")

    # initialize the adjectives
    adjective1 = "slow"
    comparative1 = "slower"
    adjective2 = "strong"

    # obtain the similarity between the adjective and its comparative
    adj1 = nlp(adjective1)
    comp1 = nlp(comparative1)
    print(("similarity between " + adjective1 + " and " + comparative1 + ": %s\n") % comp1.similarity(adj1))

    adj2 = nlp(adjective2)

    # gather all known words in pre-trained vectors, except the given words
    # take only the lowercased versions
    allWords = []
    for w in nlp.vocab:
        if w.has_vector and w.orth_.islower():
            if w.lower_ not in [adjective1, comparative1, adjective2]:
                allWords.append(w)

    # sort the word list by the similarity of each word against the original adjective2
    allWords.sort(key = lambda w: w.similarity(adj2), reverse=True)

    topN = 2

    for word in allWords[:topN]:   
        print(word.orth_)
        print(word.similarity(adj2))
        print("\n")

    print(adjective1 + " : " + comparative1 + " = " + adjective2 + " : (" + allWords[:topN][0].orth_ + ")")

if __name__ == '__main__':
    main()