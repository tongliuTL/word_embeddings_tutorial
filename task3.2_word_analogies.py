import spacy, numpy

def similarity(wv1, wv2):

    if (numpy.linalg.norm(wv1) == 0) or (numpy.linalg.norm(wv2) == 0):
        return 0.0    
    return numpy.dot(wv1, wv2) / numpy.linalg.norm(wv1) * numpy.linalg.norm(wv2)    

def main():

    nlp = spacy.load("en_vectors_glove_md")

    king = nlp.vocab['king']
    man = nlp.vocab['man']
    woman = nlp.vocab['woman']

    result = king.vector - man.vector + woman.vector

    words_by_similarity = []

    for w in nlp.vocab:
        if (w.has_vector) and (w.orth_.islower()) and (w.lower_ not in ['man', 'woman', 'king']):
                words_by_similarity.append(w)
    words_by_similarity.sort(key = lambda w: similarity(w.vector, result), reverse=True)

    for word in words_by_similarity[:10]:   
        print(word.orth_, word.similarity(nlp('king')))

if __name__ == '__main__':
    main()