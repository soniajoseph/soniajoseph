---
title: "Semantic Similarity Search for Phrases"
excerpt: Matching similar phrases given two corpuses
categories: [machine learning]
tags: [machine learning, natural language processing, projects]
header:
  teaser: assets/images/posts/word2vec.png
mathjax: "true"
comments: true
classes: wide
---

Word vector averaging is a way to find semantically similar sentences. The idea is simple: find the word embedding for each word using an algorithm like word2vec or GloVe, average the embeddings together to get a sentence vector, and match sentences with the most similar sentence vectors based off Euclidean distance or cosine similarity.

I recently became interested in extending the idea to the phrase level. If we have two sentences, can we co-locate all the semantically similar phrases from each sentence? 

For example, if I have the following two sentences:

> Sentence A:  Yucaipa owned Dominick's before selling the chain to Safeway in 1998 for $2.5 billion.
>
> Sentence B:  Yucaipa bought Dominick's in 1995 for $693 million and sold it to Safeway for $1.8 billion in 1998.

I want my algorithm to return similar phrases:

> "Yucaipa owned Dominick's" / "Yucapia bought Dominick's"
>
> "selling the chain to Safeway" / "sold it to Safeway"
>
> "in 1998 for $2.5 billion" / "for $1.8 billion in 1998"

To accomplish this, I wrote an algorithm with the following steps:

1) Parse the sentence into phrases using a statistical dependency parser. 

2) Create a word vector average for each phrase using word2vec embeddings.

3) Match the most similar phrases based off closest Euclidean distance between word vector averages.

I've explained the steps with code in the blog post below, and you can find a Jupyter notebook [here](https://github.com/soniajoseph/phrase-similarity).

## Load the data

Let's test our algorithm on the [Microsoft Research Paraphrase Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52398), which contains pairs of paraphrased sentences extracted from news articles. We'll create a list of paraphrase lists:

```python
def load_data(n=5):
  '''
  Function to get spacy model with medium neural net with the constituency parsing extension
  described in "Constituency Parsing with a Self-Attentive Encoder" (2018)

  Args:
      n (int): Number of paraphrase pairs to load.

  Returns:
      new_list: A nested list of n paraphrase pairs.
  '''
  target_url = 'https://raw.githubusercontent.com/wasiahmad/paraphrase_identification/master/dataset/msr-paraphrase-corpus/msr_paraphrase_data.txt'
  i = 0; data = []
  for sentence in urllib.request.urlopen(target_url):
      # skip first sentence
      if i == 0: i += 1; continue
      sentence = sentence.decode()
      sentence =  re.split(r'\t+', sentence)
      data.append(sentence[1])
      # increment counter for number of data
      i += 1
      if i > n*2: break 
    # turn into nested list
  new_list = []
  for i in range(0, len(data)-1, 2):
    new_list.append([data[i], data[i+1]])

  print("Data loaded")

  return new_list
```

We'll also load a medium [spaCy model](https://spacy.io/models) pretrained on English text:

```python
def get_model():
  '''
  Function to get spacy model with medium neural net.

  Args:
      None

  Returns:
      nlp: A loaded model with constituency parsing functionality.
  '''
  nlp = en_core_web_md.load()
  print("Model loaded")
  return nlp 
```

## Load phrase parser
The first step in solving this problem is parsing the sentences into phrases. We can use the Stanford Statistical Parser, which we can download [here](https://nlp.stanford.edu/software/lex-parser.shtml) to create a dependency parser that turns our sentences into phrases based off pre-trained probabilistic dependencies.

```python
def load_parser():
  '''
  Function to load Stanford parser,

  Args:
      None

  Returns:
      parser: return parser object
  '''
  path_exists = os.path.exists('/content/stanford-parser-full-2018-10-17') ## put your path to the downloaded parser here
  if path_exists:
    print(True)
  else:
    print("Load and configure StanfordParser")
    !wget https://nlp.stanford.edu/software/stanford-parser-full-2018-10-17.zip
    !unzip stanford-parser-full-2018-10-17.zip

  stanford_parser_dir = '/content/stanford-parser-full-2018-10-17'
  path_to_models = stanford_parser_dir  + "/stanford-parser-3.9.2-models.jar"
  path_to_jar = stanford_parser_dir  + "/stanford-parser.jar"
  parser=StanfordParser(path_to_models, path_to_jar)
  return parser
```

Using the parser, we can create a tree traversal object so that we can load the sentences into the parser and then traverse the  tree to extract relevant phrases. The traversal object gathers both noun and prepositional phrases.

```python
class Traverse():
  '''
  Traverse object to create trees to find noun phrases.
  To use, call traverse_tree() with input from the StanfordParser
  Then call the phrase_strings() function with self.phrases to get the noun phrases of
  the input sentence.
  '''
  def __init__(self):
    self.phrases = []
    
  def traverse_phrase(self, tree, phrases): 
      for subtree in tree:
          if type(subtree) == nltk.tree.Tree:
              self.traverse_phrase(subtree, phrases)
          else:
              phrases.append(subtree)

  # traverse the tree to gather noun phrases and prepositional phrases
  def traverse_tree(self, tree):
      for subtree in tree:
          if type(subtree) == nltk.tree.Tree:
              if subtree.label() == 'NP' or subtree.label() == 'PP':
                  self.traverse_phrase(subtree, self.phrases)
                  self.phrases.append('\n')
              else :
                  self.traverse_tree(subtree)

  # put noun phrases in list
  def phrase_strings(self, phrase_list):
    a = " ".join(phrase_list).split("\n")
    a = [i.strip() for i in a if i]
    return a
```

## Let's write a function to calculate a semantic measure for the phrase.

Now we need a metric to calculate the semantic measure of each phrase. I calculate the word vector of each word of the phrase and then average the words together. Notably, I did not normalize the word vectors before averaging because the different lengths give rise to a weighted average, which [Arefyev et al suggests is more accurate](https://arxiv.org/pdf/1805.09209.pdf).

While research regarding weighted vs unweighted word vector averages is scarce, one theory is that infrequent words are correlated with longer word vectors. Infrequent words may be more poorly represented by the embedding, which is based off distributional frequency, and so having a larger value in the total average may make the overall phrase vector more accurate.

```python
def wva(string):
    '''
    Finds document vector through an average of each word's vector.

    Args: 
      string (str): Input sentence

    Returns:
      array: Word vector average
    '''
    doc = nlp(string)
    wvs = np.array([doc[i].vector for i in range(len(doc))])
    return np.mean(wvs, axis=0) 
 ```

Now we will calculate the phrases with the greatest semantic similarity. The most common measures of semantic similarity are cosine similarity and Euclidean distance. Perhaps unconventionally, I chose the latter in order to capture the information in the word average lengths, which Arefyez et al. suggest is significant in representing word frequency.

```python
def match_wv_pair(phrasesA, phrasesB):
  '''
  Takes two lists of phrases from one sentence each and finds the smallest Euclidean distance for each pair's word vector (non-exclusive).

  Args:
  phraseA (list of str): List of parsed phrases from or sentence A
  phraseB (list of str): List of parsed phrases from sentence B to compare with sentence A 

  Returns:
  matches (list of str): Returns list of matches between the two phrase (surjectively, i.e. multiple phrases can have the same match).

  '''
  # get word vectors
  wva_a = []; wva_b = []
  for i in phrasesA:
    wva_a.append(wva(i))
  for j in phrasesB:
    wva_b.append(wva(j))

  # swap so that shortest is on the outer for loop
  if len(wva_a) > len(wva_b):
    temp = wva_a
    wva_a = wva_b
    wva_b = temp

    temp = phrasesA
    phrasesA = phrasesB
    phrasesB = temp

  matches = []
  for i in range(len(wva_a)):
    distances = []
    for j in range(len(wva_b)):
      distances.append(numpy.linalg.norm(wva_a[i] - wva_b[j]))
      # indices_total.append(np.argsort(distances)[0])
    matches.append("Sentence A: " + phrasesA[i] + "\n Sentence B:" + phrasesB[np.argsort(distances)[0]] + "\n Euclidean Distance:" + str(np.sort(distances)[0]) + "\n")

  return matches
 ```

## Run the final function

Finally, I wrote a function that runs the functions above: take in our data and parser, parse the sentences into phrases, turn the phrases into word vector averages, then match word vector averages based on Euclidean distance. The function prints out semantically similar phrases for each paraphrase-pair of the original dataset.

```python
def find_similar_phrases(data, parser):
  '''
  Uses Traverse object to create phrase trees of each sentence, then recurses through tree to collect noun phrases.

  Args: 
  string (list of strings): List of string lists in the format [[a,b],[c,d]] to find similarity between each pair.

  Returns:
  Nothing (prints out original sentences, a phrases, b phrases, matching phrases, and their Euclidean distance)
  '''
  for a, b in data:
    print("Original sentences:")
    print("Sentence A: ", a)
    print("Sentence B: ", b)
    print()

    ta = Traverse()
    phrasesA = parser.raw_parse(a)
    ta.traverse_tree(phrasesA)
    a = ta.phrases
    a = ta.phrase_strings(a)
    
    tb = Traverse()
    phrasesB = parser.raw_parse(b)
    tb.traverse_tree(phrasesB)
    b = tb.phrases
    b = tb.phrase_strings(b)
    
    print("A phrases:", a)
    print("B phrases:", b)
    print()

    matches = match_wv_pair(a,b)
    print("Similar phrases:")
    for i in matches:
      print(i)
    print()
    print()
```

The function returns:

```python

Original sentences:
Sentence A:  Yucaipa owned Dominick's before selling the chain to Safeway in 1998 for $2.5 billion.
Sentence B:  Yucaipa bought Dominick's in 1995 for $693 million and sold it to Safeway for $1.8 billion in 1998.

A phrases: ['Yucaipa', 'Dominick', 'before selling the chain to Safeway in 1998 for $ 2.5 billion']
B phrases: ['Yucaipa', "Dominick 's in 1995", 'for $ 693 million', 'it', 'to Safeway', 'for $ 1.8 billion in 1998']

Similar phrases:
Sentence A: Yucaipa
Sentence B:Yucaipa
Euclidean Distance:0.0

Sentence A: Dominick
Sentence B: Dominick 's in 1995
Euclidean Distance:5.5339174

Sentence A: before selling the chain to Safeway in 1998 for $ 2.5 billion
Sentence B: for $ 1.8 billion in 1998
Euclidean Distance: 2.0490212


Original sentences:
Sentence A:  They had published an advertisement on the Internet on June 10, offering the cargo for sale, he added.
Sentence B:  On June 10, the ship's owners had published an advertisement on the Internet, offering the explosives for sale.

A phrases: ['They', 'an advertisement on the Internet on June 10', 'the cargo for sale', 'he']
B phrases: ['On June 10', "the ship 's owners", 'an advertisement on the Internet', 'the explosives', 'for sale']

Similar phrases:
Sentence A: They
Sentence B: the ship 's owners
Euclidean Distance: 4.217091

Sentence A: an advertisement on the Internet on June 10
Sentence B: an advertisement on the Internet
Euclidean Distance: 1.383867

Sentence A: the cargo for sale
Sentence B: for sale
Euclidean Distance: 2.6837785

Sentence A: he
Sentence B: the ship 's owners
Euclidean Distance:5.162956
```

To improve the function, we could train a dependency parser on the corpus to extract better phrases. We could also experiment with other ways of traversing the existing tree and trying other similarity measures, including cosine similarity, Mahalanobis distance, and relaxed word mover's distance.

I am also interested in experimenting with accuracy in normalized vs. unnormalized word vector averages and the relationship between vector length and word frequency.

I hope this tutorial was helpful to you! The full code for this post is available on Github [here](https://github.com/soniajoseph/phrase-similarity).