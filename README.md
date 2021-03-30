# pseudo-ee-salience-estimation
Data and code for FLAIRS'34 paper "Evaluation of Unsupervised Entity and Event Salience Estimation".

## Data
For the raw data, please refer to [The New York Times Annotated Corpus](https://catalog.ldc.upenn.edu/LDC2008T19) and [Semantic Scholar Corpus](http://boston.lti.cs.cmu.edu/appendices/WWW2016/).

The pesudo annotation we used to report the numbers on our paper is avaliable at [entity-event-pseudo-annotation](https://figshare.com/articles/dataset/entity-event-pseudo-annotation/14337323?file=27362669).  
The annotation is formated in `jsonlines`, so each line represent a document with its pesudo salient entities and events.

For example, for document `#1013869` in NYT, 
```Json
{"docid": "1013869", 
"abstract": "Glen Slattery Op-Ed article mocks exhumation of bodies of famous people to test their DNA or for other odd and sundry reasons; opposes backhoe-induced resurrection, unless for something truly important, such as retrieving his only set of car keys (S)", 
"toks": [["Glen", "Slattery", "Op", "-", "Ed", "article", "mocks", "exhumation", "of", "bodies", "of", "famous", "people", "to", "test", "their", "DNA", "or", "for", "other", "odd", "and", "sundry", "reasons", ";", "opposes", "backhoe", "-", "induced", "resurrection", ",", "unless", "for", "something", "truly", "important", ",", "such", "as", "retrieving", "his", "only", "set", "of", "car", "keys", "(S)"]], 
"entities": [["Glen Slattery Op-Ed article", ["glen", "slattery", "op", "ed", "article"], 5, [0, 5]], ["exhumation", ["exhumation"], 7, [7, 7]], ["bodies", ["body"], 9, [9, 9]], ["famous people", ["famous", "people"], 12, [11, 12]], ["their DNA", ["their", "dna"], 16, [15, 16]], ["other odd and sundry reasons", ["other", "odd", "and", "sundry", "reason"], 23, [19, 23]], ["backhoe", ["backhoe"], 26, [26, 26]], ["something truly important", ["something", "truly", "important"], 33, [33, 35]], ["his only set", ["his", "only", "set"], 42, [40, 42]], ["car keys", ["car", "key"], 45, [44, 45]]], 
"events": [["resurrection", ["resurrection"], 29, [29, 29]], ["mocks", ["mock"], 6, [6, 6]], ["test", ["test"], 14, [14, 14]]], 
"event_arguments": [[], [["Glen Slattery Op-Ed article", ["glen", "slattery", "op", "ed", "article"], 5, [0, 5]], ["exhumation", ["exhumation"], 7, [7, 7]]], [["their DNA", ["their", "dna"], 16, [15, 16]]]]
}
```
- `abstract`: raw text of document abstract 
- `toks`: the tokenized abstract
- `entities`: a list of tuple `[salient entity, [tok_1, ..., tok_n], head token index, (phrase start index, phrase end index)]`
- `events`: a list of tuple `[salient event, [tok_1, ..., tok_n], head token index, (phrase start index, phrase end index)]`
- `event_arguments`: a list of correspond tuple of `events`, formatted as a list of tuple `[[argument1], [argument2], ...]`. Each event argument follows the same format and contents of `[argument, [tok1, ..., tok_n], head token index, (phrase start index, phrase end index)]`
