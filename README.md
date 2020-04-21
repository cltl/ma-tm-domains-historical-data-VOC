# Mining the VOC data for historical research

The VOC takes an important place in Dutch history, for its role in the development of the Dutch Republic in the 17th and 18th centuries. For historians enquiring about the VOC, data are in plenty: we notably dispose of two corpora, [*Pieter van Dam’s Beschryvinge van de Oostindische Compagnie*](http://resources.huygens.knaw.nl/vocbeschrijvingvandam) and the [*Generale Missiven van Gouverneurs-Generaal en Raden aan Heren XVII der Verenigde Oostindische Compagnie*](http://resources.huygens.knaw.nl/vocgeneralemissiven). The first corpus relates the history of the VOC in the 17th century and was written on request of the VOC by Pieter van Dam at the end of that century. The second corpus is a collection of VOC-post reports to the VOC over the period 1610-1761, edited and completed with notes between 1960 and 2007.

Processing these corpora for NLP and text mining presents interesting challenges:

* non-standard language: 17th-century Dutch differs from modern Dutch, and this is aggravated by language change in the case the *Generale Missiven*; NLP models trained on modern data are likely to underperform when applied to these data.
* low-resource language: while plenty of historical Dutch data are available, annotations are lacking, and one cannot simply train supervised models on them.

For this project, the VOC corpus was mined with a view to answering historical-research questions like:

* How did trade evolve during the activities of the VOC? What goods were traded and in what amounts?
* What alliances were made over time? What networks emerged?

## Project goal and outcomes

The practical goal of this project was to develop timelines for prominent people, traded goods and places. The code for the timelines can be found in the `Q-EMNLP` and the `final_version_map` folders:

* `Q-EMNLP`: timelines for people and goods
* `final_version_map`: spatial timeline of locations

The low-resource issue for NER training was solved by annotating a subset of the VOC corpus, and training a charCNN-BiLSTM-CRF NER model on the resulting data. The annotated data can be found under `Q-EMNLP/data`.

### Note

A number of large files have been kept out of this repository (see `.gitignore`).

## Acknowledgements

This project was created as part of the course Text Mining Domains, offered by CLTL at Vrije Universiteit, Amsterdam.

The input data for the project were prepared by the *Instituut voor de Nederlandse Taal*, and further processed at the CLTL.

## Project Team

Stan Frinking

Quincy Liem

Luca Meima

Mehul Verma

Eva Zegelaar

Supervisor: Sophie Arnoult.

Computational Lexicology & Terminology Lab, Vrije Universiteit Amsterdam.




