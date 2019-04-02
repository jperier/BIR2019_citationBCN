# BIR2019_citationBCN

This repository contains the code used in the paper : **A preliminary study to compare deep learning and rule-based approaches for citations classification.** _Julien Perier-Camby, Marc Bertin, Iana Atanassova and Frederic Armetta._ (BIR) 2019

This work was done at the [LIRIS](https://liris.cnrs.fr/) and [ELICO](http://elico-recherche.ish-lyon.cnrs.fr/) research centers under the supervision of [Frédéric Armetta](https://liris.cnrs.fr/page-membre/frederic-armetta) and [Marc Bertin](http://elico-recherche.ish-lyon.cnrs.fr/membres/marc-bertin)

We used a BCN (Biattentive Classification Network) model with ELMo embeddings, implemented with AllenNLP & PyTorch, to classify citations by function.
 
# Requirements

This project works with Python 3 and the following packages :
`
allennlp (0.7.2)
torch (0.4.1)
numpy (1.15.4)
`


The data used in the paper was presented in [Measuring the Evolution of a Scientific Field through Citation Frames](http://jurgens.people.si.umich.edu/citation-function/) by Jurgens Jurgens et al., it is availiable [here](http://jurgens.people.si.umich.edu/citation-function/data/annotated-json-data.tar.gz). 


