# NewStatementPlacement
Reccomends a placement for a new post on a deliberation map. 

For more info on argBERT, see https://docs.google.com/presentation/d/1XRmIDxqlrO5DoLFufQfgelbiVb55GroV0Em-t4OyQHM/edit?usp=sharing

# License
'Attribution-NonCommercial-ShareAlike 3.0'

# Setup
Requirements: transformers 3.0.2, torch 1.2.0, numpy, pandas

```
pip install requirements.txt

```

**Download argBERT models**
```
https://drive.google.com/drive/folders/1wz7BB7FcaS1V6mTBZM8XPpym-t89Ycx7?usp=sharing
```

**clone this repo**
```
git clone https://github.com/lievan/NewStatementPlacement.git
```

**import ArgumentMap module**
```
from NewStatementPlacement import ArgumentMap
```

# Initialize argBERT model

Models available:

**argBERT-standard**

takes two text snippets and predicts the distance:

[parent text] + [new statement text] --> argBERT-type-included --> taxonomic distance score

**argBERT-includes-special-tokens**

takes enhanced parent text snippets that contain IDEA, ISSUE, PRO, CON, as well as PARENT: in the training and prediction phase:

[enhanced parent text] + [new statement type + new statement text] --> argBERT-type-included --> taxonomic distance score

**argBERT-driverlesscar-specialtokens**

map specific version of argBERT-includes-special-tokens

**argBERT-globalwarming-specialtokens** 

map specific version of argBERT-includes-special-tokens

**argBERT-crowdcomputing-specialtokens**

map specific version of argBERT-includes-special-tokens




Find the path to the desired argBERT model in the "pretrained_models" folder. argBERT models should contain a config.json file among others. The device specifies your runtime. 

```
argBERT_model = ArgumentMap.argBERT(model_name='path_to_argBERT_model', device='cuda')
```

# Initialize Map

Argument map should be a tab-delimited text file. The initialize_map function has one required parameter which is the map name. 

Other parameters:

test_sample_length -- how many test samples you want from the map

bare_text -- Specifies if we have "enhanced" representations of the text. if you are using a "special-tokens" model, keep default bare_text=False. Else set bare_text=True

```
map_name = 'path_to_argmap_text_file'
map, dataset = ArgumentMap.initialize_map(map_name)
```

# Fine tune argBERT to get a map specific model

The fine-tuned model would be saved under output_path. 

bare_text parameter default False.

We go through 10 epochs and save the best version of the model. The best version of the model is already updated in argBERT_model after fine-tuning, but is also saved in the specified output_path

```
argBERT_model.fine_tune_model(dataset, map, output_path='./best_model')
```

# Input new arguments

There are two ways to input new posts. 

Either input_new_post, which prompts you to enter a title, text, type, and entity

```
map = ArgumentMap.input_new_post(map, argBERT_model)
```
Output:

```
NEW STATEMENT TITLE: We are not culpable
NEW STATEMENT TEXT: All animals have their natural effects on their environment
POST TYPE: IDEA
ENTITY: 1
 
PRINTING PLACEMENT SUGESTIONS--------------
 
POST TEXT: [ISSUE] Are humans responsible?  
POST ENTITY: E-3MAORN-135
POST TEXT: [IDEA] human activities have minimal impact on climate NIL
POST ENTITY: E-3MAORN-140
POST TEXT: [IDEA] human activities are causing global warming NIL
POST ENTITY: E-3MAORN-138
POST TEXT: [IDEA] Climate change will have minimal or positive impacts Climate is changing but this will not give negative consequences. 
POST ENTITY: E-3NNLOF-746
POST TEXT: [IDEA] Drop in volcanic activity NIL
POST ENTITY: E-3NNLOF-764
```

Or you can initialize a new Post object with an entity (string), type (string), name (string), text (string), children=None, and bare_text=True/False

The top_n parameter tells us how many reccomendations we want to recieve, default top_n=5

```
entity = 'newpost'
arg_type ='IDEA'
title='We are not culpable'
text='All animals have their unique effects on their environment'
new_post = ArgumentMap.Post(entity=entity, type=arg_type, name=title, text=text, children=None, bare_text=False)

reccomendations = ArgumentMap.get_reccomendations(new_post.text, new_post.type, map, argBERT_model, bare_text=True, top_n=5)

for rec in reccomendations:
  print(rec[1].text)
  print(rec[1].entity
  print(rec[2])
  print(rec[0])
```

The "get_reccomendations" method returns a list of five suggestions. Each suggestion is also in the form of a list.

index 0 of the suggestion is the predicted distance

index 1 of the suggestion is the actual argument object

index 2 of the suggestion is the index of the suggested argument object from arg_map.argument_list 


The add_argument function adds an argument to the argument map. The first parameter is the new_statement object, the second parameter is the entity of the chosen parent

The add_new_training_data function creates new training data from this new statement. You can 

You can access this training data in arg_map.new_training_data

```
arg_map.add_argument(new_statement, true_placement)
arg_map.add_new_training_data(new_statement, true_placement, other_placements)
```

If you just want the taxonomic distance prediction, using argBERT.predict_distance(). Add the argument type to the text snippet if you are using an includes-type model. 

```
parent = "[IDEA] We should slow down the adoption of driverless cars"

child = "[CON] driverless cars are safe to use now and human drivers are error prone"

taxonomic_distance = argBERT_model.predict_distance(parent, child)

```
or the following if you are using a standard argBERT model
```
parent = "We should slow down the adoption of driverless cars"

child = "driverless cars are safe to use now and human drivers are error prone"

taxonomic_distance = argBERT_model.predict_distance(parent, child)
```

# Semantic Similarity and Clustering

**Download word vectors**
Download word vectors

```
!wget http://nlp.stanford.edu/data/glove.6B.zip
```

```
!unzip glove.6B.zip
```

**Initialize SemanticSearcher model**

Specify path to 300D word vectors text file 

```
similarity_model = ArgumentMap.SemanticSearcher('pathto/glove.6B.300d.txt')
```

Initialize new map, remember to set bare_text=True

```
map, dataset, test_samples  = ArgumentMap.initialize_map('map-E-1QSHFV5-200.txt', bare_text=True)
```
Get top 5 recs based on semantic similarity

```
entity = 'newpost'
arg_type ='PRO'
title='Ice melting is a sign of warming'
text='Ice caps have been melting for years resulting in a loss of habitat'

new_post = ArgumentMap.Post(entity=entity, type=arg_type, name=title, text=text, children=None, bare_text=True)

recs = similarity_model.get_recs(new_post, map)

for rec in recs:
  print('NEW RECCOMENDATION ---')
  print(' ')
  print(rec[0])
  print(rec[1].text)
  print(rec[1].entity)
  print(' ')
 ```
 Output:
 
  ```
  NEW RECCOMENDATION ---
 
0.8413640260696411
ice is melting worldwide NIL
E-3OXYV4-608
 
NEW RECCOMENDATION ---
 
0.8234208226203918
ice melting puts the polar bear at grave risk While there is some uncertainty on current polar bear population trends, one thing is certain. No sea ice means no seals which means no polar bears. With Arctic sea ice retreating at an accelerating rate, the polar bear is at grave risk of extinction
E-3NNLOF-752
 
NEW RECCOMENDATION ---
 
0.8190104365348816
polar melting Detrimental effects include loss of polar bear habitat, and increased mobile ice hazards to shipping.&#10;&#10;&#10;
E-3OXYV4-597
 
NEW RECCOMENDATION ---
 
0.8114203214645386
Antarctic ice has shown long-term growth Antarctic sea ice has shown long term growth since satellites began measurements in 1979. 
E-3MAORN-161
 
NEW RECCOMENDATION ---
 
0.8043667674064636
GreenlandÃÂs interior ice sheet has grown from 1993 to 2002  An international team of climatologists &amp; oceanographers estimate GreenlandÃÂs interior ice sheet has grown 6cm per year in areas above 1 500m between 1992 and 2003. Lead author, Ola Johannessen says sheet growth is due to increased snowfall brought about by variability in regional atmospheric circulation or the so-called North Atlantic Oscillation (Source: Latest Scientific Studies Refute Fears of Greenland Melt). 
E-3NNLOF-730
 
```

You can also get reccomendations based on cluster

```
entity = 'newpost'
arg_type ='PRO'
title='Ice melting is a sign of warming'
text='Ice caps have been melting for years resulting in a loss of habitat'

new_post = ArgumentMap.Post(entity=entity, type=arg_type, name=title, text=text, children=None, bare_text=True)

recs = similarity_model.get_clusters(new_post, map, NUM_CLUSTERS=20)



for cluster in recs:
  if new_post in cluster:
    print('--NEW POST CLUSTER')
    print('              ')
    print('              ')
    for post in cluster:
      print(post.text)
    print('              ')
    print('              ')
  else:
    print('---OTHER CLUSTER')
    print('              ')
    print('              ')
    for post in cluster:
      print(post.text)
    print('              ')
    print('              ')
```
Output:

```
Other clusters
.
.
.

--NEW POST CLUSTER
              
              
ice is melting worldwide NIL
Arctic ice is disappearing Arctic sea ice has decreased steadily since the late 1970s.&#10;&#10;&#10;&lt;a target=newin href=
Greenland ice loss is acceleratin Greenland ice loss is accelerating (Velicogna 2009, van den Broeke et al 2009)
Glaciers are shrinking globally Glaciers are shrinking globally at an accelerating rate. 
polar melting Detrimental effects include loss of polar bear habitat, and increased mobile ice hazards to shipping.&#10;&#10;&#10;
The coastlines are losing ice While the Greenland interior is in mass balance, the coastlines are losing ice. Overall Greenland is losing ice mass at an accelerating rate. From 2002 to 2009, the rate of ice mass loss doubled.&lt;br&gt;&#10;&#10;&lt;a target=newin href=
Ice melting is a sign of warming Ice caps have been melting for years resulting in a loss of habitat

.
.
.
Other clusters
```
