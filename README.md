# NewStatementPlacement
Reccomends a placement for an argument on a deliberation map. LICENSE:'Attribution-NonCommercial-ShareAlike 3.0'
# Setup
Requirements: transformers 3.0.2, torch 1.2.0, numpy, pandas

```
pip install requirements.txt

```

**install git-lfs**
```
sudo apt-get install git-lfs
```

**clone this repo**
Clong the whole repo would download the four argBERT models, taking about 5 GB of space
```
git lfs clone https://github.com/lievan/NewStatementPlacement.git
```

**import ArgumentMap module**
```
from NewStatementPlacement import ArgumentMap
```

# Initialize argBERT model

There are four argBERT models available:

**argBERT-standard**

takes two text snippets and predicts the distance:

[parent text] + [new statement text] --> argBERT-type-included --> taxonomic distance score

**argBERT-includes-special-tokens**

takes enhanced parent text snippets that contain IDEA, ISSUE, PRO, CON, as well as PARENT: in the training and prediction phase:

[enhanced parent text] + [new statement type + new statement text] --> argBERT-type-included --> taxonomic distance score

**argBERT-driverlessdar-specialtokens**

map specific version of argBERT-type-included.

**argBERT-globalwarming-specialtokens** 

map specific version of argBERT-type-included.



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
map, dataset, test_samples = ArgumentMap.initialize_map(map_name, test_sample_length=30)
```

# Fine tune argBERT to get a map specific model

The fine-tuned model would be saved under output_path. 

bare_text parameter default False.

We go through 10 epochs and save the best version of the model. The best version of the model is already updated in argBERT_model after fine-tuning, but is also saved in the specified output_path

```
argBERT_model.fine_tune_model(dataset, test_samples, arg_map, output_path='./best_model', bare_text=False)
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

Or you can initialize a new argument object with an entity (string), type (string), name (string), text (string), children=None, and bare_text=True/False

The top_n parameter tells us how many reccomendations we want to recieve 

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
