# NewStatementPlacement
Reccomends a placement for an argument on a deliberation map.
# Setup
Requirements: transformers 3.0.2, torch 1.2.0, numpy, pandas

**install git-lfs**
```
! sudo apt-get install git-lfs
```

**clone this repo**
Clong the whole repo would download the four argBERT models, taking about 5 GB of space
```
! git lfs clone https://github.com/lievan/NewStatementPlacement.git
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

**argBERT-type-included**

takes enhanced parent text snippets that contain IDEA, ISSUE, PRO, CON, as well as PARENT: in the training and prediction phase:

[enhanced parent text] + [new statement type + new statement text] --> argBERT-type-included --> taxonomic distance score

**argBERT-DriverlessCar-type-included**

map specific version of argBERT-type-included.

**argBERT-GlobalWarming-type-included** 

map specific version of argBERT-type-included.



Find the path to the desired argBERT model in the "pretrained_models" folder. argBERT models should contain a config.json file among others. The device specifies your runtime. 

```
argBERT_model = ArgumentMap.argBERT(model_name='path_to_argBERT_model', device='cuda')
```

# Initialize ArgumentMap

Argument map should be a tab-delimited text file. The initialize_map function has one required parameter which is the map name. It has two optional parameters:

dataset_length -- how many test samples you want, default is 30

bare_text -- Specifies if we have "enhanced" representations of the text. if you are using a "type-included' model, set bare_text=False. Default is bare_text=True. 

```
map_name = 'path_to_argmap_text_file'
arg_map, dataset, test_samples = ArgumentMap.initialize_map(map_name, bare_text=False)
```

# Fine tune argBERT to get a map specific model

The fine-tuned model would be saved under output_path. 

Be sure to set the optional parameter bare_text=False if you are using a type-included model. It is default True.

This fine-tuning process takes quite a while (30 min) because the validation processs. We go through 10 epochs and save the best version of the model. 

```
argBERT_model.fine_tune_model(dataset, test_samples, arg_map, output_path='./best_model', bare_text=False)
```

# Input new arguments

Initialize a new argument object with an entity (string), type (string), name (string), text (string), children=None, and bare_text=True/False

```
new_statement = Argument(entity=entity, type=arg_type, name=title, text=text, children=None, bare_text=False)
```

The "get_reccomendations" method returns a list of five suggestions. Each suggestion is also in the form of a list.

index 0 of the suggestion is the predicted distance

index 2 of the suggestion is the actual argument object

index 3 of the suggestion is the index of the suggested argument object from arg_map.argument_list 

The following code prints the text and entity of the top five suggestions:

```
top_suggestions = get_reccomendations(new_statement.text, arg_type, arg_map, argBERT_model, bare_text=bare_text)

for parent in top_suggestions:
 print("ARGUMENT TEXT: %s" % parent[1].text)
 print("ARGUMENT ENTITY: %s" % parent[1].entity)
 
```
The add_argument function adds an argument to the argument map. The first parameter is the new_statement object, the second parameter is the entity of the chosen parent

The add_new_training_data function creates new training data from this new statement. You can 

You can access this training data in arg_map.new_training_data

```
arg_map.add_argument(new_statement, true_placement)
arg_map.add_new_training_data(new_statement, true_placement, other_placements)
```

The input arguments function prompts the user to go through the above process on an infinite loop.

```
ArgumentMap.input_arguments(arg_map, argBERT_model)
```

If you just want the taxonomic distance prediction, using argBERT.predict_distance(). Add the argument type to the text snippet if you are using an includes-type model. 

```
parent = "IDEA We should slow down the adoption of driverless cars"

child = "CON driverless cars are safe to use now and human drivers are error prone"

taxonomic_distance = argBERT_model.predict_distance(parent, child)

```
or the following if you are using a standard argBERT model
```
parent = "We should slow down the adoption of driverless cars"

child = "driverless cars are safe to use now and human drivers are error prone"

taxonomic_distance = argBERT_model.predict_distance(parent, child)
```
