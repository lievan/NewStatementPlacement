import pandas as pd
import itertools
import re
import random

from transformers import RobertaForSequenceClassification, AdamW, RobertaConfig
from transformers import RobertaTokenizer
from transformers import get_linear_schedule_with_warmup
import torch
import os
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import nn
import numpy as np


class argBERT(nn.Module):

    def __init__(self, model_name: str = 'argBERT-type-included', device: str = None):
        super(argBERT, self).__init__()
        self.argBERT = RobertaForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name, bos_token='<s>', eos_token='</s>',
                                                          unk_token='<unk>',
                                                          pad_token='<pad>', mask_token='mask_token', sep_token="</s>",
                                                          cls_token='<s>')
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.argBERT.to(self.device)
        self.best_num_correct = 0
        self.smallest_total_misses = 1000


    def fine_tune_model(self, training_data, test_samples, arg_map, output_path):
        seed_val = 32
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        train_dataloader = self.get_dataloaders(training_data)
        epochs = 10

        for epoch_i in range(0, epochs):

          print("")
          print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
          print('Training...')
          self.train_data(train_dataloader)
          print("")
          print("Running Validation...")

          self.argBERT.eval()

          num_correct, total_misses = evaluate_map(test_samples, arg_map, self)

          if num_correct > self.best_num_correct:
            self.best_num_correct = num_correct
            print("Saving new model ------")
            self.save_model(output_path)
            self.smallest_total_misses = total_misses
          elif num_correct == self.best_num_correct:
            if total_misses < self.smallest_total_misses:
              self.smallest_total_misses = total_misses
              print("Saving new model ------")
              self.save_model(output_path)

        print("loading best model...")
        self.argBERT = self.load_model(output_path)

    def train_data(self, train_dataloader, epochs=10):

        loss_fn = nn.MSELoss().to(self.device)

        total_steps = len(train_dataloader) * epochs

        optimizer = AdamW(self.argBERT.parameters(),
                          lr=2e-5,
                          eps=1e-8
                          )

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)

        self.argBERT.train()
        total_train_loss = 0

        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            self.argBERT.zero_grad()

            _, logits = self.argBERT(b_input_ids,
                                     token_type_ids=None,
                                     attention_mask=b_input_mask,
                                     labels=b_labels)
            logits = logits.type(torch.float32)
            b_labels = b_labels.type(torch.float32)
            loss = loss_fn(logits, b_labels.view(-1, 1))

            total_train_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.argBERT.parameters(), 1.0)

            optimizer.step()

            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))

    def get_dataloaders(self, dataset):
        input_ids = []
        attention_masks = []
        labels = []

        for i in range(len(dataset)):
            dataset[i][0] = str(dataset[i][0])
            dataset[i][1] = str(dataset[i][1])
            dataset[i][2] = float(dataset[i][2])
            labels.append(dataset[i][2])

        for data in dataset:
            parent = data[0]
            child = data[1]

            encoded_dict = self.tokenizer.encode_plus(
                child,  # parent/child to encode
                parent,
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=128,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
                truncation="longest_first"
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)

        train_dataset = TensorDataset(input_ids, attention_masks, labels)

        batch_size = 32

        train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler=RandomSampler(train_dataset),  # Select batches randomly
            batch_size=batch_size  # Trains with this batch size.
        )

        return train_dataloader

    def save_model(self, output_dir):
      if not os.path.exists(output_dir):
        os.makedirs(output_dir)
      print("Saving model to %s" % output_dir)
      model_to_save = self.argBERT.module if hasattr(self.argBERT, 'module') else self.argBERT
      model_to_save.save_pretrained(output_dir)
      self.tokenizer.save_pretrained(output_dir)

    def load_model(self, output_dir):
      loaded_model=RobertaForSequenceClassification.from_pretrained(output_dir)
      tokenizer=RobertaTokenizer.from_pretrained(output_dir)
      loaded_model.to(self.device)
      return loaded_model

    def predict_distance(self, parent_text, child_text):
        self.argBERT.eval()
        encoded_input = self.tokenizer.encode_plus(
            child_text,  # parent/child to encode
            parent_text,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=128,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
            truncation="longest_first"
        )

        input_ids = torch.tensor(encoded_input['input_ids']).to(self.device)
        attention_masks = torch.tensor(encoded_input['attention_mask']).to(self.device)
        with torch.no_grad():
            output = self.argBERT(input_ids,
                                  token_type_ids=None,
                                  attention_mask=attention_masks)

        logits = output[0].detach().cpu().numpy()

        return logits[0]


class Argument:
    def __init__(self, entity, type, name, text, children):
        self.entity = entity
        self.type = type
        self.topic = name
        if str(text) != "nan" and name is not None:
            text = re.sub(r'http\S+', '', text)
            self.text = type + " " + name + " " + text
        elif name is None:
            text = re.sub(r'http\S+', '', text)
            self.text = type + " " + text
        else:
            self.text = type + " " + name
        child_list = []
        if children is not None:
            children = children.strip(')')
            children = children.strip('(')
            child_list = children.split(' ')
        self.children_entities = child_list
        self.parent = None
        self.children_objs = []

    def initialize_parent(self, parent):
        self.parent = parent

    def initialize_children(self, children):
        self.children_objs = children


def get_parent_plus_children(argument, compared_child):
    full_text = ""
    full_text += argument.text
    if argument.parent is not None:
        full_text += " PARENT: "
        parent_text = argument.parent.text
        if len(argument.parent.text.split()) > 10:
            text_list = argument.parent.text.split()
            sep = ' '
            parent_text = sep.join(text_list[:10])
        full_text += parent_text
    for child in argument.children_objs:
        if child.text != compared_child:
            full_text += " "
            full_text += child.text
    return full_text


def possible_response(parent, child):
    if child.type == "IDEA" and (parent.type == "ISSUE" or parent.type == "IDEA"):
        return True
    elif child.type == "ISSUE" and parent.type == "IDEA":
        return True
    elif (child.type == "PRO" or child.type == "CON") and (
            parent.type == "PRO" or parent.type == "CON" or parent.type == "IDEA"):
        return True
    return False


def get_ancestor_generations(arg1, arg2, argument_list):
    arg2_list = []
    arg1_list = []
    arg2_list.append(arg2)
    arg1_list.append(arg1)
    index2 = 0
    index1 = 0
    for i in range(len(argument_list)):
        arg2_next_parent = arg2_list[index2].parent
        arg1_next_parent = arg1_list[index1].parent

        if arg2_next_parent == arg1:
            return len(arg2_list)
        if arg1_next_parent == arg2_next_parent:
            if len(arg2_list) + len(arg1_list) == 2:
                return 2

        if arg2_next_parent is not None:
            arg2_list.append(arg2_next_parent)
            index2 += 1
        if arg1_next_parent is not None:
            arg1_list.append(arg1_next_parent)
            index1 += 1

    return taxonomic_distance(arg1, arg2, argument_list) * -1


def taxonomic_distance(arg1, arg2, argument_list):
    arg1_list = []
    arg2_list = []
    arg1_list.append(arg1)
    arg2_list.append(arg2)
    index1 = 0
    index2 = 0
    for i in range(len(argument_list)):

        arg2_next_parent = arg2_list[index2].parent
        arg1_next_parent = arg1_list[index1].parent

        if arg1_next_parent == arg2_next_parent:
            return len(arg2_list) + len(arg1_list)
        elif arg2_next_parent in arg1_list:
            return len(arg2_list) + len(arg1_list[:arg1_list.index(arg2_next_parent)])
        elif arg1_next_parent in arg2_list:
            return len(arg1_list) + len(arg2_list[:arg2_list.index(arg1_next_parent)])
        elif arg1_next_parent == arg2:
            return len(arg2_list)
        elif arg2_next_parent == arg1:
            return len(arg1_list)

        if arg1_next_parent is not None:
            arg1_list.append(arg1_next_parent)
            index1 += 1
        if arg2_next_parent is not None:
            arg2_list.append(arg2_next_parent)
            index2 += 1
    return len(argument_list)


def arguments_to_pairs(argument_list, include_children):
    taxonomic_distance_list = []
    combinations_object = itertools.combinations(argument_list, 2)
    combinations_list = list(combinations_object)
    pairs = []
    for combo in combinations_list:
        if possible_response(combo[0], combo[1]):
            distance = taxonomic_distance(combo[0], combo[1], argument_list)

            if include_children:
                pairs.append([get_parent_plus_children(combo[0], combo[1].text), combo[1].text, distance])
            else:
                pairs.append([combo[0].text, combo[1].text, distance])

            taxonomic_distance_list.append(distance)
        elif possible_response(combo[1], combo[0]):

            distance = taxonomic_distance(combo[1], combo[0], argument_list)
            if include_children:
                pairs.append([get_parent_plus_children(combo[1], combo[0].text), combo[0].text, distance])
            else:
                pairs.append([combo[1].text, combo[0].text, distance])

            taxonomic_distance_list.append(distance)
    return pairs, taxonomic_distance_list


def balance_data(dataset, max_length):
    # store data increment value
    splits = []
    counter = 1
    while counter <= max_length:
        splits.append([counter, []])
        counter += 1

    for data in dataset:
        for i in range(len(splits)):
            if data[2] == splits[i][0]:
                splits[i][1].append(data)

    num_parents = len(splits[0][1])  # length of the number of parent args there are
    test_data = []
    verification = []
    for i in range(len(splits)):
        verification.append([splits[i][0], []])
        if len(splits[i][1]) > num_parents:
            for a in range(num_parents):
                ran = random.randrange(len(splits[i][1]))
                test_data.append(splits[i][1][ran])
                verification[i][1].append(splits[i][1][ran])
        else:
            test_data += splits[i][1]
            verification[i][1] += splits[i][1]

    for data_range in verification:
        print("You have %s values for %s taxonomic distance" % (len(data_range[1]), data_range[0]))

    return test_data


def get_arg_from_entity(parent_entity, arg_map):
    for arg in arg_map.argument_list:
        if arg.entity == parent_entity:
            return arg
    return None


class Argument_Map:
    def __init__(self, map_name):
        self.argument_list = []
        self.new_training_data = []
        self.max_traverse_steps = 0
        test_df_dc = pd.read_csv(map_name, delimiter="\t", header=0)
        entities = test_df_dc.Entity.values
        types = test_df_dc.Type.values
        names = test_df_dc.Name.values
        descriptions = test_df_dc.Description.values
        children = test_df_dc.Children.values

        for entity, type, name, description, childs in zip(entities, types, names, descriptions, children):
            entity = entity.strip('(')
            entity = entity.strip(')')
            self.argument_list.append(Argument(entity, type, name, description, childs))
        for i in range(len(self.argument_list)):
            children_objs = []
            parent = None
            for arg in self.argument_list:
                if arg.entity in self.argument_list[i].children_entities:
                    children_objs.append(arg)
                if self.argument_list[i].entity in arg.children_entities:
                    parent = arg
            self.argument_list[i].initialize_children(children_objs)
            self.argument_list[i].initialize_parent(parent)

    def add_argument(self, new_statement, parent_entity):
        parent = get_arg_from_entity(parent_entity, self)
        new_statement.initialize_parent(parent)
        self.argument_list.append(new_statement)
        self.argument_list[self.argument_list.index(parent)].children_objs.append(new_statement)

    def add_new_training_data(self, new_statement, parent_entity, viable_placement_entities):
        max_steps = 0

        parent = get_arg_from_entity(parent_entity, self)

        new_data = [[new_statement.text, parent.text, 1]]

        for entity in viable_placement_entities:
            viable_parent = get_arg_from_entity(parent_entity, self)
            new_data.append([new_statement.text, viable_parent.text, 1])

        for arg in self.argument_list:
            if arg.entity not in viable_placement_entities and arg.entity != parent_entity:
                distance = taxonomic_distance(arg, new_statement, self.argument_list)
                new_data.append([arg.text, new_statement.text, distance])
                if distance > max_steps:
                    max_steps = distance

        if max_steps > self.max_traverse_steps:
            self.max_traverse_steps = max_steps

        self.new_training_data = self.new_training_data + new_data

    def create_dataset(self, test_size, include_children):
        random_argument_list = self.argument_list
        random.shuffle(random_argument_list)
        test_data = []
        train_data = []
        count = 0
        for arg in self.argument_list:
            if not arg.children_objs and count < test_size:
                test_data.append(arg)
                count += 1
            else:
                train_data.append(arg)
        training_data, taxonomic_distance_list = arguments_to_pairs(train_data, include_children)
        self.max_traverse_steps = max(taxonomic_distance_list)
        return training_data, test_data


def get_reccomendations(argument, type, argument_map, argBERT_model):
    parent_type = []
    if type == "IDEA":
        parent_type.append("ISSUE")
        parent_type.append("IDEA")
    elif type == "ISSUE":
        parent_type.append("IDEA")
    elif type == "PRO" or type == "CON":
        parent_type.append("IDEA")
        parent_type.append("PRO")
        parent_type.append("CON")

    recs = [[len(argument_map.argument_list), None, 0],
            [len(argument_map.argument_list), None, 0],
            [len(argument_map.argument_list), None, 0],
            [len(argument_map.argument_list), None, 0],
            [len(argument_map.argument_list), None, 0]]

    for potential_parent in argument_map.argument_list:
        if potential_parent.text != argument and potential_parent.type in parent_type:

            largest = 0
            worst_index = 0

            for i in range(len(recs)):
                if recs[i][0] > largest:
                    largest = recs[i][0]
                    worst_index = i

            distance = argBERT_model.predict_distance(get_parent_plus_children(potential_parent, argument), argument)

            if distance < recs[worst_index][0]:
                recs[worst_index] = [distance, potential_parent, argument_map.argument_list.index(potential_parent)]

    return recs


def input_arguments(arg_map, argBERT_model):
    while True:
        title = input("NEW STATEMENT TITLE: ")
        text = input("NEW STATEMENT TEXT: ")
        arg_type = input("ARGUMENT TYPE: ")
        entity = input("ENTITY: ")

        new_statement = Argument(entity=entity, type=arg_type, name=title, text=text, children=None)

        top_suggestions = get_reccomendations(new_statement.text, arg_type, arg_map, argBERT_model)

        print(" ")
        print("PRINTING PLACEMENT SUGESTIONS--------------")
        print(" ")

        for parent in top_suggestions:
            print("ARGUMENT TEXT: %s" % parent[1].text)

            print("ARGUMENT ENTITY: %s" % parent[1].entity)

        print(" ")

        print(" ----------------------------------------------------------")

        true_placement = input("entity of suggested placement: ")
        potential_other_placements = input("Did any other suggestions 'make sense?' (YES/SKIP)")

        other_placements = []

        while potential_other_placements == "YES":
            other_placement = input("Type entity of other correct suggestions, or type 'SKIP' if there are none left")
            if other_placement == "SKIP":
                potential_other_placements = ""
            else:
                other_placements.append(other_placement)

        arg_map.add_argument(new_statement, true_placement)
        arg_map.add_new_training_data(new_statement, true_placement, other_placements)

    return arg_map


def initialize_map(map_name, dataset_length=30):
    map_name = 'Argument map.txt'

    arg_map = Argument_Map(map_name)

    print("Argument map initialized: displaying first 10 arguments")

    print("---------------")

    for i in range(10):
        print(arg_map.argument_list[i].text)

    print("Data/training set --------")

    dataset, test_samples = arg_map.create_dataset(30, include_children=True)

    max_steps = 0
    for sample in dataset:
        if sample[2] > max_steps:
            max_steps = sample[2]

    dataset = balance_data(dataset, max_steps)

    print(test_samples[0].text)
    print(dataset[0])

    return arg_map, dataset, test_samples


def evaluate_map(test_samples, arg_map, argBERT_model, display_results_only=True):
    num_correct = 0
    total_average_distance = 0
    average_smallest_distance = 0
    same_branch = 0
    ancestors = 0
    total_misses = 0
    total_ancestor_distance = 0
    total_parent_score = 0

    for arg in test_samples:
        arg_types = ["IDEA", "PRO", "CON", "ISSUE"]
        ancestor_distances = []
        if arg.type in arg_types:

            if arg.parent is not None:
                parent_score = argBERT_model.predict_distance(get_parent_plus_children(arg.parent, arg.text), arg.text)
                total_parent_score += parent_score
                if not display_results_only:
                    print(" ----------- NEW ARG -----------")
                    print(arg.text)
                    print(arg.parent.text)
                    print(parent_score)
                    print("--------------")

            parent_recs = get_reccomendations(arg.text, arg.type, arg_map, argBERT_model)

            smallest_distance = arg_map.max_traverse_steps

            for parent in parent_recs:
                distance = 1

                if arg in parent[1].children_objs:
                    num_correct += 1
                else:
                    distance = taxonomic_distance(parent[1], arg, arg_map.argument_list)

                total_average_distance += distance
                generations = get_ancestor_generations(parent[1], arg, arg_map.argument_list)
                ancestor_distances.append(generations)

                if not display_results_only:
                    print(" ")
                    print(parent[1].text)
                    print(" ")
                    print("Distance: %s" % distance)
                    print("Prediction: %s" % parent[0])
                    print("Ancestor Generations: %s " % generations)

                if distance < smallest_distance:
                    smallest_distance = distance
            if not display_results_only:
                print("-----smallest distance: %s" % smallest_distance)

            has_ancestor = False

            for dis in ancestor_distances:
                if dis > 0:
                    has_ancestor = True

            if has_ancestor:
                ancestors += 1
                best_ancestor = arg_map.max_traverse_steps
                for dis in ancestor_distances:
                    if best_ancestor > dis > 0:
                        best_ancestor = dis
                total_ancestor_distance += best_ancestor

            if smallest_distance <= 2:
                same_branch += 1
            elif not has_ancestor:
                total_misses += 1

            average_smallest_distance += smallest_distance

    total_average_distance = total_average_distance / (5 * len(test_samples))
    average_smallest_distance = average_smallest_distance / (len(test_samples))
    total_ancestor_distance = total_ancestor_distance / ancestors
    total_parent_score = total_parent_score / len(test_samples)

    print("-----------TESTING STATS---------------")
    print("NUMBER CORRECT: %s / %s " % (num_correct, len(test_samples)))
    print("TOTAL AVERAGE DISTANCE: %s out of max %s " % (total_average_distance, arg_map.max_traverse_steps))
    print("AVERAGE SMALLEST DISTANCE: %s out of max %s" % (average_smallest_distance, arg_map.max_traverse_steps))
    print("SAME BRANCH: %s / %s" % (same_branch, len(test_samples)))
    print("# OF ANCESTORS: %s / %s" % (ancestors, len(test_samples)))
    print("# AVERAGE GENERATIONS OF BEST ANCESTOR RECCOMENDED: %s " % total_ancestor_distance)
    print("# OF TOTAL MISSES: %s " % total_misses)
    print("# PARENT DISTANCE: %s " % total_parent_score)

    return num_correct, total_misses
