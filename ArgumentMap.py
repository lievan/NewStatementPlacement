import pandas as pd
import itertools
import re
import random
import torch

import gensim
from scipy import spatial
import csv
import gensim.downloader as api
import nltk
from nltk.cluster import KMeansClusterer
from nltk.corpus import stopwords

from sklearn.cluster import AgglomerativeClustering
from transformers import RobertaForSequenceClassification, AdamW, RobertaConfig
from transformers import RobertaTokenizer
from transformers import get_linear_schedule_with_warmup
import torch
import os
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import nn
import numpy as np

nltk.download('stopwords')

stopword = set(stopwords.words('english'))


class SemanticSearcher:
    def __init__(self, path):

        self.embeddings_dict = {}
        print('Loading word vectors...')
        with open(path, 'r') as f:
            for line in f:
                values = line.split()
                split = len(values) - 300
                word = values[split - 1]
                vector = np.asarray(values[split:], "float32")
                self.embeddings_dict[word] = vector
        print('Word vectors loaded')

    def get_mean_vector(self, text):
        '''
          inputs: text (string)

            -we preprocess (remove stopwords, seperate into tokens) the text input
            -obtain word vectors for every token in text
            -sum the word vectors

          outputs: vector representation of text (numpy array)

        '''

        words = self.preprocess_text(text)

        word_list = []
        for word in words:
            if word in self.embeddings_dict:
                word_list.append(word)

        word_vectors = []
        for word in word_list:
            word_vectors.append(self.embeddings_dict[word])
        return np.sum(word_vectors, axis=0)

    def predict_distance(self, post1, post2):
        '''
          inputs: post1 (string), post1 (string)

            -we obtain the mean vector for post 1 and post 2
            -compute cosine distance between two vectors

          outputs: cosine similarity (float)

        '''

        vector0 = self.get_mean_vector(post1)
        vector1 = self.get_mean_vector(post2)
        norm_vector0 = vector0
        norm_vector1 = vector1
        similarity = 1 - spatial.distance.cosine(norm_vector0, norm_vector1)
        return similarity

    def preprocess_text(self, text):
        # helper method to preprocess text
        text = gensim.utils.simple_preprocess(text)
        return_text = []
        for tok in text:
            if not tok in stopword:
                return_text.append(tok)
        return return_text

    def get_clusters(self, new_post, map, NUM_CLUSTERS):
        '''
          inputs: new_post (Post object), map (Map object), NUM_CLUSTERS (int)

            -we obtain embedding representation for each post in the map
            -we obtain embedding representation for the new post
            -use k-means cluster, with 100 repeats, on the embeddings

          outputs: clusters (list), with each index of the list storing a cluster of posts

        '''
        output_objs = []
        post_embeddings = []
        for potential_parent in map.post_list:
            if potential_parent != new_post:
                vec = self.get_mean_vector(potential_parent.text)
                if isinstance(vec, np.ndarray):
                    post_embeddings.append(vec)
                    output_objs.append(potential_parent)

        post_embeddings.append(self.get_mean_vector(new_post.text))
        output_objs.append(new_post)

        kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=100,
                                     avoid_empty_clusters=True)
        assigned_clusters = kclusterer.cluster(post_embeddings, assign_clusters=True)

        clusters = []
        for i in range(NUM_CLUSTERS):
            clusters.append([])

        for assignment, output_obj in zip(assigned_clusters, output_objs):
            clusters[assignment].append(output_obj)

        return clusters

    def get_recs(self, new_post, map, top_n=5):
        '''
          inputs: new_post (Post object), map (Map object), top_n (int)

            -we calculate cosine similarity between the new post and every possible parent
            -we sort this list of predictions
            -we return the top_n reccomendations from the sorted list

          outputs: clusters (list), with each index of the list storing [distance prediction, parent post object, index of parent post object]

        '''
        parent_type = []
        if new_post.type == "IDEA":
            parent_type.append("ISSUE")
            parent_type.append("IDEA")
        elif new_post.type == "ISSUE":
            parent_type.append("IDEA")
        elif new_post.type == "PRO" or type == "CON":
            parent_type.append("IDEA")
            parent_type.append("PRO")
            parent_type.append("CON")

        similarity_scores = []
        output_examples = []
        for potential_parent in map.post_list:
            if potential_parent.type in parent_type:
                post1_text = get_parent_plus_children(potential_parent, new_post, bare_text=True)
                post2_text = new_post.text

                output_examples.append([self.predict_distance(post1_text, post2_text), potential_parent,
                                        map.post_list.index(potential_parent)])

        output_examples = sorted(output_examples, key=lambda x: x[0], reverse=True)

        return output_examples[:top_n]


class argBERT(nn.Module):

    def __init__(self, model_name, device: str = None):

        '''
          inputs: model_name (str), device: (str)

              -initializes argBERT model from the RobertaForSequenceClassification class in transformers
              -initializes tokenizer for argBERT model
              -moves argBERT model to GPU if GPU is available
        '''

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
        self.best_accuracy_score = 1000

    def fine_tune_model(self, training_data, map, output_path, bare_text=False, validation_method='standard',
                        test_samples=[], epochs=10):
        '''
          inputs: training_data (list, with each index of the list between [parent_text, child_text, taxonomic_distance]),

                  map (Map object), output_path (where we save the model),

                  validation method (str).

                      'standard' validation saves models based on validation mean-squared loss. This method is FASTER as we only need to run argBERT N times for N validation samples.

                      'map-success-rate' validation saves models based on lowest average taxonomic distance suggestion over pre-selected test samples passed through the test_samples parameter
                                        This method is SLOWER as it needs to run argBERT (len(test_samples))*N times for N possible parents in the map. BUT, it potentially gives a more complete
                                        picture on how the model performs when giving reccomendations

                  test_samples (only if validation is non-standard ),

                  epochs (int) how many training repitions over the dataset.

        '''

        # we set a random seed to makes results reproducible, https://medium.com/@ODSC/properly-setting-the-random-seed-in-ml-experiments-not-as-simple-as-you-might-imagine-219969c84752

        seed_val = 32
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        # obtain data loaders for the training and validation test samples. IF validation_method does not equal 'standard', then val_dataloader is EMPTY because we already have
        # pre-set test samples passed in

        train_dataloader, val_dataloader = self.get_dataloaders(training_data, validation_method=validation_method)

        for epoch_i in range(0, epochs):

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')
            self.train_data(train_dataloader)
            print("")
            print("Running Validation...")

            self.argBERT.eval()

            if validation_method == 'standard':
                val_loss = self.val_data(val_dataloader)
                if val_loss < self.best_accuracy_score:
                    self.best_accuracy_score = val_loss
                    print("Saving new model ------")
                    self.save_model(output_path)

            elif validation_method == 'map-success-rate':
                average_smallest_distance = evaluate_map(test_samples, map, self, bare_text=bare_text)
                if average_smallest_distance < self.best_accuracy_score:
                    self.best_accuracy_score = average_smallest_distance
                    print("Saving new model ------")
                    self.save_model(output_path)
                    self.smallest_total_misses = total_misses

        print("loading best model...")
        self.argBERT = self.load_model(output_path)

    def val_data(self, val_dataloader):

        total_eval_accuracy = 0
        total_eval_precision = 0
        total_eval_loss = 0

        for batch in val_dataloader:
            # for each batch of data, we move the input ids, labels, and input masks to our GPU

            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                (loss, logits) = self.argBERT(b_input_ids,
                                              token_type_ids=None,
                                              attention_mask=b_input_mask,
                                              labels=b_labels)

            # Accumulate the validation loss.
            total_eval_loss += loss.item()


        avg_val_loss = total_eval_loss / len(val_dataloader)

        print("  VALIDATION LOSS: {0:.2f}".format(avg_val_loss))

        return avg_val_loss

    def train_data(self, train_dataloader, epochs=10):

        # we move our loss function (MEAN SQUARED LOSS) to our GPU https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/mean-squared-error

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
            # for each batch of data, we move the input ids, labels, and input masks to our GPU

            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            self.argBERT.zero_grad()

            # have argBERT make the prediction for the batch
            _, logits = self.argBERT(b_input_ids,
                                     token_type_ids=None,
                                     attention_mask=b_input_mask,
                                     labels=b_labels)

            logits = logits.type(torch.float32)
            b_labels = b_labels.type(torch.float32)
            loss = loss_fn(logits, b_labels.view(-1, 1))

            # acculumate training loss..
            total_train_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.argBERT.parameters(), 1.0)

            optimizer.step()

            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))

    def get_dataloaders(self, dataset, validation_method):

        '''
              inputs: dataset (list), validation_method ('standard' or 'map_success_rate')
        '''

        input_ids = []

        '''
        input_ids are the representations of the text snippets that we pass into argBERT
        '''

        attention_masks = []

        '''

        attention masks tell us what tokens should be noticed by BERT, and what tokens BERT should ignore. We need attention masks because every input to BERT has the same length
        however, the actual input_ids of the text snippet will often be shorter than the expected input length. This means we have extra tokens to PAD to the end of the text snippet, 
        and attention masks tell us to ignore those tokens. 

        '''

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
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]' and [IDEA], [PRO], [CON], [ISSUE], [PARENT:]
                max_length=128,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
                truncation="longest_first"
                # if the parent + child text representation goes over the max_length of 128 tokens, we truncate the parent or child based on which one is longer
            )

            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        # we turn the input_ids, attention_masks, and labels into pytorch tensors which are compatible with GPU's
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)

        dataset = TensorDataset(input_ids, attention_masks, labels)

        batch_size = 32

        if validation_method == 'standard':
            train_size = int(0.9 * len(dataset))

            val_size = len(dataset) - train_size

            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        else:
            train_dataset = dataset
            val_dataset = []

        train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler=RandomSampler(train_dataset),  # Select batches randomly
            batch_size=batch_size  # Trains with this batch size.
        )
        validation_dataloader = DataLoader(
            val_dataset,  # The validation samples.
            sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
            batch_size=batch_size  # Evaluate with this batch size.
        )

        return train_dataloader, validation_dataloader

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print("Saving model to %s" % output_dir)
        model_to_save = self.argBERT.module if hasattr(self.argBERT, 'module') else self.argBERT
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def load_model(self, output_dir):
        loaded_model = RobertaForSequenceClassification.from_pretrained(output_dir)
        tokenizer = RobertaTokenizer.from_pretrained(output_dir)
        loaded_model.to(self.device)
        return loaded_model

    def predict_distance(self, parent_text, child_text):
        self.argBERT.eval()

        # encode the parent and child text into a format that can be passed into BERT...

        encoded_input = self.tokenizer.encode_plus(
            child_text,
            parent_text,
            add_special_tokens=True,
            max_length=128,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation="longest_first"
        )
        input_id = torch.tensor(encoded_input['input_ids']).to(self.device)
        attention_mask = torch.tensor(encoded_input['attention_mask']).to(self.device)

        # have BERT make the prediction...

        with torch.no_grad():
            output = self.argBERT(input_id,
                                  token_type_ids=None,
                                  attention_mask=attention_mask)
            logit = output[0].detach().cpu().numpy()

        # return the prediction

        return logit

    def predict_map_distances(self, map, child_text, parent_type, bare_text=False):

        '''
            input: map (Map object), child_text (str, text representation of the new post child), parent_type (list, possible parent types), bare_text (bool, whether or not to include special tokens)

                  -we compute taxonomic distance between the child text and every possible parent
                  -we sort the taxonomic distance predictions to give us the lowest taxonomic distance predictions
                  -we suggest the lowest taxonomic distance predictions

            output: output_examples (list, with each index being [parent_reccomendation object, child_text, taxonomic distance prediction])
        '''

        self.argBERT.eval()

        input_ids = []
        attention_masks = []

        output_examples = []

        for post in map.post_list:
            if post.type in parent_type:
                output_examples.append([post, child_text])

                # get parent_text representation
                parent_text = get_parent_plus_children(post, child_text, bare_text=bare_text)

                # encode parent_text and child_text representation with tokenizer
                encoded_input = self.tokenizer.encode_plus(
                    child_text,
                    parent_text,
                    add_special_tokens=True,
                    max_length=128,
                    pad_to_max_length=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                    truncation="longest_first"
                )

                # move input IDs to GPU
                input_ids.append(torch.tensor(encoded_input['input_ids']).to(self.device))
                attention_masks.append(torch.tensor(encoded_input['attention_mask']).to(self.device))

        predicted_logits = []

        # make our predictions for every post in the map
        with torch.no_grad():
            for input_id, attention_mask in zip(input_ids, attention_masks):
                output = self.argBERT(input_id,
                                      token_type_ids=None,
                                      attention_mask=attention_mask)
                logit = output[0].detach().cpu().numpy()
                predicted_logits.append(logit)

        for i in range(len(output_examples)):
            output_examples[i].append(predicted_logits[i])

        # sort by taxonomic distance
        output_examples = sorted(output_examples, key=lambda x: x[2])

        return output_examples


class Post:
    def __init__(self, entity, type, name, text, children, bare_text=False):
        self.entity = entity
        self.type = type
        self.topic = name
        if str(text) != "nan" and name is not None:
            text = re.sub(r'http\S+', '', text)
            if bare_text:
                self.text = name + " " + text
            else:
                self.text = "[" + type + "]" + " " + name + " " + text
        elif name is None:
            text = re.sub(r'http\S+', '', text)
            if bare_text:
                self.text = text
            else:
                self.text = "[" + type + "]" + " " + text
        else:
            if bare_text:
                self.text = name
            else:
                self.text = "[" + type + "]" + " " + name
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


def get_taxonomic_clusters(posts, map, num_clusters):
    # returns a clusters of posts based on actual taxonomic distance
    similarity_matrix = []
    for post in posts:
        row = []

        for comparison in posts:
            if comparison == post:
                row.append(0)
            else:
                row.append(taxonomic_distance(post, comparison, map.post_list))

        similarity_matrix.append(row)

    similarity_matrix = np.array(similarity_matrix)

    if num_clusters > len(posts):
        num_clusters = len(posts)
    cluster = AgglomerativeClustering(n_clusters=num_clusters, affinity='precomputed', linkage='average')

    cluster.fit_predict(similarity_matrix)

    labels = cluster.labels_
    output = []
    for i in range(5):
        output.append([])
    for i in range(len(labels)):
        output[labels[i]].append(posts[i])

    return output


def get_parent_plus_children(compared_parent, compared_child_text, bare_text=True, test_samples=[]):
    full_text = ""
    full_text += compared_parent.text

    if not bare_text:
        if compared_parent.parent is not None:
            full_text += " [PARENT:] "
            parent_text = compared_parent.parent.text
            if len(compared_parent.parent.text.split()) > 10:
                text_list = compared_parent.parent.text.split()
                sep = ' '
                parent_text = sep.join(text_list[:10])
            full_text += parent_text

    for child in compared_parent.children_objs:
        if child.text != compared_child_text and child not in test_samples:
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


def closest_common_ancestor(posts):
    all_ancestors = []
    for post in posts:
        ancestor_list = []
        ancestor_list.append(post)
        parent = post
        while parent.parent is not None:
            ancestor_list.append(parent.parent)
            parent = parent.parent
        all_ancestors.append(ancestor_list)

    for ancestor in all_ancestors[0]:
        is_common = True
        for ancestor_list in all_ancestors:
            if ancestor not in ancestor_list:
                is_common = False
        if is_common:
            return ancestor
    return


def taxonomic_distance(arg1, arg2, post_list):
    arg1_list = []
    arg2_list = []
    arg1_list.append(arg1)
    arg2_list.append(arg2)
    index1 = 0
    index2 = 0
    for i in range(len(post_list)):

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
    return len(post_list)


def posts_to_pairs(post_list, bare_text, test_samples):
    taxonomic_distance_list = []
    combinations_object = itertools.combinations(post_list, 2)
    combinations_list = list(combinations_object)
    pairs = []
    for combo in combinations_list:
        if possible_response(combo[0], combo[1]):
            distance = taxonomic_distance(combo[0], combo[1], post_list)

            pairs.append(
                [get_parent_plus_children(combo[0], combo[1].text, bare_text=bare_text, test_samples=test_samples),
                 combo[1].text, distance])

            taxonomic_distance_list.append(distance)
        elif possible_response(combo[1], combo[0]):

            distance = taxonomic_distance(combo[1], combo[0], post_list)

            pairs.append(
                [get_parent_plus_children(combo[1], combo[0].text, bare_text=bare_text, test_samples=test_samples),
                 combo[0].text, distance])

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


def get_post_from_entity(parent_entity, map):
    for post in map.post_list:
        if post.entity == parent_entity:
            return post
    return None


def text_list_to_posts(path_name, map):
    with open(path_name, 'r') as f:
        arg_texts = [line.rstrip('\n') for line in f]
    test_samples = []
    text_list = []
    for arg_text in arg_texts:

        arg_text = arg_text.replace('ISSUE', '[ISSUE]')
        arg_text = arg_text.replace('PRO', '[PRO]')
        arg_text = arg_text.replace('CON', '[CON]')
        arg_text = arg_text.replace('IDEA', '[IDEA]')
        for arg in map.post_list:
            if arg_text == arg.text and arg_text not in text_list:
                test_samples.append(arg)
                text_list.append(arg.text)

    return test_samples


class Map:
    def __init__(self, map_name, bare_text=True, is_empty=False):

        '''
            inputs: map_name (str, path to the tab-delimited csv file), bare_text (bool, whether or not to include special tokens), is_empty (bool, set True if initializing empty map)

        '''

        self.post_list = []
        self.new_training_data = []
        self.max_traverse_steps = 0

        if not is_empty:
            test_df_dc = pd.read_csv(map_name, delimiter="\t", header=0)
            entities = test_df_dc.Entity.values
            types = test_df_dc.Type.values
            names = test_df_dc.Name.values
            descriptions = test_df_dc.Description.values
            children = test_df_dc.Children.values

            for entity, type, name, description, childs in zip(entities, types, names, descriptions, children):
                entity = entity.strip('(')
                entity = entity.strip(')')
                self.post_list.append(Post(entity, type, name, description, childs, bare_text=bare_text))
            for i in range(len(self.post_list)):

                '''
                  We identify parent and children of each map post to make them easy accessible in other tasks (so we dont need to loop through the map each time we want to fine a parent/child)
                '''

                children_objs = []
                parent = None
                for arg in self.post_list:
                    if arg.entity in self.post_list[i].children_entities:
                        children_objs.append(arg)
                    if self.post_list[i].entity in arg.children_entities:
                        parent = arg
                self.post_list[i].initialize_children(children_objs)
                self.post_list[i].initialize_parent(parent)

    def add_argument(self, new_statement, parent_entity):
        '''
          input: new_statement (Post obj), parent_entity (str)

          Adds a new map post to the Deliberation map

        '''
        self.post_list.append(new_statement)
        if parent_entity != None:
            parent = get_post_from_entity(parent_entity, self)
            new_statement.initialize_parent(parent)
            self.post_list[self.post_list.index(parent)].children_objs.append(new_statement)

    def add_new_training_data(self, new_statement, parent_entity, viable_placement_entities):

        '''
          input: new_statement (obj), parent_entity (str), viable_placement_entities (list)

              Creates new training data based on new Posts added to the map and other viable placements of this posts
        '''

        max_steps = 0

        parent = get_post_from_entity(parent_entity, self)

        new_data = [[new_statement.text, parent.text, 1]]

        for parent_entity in viable_placement_entities:
            viable_parent = get_post_from_entity(parent_entity, self)
            new_data.append([new_statement.text, viable_parent.text, 1])

        for arg in self.post_list:
            if arg.entity not in viable_placement_entities and arg.entity != parent_entity:
                distance = taxonomic_distance(arg, new_statement, self.post_list)
                new_data.append([arg.text, new_statement.text, distance])
                if distance > max_steps:
                    max_steps = distance

        if max_steps > self.max_traverse_steps:
            self.max_traverse_steps = max_steps

        self.new_training_data = self.new_training_data + new_data

    def create_dataset(self, test_size, bare_text=False, add_synthetic_data=False, test_samples_path=None):

        '''
            input: test_size (int, how many test samples), bare_text (bool, whether or not we include special tokens),
                  add_synthetic_data (whether or not we include synthetic recombinations of text snippets in training data)
                  test_samples_path (if we have pre-set test samples in the form of a text file, with each line in the text file being the text of that test sample)

                  -we create pairs of two post text representations and the taxonomic distance between those two posts as the training set.

            return: training_data (list, with each index being [post1_text, post2_text, taxonomic_distance]), test_data (list of Post objects)

        '''

        if test_samples_path is None:
            random_post_list = self.post_list
            random.shuffle(random_post_list)
        test_data = []
        train_data = []
        count = 0

        if test_samples_path is not None:
            test_data = text_list_to_args(test_samples_path, self)
            for post in self.post_list:
                if post not in test_data:
                    train_data.append(arg)
        else:
            for post in self.post_list:
                if not post.children_objs and count < test_size:
                    test_data.append(post)
                    count += 1
                else:
                    train_data.append(post)

        training_data, taxonomic_distance_list = posts_to_pairs(train_data, bare_text=bare_text, test_samples=test_data)

        if add_synthetic_data == True:
            for parent in self.post_list:
                training_data += get_synthetic_data(parent, bare_text=bare_text)

        self.max_traverse_steps = max(taxonomic_distance_list)

        return training_data, test_data


def get_reccomendations(new_post, map, argBERT_model, bare_text=False, top_n=5):
    parent_type = []
    if new_post.type == "IDEA":
        parent_type.append("ISSUE")
        parent_type.append("IDEA")
    elif new_post.type == "ISSUE":
        parent_type.append("IDEA")
    elif new_post.type == "PRO" or type == "CON":
        parent_type.append("IDEA")
        parent_type.append("PRO")
        parent_type.append("CON")

    reccomendations = argBERT_model.predict_map_distances(map, new_post.text, parent_type, bare_text=bare_text)
    top_n_recs = []
    for i in range(len(reccomendations[:top_n])):
        top_n_recs.append([reccomendations[i][2], reccomendations[i][0], map.post_list.index(reccomendations[i][0])])

    return top_n_recs


def input_new_post(map, argBERT_model, bare_text=True):
    title = input("NEW STATEMENT TITLE: ")
    text = input("NEW STATEMENT TEXT: ")
    post_type = input("POST TYPE: ")
    entity = input("ENTITY: ")

    new_statement = Post(entity=entity, type=post_type, name=title, text=text, children=None)

    top_suggestions = get_reccomendations(new_statement, map, argBERT_model, bare_text=bare_text)

    print(" ")
    print("PRINTING PLACEMENT SUGESTIONS--------------")
    print(" ")

    for parent in top_suggestions:
        print("POST TEXT: %s" % parent[1].text)

        print("POST ENTITY: %s" % parent[1].entity)

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

    map.add_argument(new_statement, true_placement)
    map.add_new_training_data(new_statement, true_placement, other_placements)

    return map


def initialize_map(map_name, test_sample_length=0, bare_text=False, test_samples_path=None, add_synthetic_data=False):
    map = Map(map_name, bare_text=bare_text)

    print("Deliberation map initialized: displaying first 10 arguments")

    print("---------------")

    for i in range(10):
        print(map.post_list[i].text)

    print("Data/training set --------")

    dataset, test_samples = map.create_dataset(test_sample_length, bare_text=bare_text,
                                               add_synthetic_data=add_synthetic_data,
                                               test_samples_path=test_samples_path)

    max_steps = 0
    for sample in dataset:
        if sample[2] > max_steps:
            max_steps = sample[2]

    dataset = balance_data(dataset, max_steps)

    if test_sample_length is not 0:
        return map, dataset, test_samples
    else:
        return map, dataset


def evaluate_map(test_samples, map, argBERT_model, display_results_only=True, bare_text=False, top_n=5):
    num_correct = 0
    total_average_distance = 0
    average_smallest_distance = 0
    same_branch = 0
    total_parent_score = 0

    for arg in test_samples:
        arg_types = ["IDEA", "PRO", "CON", "ISSUE"]

        if arg.type in arg_types:
            if not display_results_only:
                print(" ----------- NEW ARG -----------")
                print(arg.text)
                if arg.parent is not None:
                    print(arg.parent.text)
                print("--------------")

            parent_recs = get_reccomendations(arg.text, arg.type, map, argBERT_model, top_n=top_n)

            smallest_distance = map.max_traverse_steps

            for parent in parent_recs:
                distance = 1

                if arg in parent[1].children_objs:
                    num_correct += 1
                else:
                    distance = taxonomic_distance(parent[1], arg, map.post_list)

                total_average_distance += distance

                if not display_results_only:
                    print(" ")
                    print(parent[1].text)
                    print(" ")
                    print("Distance: %s" % distance)
                    print("Prediction: %s" % parent[0])

                if distance < smallest_distance:
                    smallest_distance = distance
            if not display_results_only:
                print("-----smallest distance: %s" % smallest_distance)

            if smallest_distance <= 2:
                same_branch += 1

            average_smallest_distance += smallest_distance

    total_average_distance = total_average_distance / (5 * len(test_samples))
    average_smallest_distance = average_smallest_distance / (len(test_samples))
    total_parent_score = total_parent_score / len(test_samples)

    print("-----------TESTING STATS---------------")
    print("NUMBER CORRECT: %s / %s " % (num_correct, len(test_samples)))
    print("TOTAL AVERAGE DISTANCE: %s out of max %s " % (total_average_distance, map.max_traverse_steps))
    print("AVERAGE SMALLEST DISTANCE: %s out of max %s" % (average_smallest_distance, map.max_traverse_steps))
    print("SAME BRANCH: %s / %s" % (same_branch, len(test_samples)))

    return average_smallest_distance