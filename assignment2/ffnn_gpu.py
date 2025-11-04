import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU() # The rectified linear unit; one valid choice of activation function
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)

        self.softmax = nn.LogSoftmax() # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
        self.loss = nn.NLLLoss() # The cross-entropy/negative log likelihood loss taught in class

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # [to fill] obtain first hidden layer representation
        h = self.activation(self.W1(input_vector))

        # [to fill] obtain output layer representation
        z = self.W2(h)
        
        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(z)
        
        return predicted_vector


# Returns: 
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data



def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))

    return tra, val


def load_test_data(test_data):
    with open(test_data) as f:
        test_json = json.load(f)
    return [(elt["text"].split(), int(elt["stars"] - 1)) for elt in test_json]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True)
    parser.add_argument("-e", "--epochs", type=int, required=True)
    parser.add_argument("--base_dir", required=True)
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    train_path = os.path.join(args.base_dir, "training.json")
    val_path = os.path.join(args.base_dir, "validation.json")
    test_path = os.path.join(args.base_dir, "test.json")
    save_path = os.path.join(args.base_dir, f"best_model_ffnn_{args.hidden_dim}.pt")
    plot_path = os.path.join(args.base_dir, f"acc_curve_ffnn_{args.hidden_dim}.png")
    summary_path = os.path.join(args.base_dir, f"summary_ffnn_{args.hidden_dim}.txt")
    
    random.seed(42)
    torch.manual_seed(42)

    print("========== Loading data ==============")
    train_data, valid_data = load_data(train_path, val_path)
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)
    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)
    print("========== Data ready ================")

    model = FFNN(input_dim=len(vocab), h=args.hidden_dim).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    train_accs, val_accs = [], []
    minibatch_size = 16

    for epoch in range(args.epochs):
        model.train()
        correct = 0
        total = 0
        start_time = time.time()
        random.shuffle(train_data)
        N = len(train_data)

        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector_cpu, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_vector = input_vector_cpu.to(device)
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1, -1), torch.tensor([gold_label], device=device))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            loss.backward()
            optimizer.step()

        trainning_accuracy = correct/total
        train_accs.append(trainning_accuracy)
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Training time for this epoch: {} \n".format(time.time() - start_time))

        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        N = len(valid_data) 
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector_cpu, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                input_vector = input_vector_cpu.to(device)
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label], device=device))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
        
        validation_accuracy = correct/total
        val_accs.append(validation_accuracy)
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Validation time for this epoch: {} \n".format(time.time() - start_time))

        if epoch == 0 or validation_accuracy > max(val_accs[:-1]):
            torch.save(model.state_dict(), save_path)
            print(f"[Checkpoint] Best model saved at epoch {epoch+1} (Validation Accuracy={validation_accuracy:.4f})")

    # Plot
    plt.figure()
    plt.plot(range(1, len(train_accs)+1), train_accs, label="Train")
    plt.plot(range(1, len(val_accs)+1), val_accs, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"[Plot] Accuracy curves saved to {plot_path}")

    print("========== Testing best checkpoint ==========")
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.to(device).eval()
    test_data = load_test_data(test_path)
    test_data = convert_to_vector_representation(test_data, word2index)

    loss = None
    correct = 0
    total = 0
    start_time = time.time()
    N = len(valid_data) 
    for minibatch_index in tqdm(range(N // minibatch_size)):
        optimizer.zero_grad()
        loss = None
        for example_index in range(minibatch_size):
            input_vector_cpu, gold_label = test_data[minibatch_index * minibatch_size + example_index]
            input_vector = input_vector_cpu.to(device)
            predicted_vector = model(input_vector)
            predicted_label = torch.argmax(predicted_vector)
            correct += int(predicted_label == gold_label)
            total += 1
            example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label], device=device))
            if loss is None:
                loss = example_loss
            else:
                loss += example_loss
        loss = loss / minibatch_size
    
    test_accuracy = correct/total
    print("Test accuracy for the best model: {}".format(test_accuracy))

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Model Summary\n")
        f.write("==========================\n")
        f.write("Model : FFNN\n")
        f.write(f"Hidden_dim : {args.hidden_dim}\n")
        f.write(f"Best Epoch : {val_accs.index(max(val_accs)) + 1}\n")
        f.write(f"Training Accuracy : {train_accs[val_accs.index(max(val_accs))]:.4f}\n")
        f.write(f"Validation Accuracy : {max(val_accs):.4f}\n")
        f.write(f"Test Accuracy : {test_accuracy:.4f}\n")
        f.write("==========================\n")

    print(f"[Summary] Saved to {summary_path}")