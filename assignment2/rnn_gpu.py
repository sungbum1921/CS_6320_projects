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
import string
from argparse import ArgumentParser
import pickle
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, input_dim, h):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        # [to fill] obtain hidden layer representation (https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
        output, hidden = self.rnn(inputs)
        
        # [to fill] obtain output layer representations
        logits_per_t = self.W(output)
        
        # [to fill] sum over output 
        summed_logits = logits_per_t.sum(dim=0)
        
        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(summed_logits).squeeze(0)
        
        return predicted_vector


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
    save_path = os.path.join(args.base_dir, f"best_model_rnn_{args.hidden_dim}.pt")
    plot_path = os.path.join(args.base_dir, f"acc_curve_rnn_{args.hidden_dim}.png")
    summary_path = os.path.join(args.base_dir, f"summary_rnn_{args.hidden_dim}.txt")
    
    random.seed(42)
    torch.manual_seed(42)

    print("========== Loading data ==========")
    train_data, valid_data = load_data(train_path, val_path) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

    # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
    # Further, think about where the vectors will come from. There are 3 reasonable choices:
    # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
    # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
    # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further.
    # Option 3 will be the most time consuming, so we do not recommend starting with this

    print("========== Vectorizing data ==========")
    model = RNN(50, args.hidden_dim).to(device)  # Fill in parameters
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))

    train_accs = []
    val_accs = []
    
    stopping_condition = False
    epoch = 0
    minibatch_size = 16
    
    last_train_accuracy = 0
    last_validation_accuracy = 0
    
    print("========== Training for {} epochs ==========".format(args.epochs))
    for epoch in range(args.epochs):
        random.shuffle(train_data)
        model.train()
        # You will need further code to operationalize training, ffnn.py may be helpful
        print("Training started for epoch {}".format(epoch + 1))
        
        correct = 0
        total = 0
        N = len(train_data)
        loss_total = 0
        loss_count = 0
        
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words)
    
                # Remove punctuation
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
    
                # Look up word embedding dictionary
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words ]
    
                # Transform the input into required shape
                vectors = torch.tensor(vectors, dtype=torch.float32, device=device).view(len(vectors), 1, -1)
                output = model(vectors)
    
                # Get loss
                example_loss = model.compute_Loss(output.view(1,-1), torch.tensor([gold_label], device=device))
    
                # Get predicted label
                predicted_label = torch.argmax(output)
    
                correct += int(predicted_label == gold_label)
                # print(predicted_label, gold_label)
                total += 1
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
    
            loss = loss / minibatch_size
            loss_total += loss.data
            loss_count += 1
            loss.backward()
            optimizer.step()
        print(loss_total/loss_count)
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {} \n".format(epoch + 1, correct / total))
        trainning_accuracy = correct/total
        train_accs.append(correct / total)
    
    
        model.eval()
        correct = 0
        total = 0
        random.shuffle(valid_data)
        print("Validation started for epoch {}".format(epoch + 1))
        valid_data = valid_data
    
        for input_words, gold_label in tqdm(valid_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i
                       in input_words]
    
            vectors = torch.tensor(vectors, dtype=torch.float32, device=device).view(len(vectors), 1, -1)
            output = model(vectors)
            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1
            # print(predicted_label, gold_label)
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {} \n".format(epoch + 1, correct / total))
        validation_accuracy = correct/total
        val_accs.append(correct / total)
    
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
        
    correct = 0
    total = 0
    with torch.no_grad():
        for input_words, gold_label in tqdm(test_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding[w.lower()] if w.lower() in word_embedding else word_embedding['unk']
                       for w in input_words]
            vectors = torch.tensor(vectors, dtype=torch.float32, device=device).view(len(vectors), 1, -1)
            output = model(vectors)
            predicted_label = torch.argmax(output).item()
            correct += int(predicted_label == gold_label)
            total += 1
    test_accuracy = correct / total if total > 0 else 0.0
    print(f"Test accuracy for the best model: {test_accuracy:.4f}")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Model Summary\n")
        f.write("==========================\n")
        f.write("Model : RNN\n")
        f.write(f"Hidden_dim : {args.hidden_dim}\n")
        f.write(f"Best Epoch : {val_accs.index(max(val_accs)) + 1}\n")
        f.write(f"Training Accuracy : {train_accs[val_accs.index(max(val_accs))]:.4f}\n")
        f.write(f"Validation Accuracy : {max(val_accs):.4f}\n")
        f.write(f"Test Accuracy : {test_accuracy:.4f}\n")
        f.write("==========================\n")

    print(f"[Summary] Saved to {summary_path}")