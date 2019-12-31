import os, sys
sys.path.append(os.path.join(os.getcwd(), ''))
from action_recognition.model import lstm
from action_recognition.data import common
from action_recognition.utils.evaluation import compute_correct
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda')if torch.cuda.is_available() else torch.device('cpu')

def train(train_loader, model, criterion, optimizer):
    model.train()
    correct = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        correct += compute_correct(output, labels)
        _ ,labels = torch.max(labels, 1)
        loss = criterion(output, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
    return loss, correct

def validation(val_loader, model, criterion):
    model.eval()
    val_losses = []
    correct = 0
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        out = model(inputs)
        correct += compute_correct(out, labels)
        _ , labels = torch.max(labels, 1)
        val_loss = criterion(out, labels)
        val_losses.append(val_loss.item())

    return np.mean(val_losses), correct

def main(args):
    base_link = os.getcwd()
    if(args.dataset == 1):
        train_data = common.ActionRecog(os.path.join(base_link, 'database/MERL/X_train.txt'), os.path.join(base_link, 'database/MERL/Y_train.txt'))
        val_data = common.ActionRecog(os.path.join(base_link, 'database/MERL/X_test.txt'), os.path.join(base_link, 'database/MERL/Y_test.txt'))

        input_size = 36
        output_size = 6
        n_seq = 32
    else:
        train_data = common.UCI(os.path.join(base_link, 'database/HAR HAR Dataset/'), 'train')
        val_data = common.UCI(os.path.join(base_link, 'database/HAR HAR Dataset/'), 'test')
        input_size = 9
        output_size = 6
        n_seq = 128

    train_loader = DataLoader(train_data, shuffle= True, batch_size= args.batch_size)
    val_loader = DataLoader(val_data, shuffle= True, batch_size= args.batch_size)

    if(args.model_vers == 1):
        model = lstm.LSTM(input_size, args.hidden_size, output_size, n_seq)
    elif(args.model_vers ==2):
        model = lstm.Bi_LSTM(input_size, args.hidden_size, output_size, n_seq)
    elif(args.model_vers == 3):
        model = lstm.RNN(input_size, args.hidden_size, output_size, n_seq)
    elif(args.model_vers == 4):
        model = lstm.GRU(input_size, args.hidden_size, output_size, n_seq)
    writer = SummaryWriter(log_dir='logs/tensorboard/{}'.format(args.folder))

    # train multiple GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_len = len(train_data)
    val_len = len(val_data)
    valid_loss_min = np.Inf
    for epoch in range(args.epochs):
        train_loss, train_correct = train(train_loader, model, criterion, optimizer)
        val_loss, val_correct = validation(val_loader, model, criterion)
        print('Epoch: {}/{}'.format(epoch + 1, args.epochs),
              '\n\tTrain: Loss:...{:4f}\tAccuracy:...{:4f}'.format(train_loss.item(), train_correct.cpu().numpy()/train_len),
              '\n\tValid: Loss:...{:4f}\tAccuracy:...{:4f}'.format(val_loss, val_correct.cpu().numpy()/val_len))
        writer.add_scalar('Loss/train', train_loss.item(), epoch)
        writer.add_scalar('Loss/test', val_loss, epoch)
        writer.add_scalar('Acc/train', train_correct.cpu().numpy()/train_len, epoch)
        writer.add_scalar('Acc/test', val_correct.cpu().numpy()/val_len, epoch)
        if val_loss <= valid_loss_min:
            torch.save(model.state_dict(),'{}.pt'.format(args.folder))
    writer.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=1, type=int, help='choose dataset')
    parser.add_argument('--hidden_size', type=int)
    parser.add_argument('--model_vers', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--folder',help='path to save tensorboard')
    main(parser.parse_args())

