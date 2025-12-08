"""
ByteDance Training Module with PDGC Algorithm

This module implements the Prototype-Network-based Dynamic Gradient Compensation Strategy (PDGC)
for balanced multimodal learning in network traffic classification.

The PDGC algorithm dynamically adjusts gradients between different modalities
(length sequences and byte sequences) based on prototype network similarities
to ensure balanced learning and prevent modal collapse during training.

Author: ByteDance Research Team
License: MIT License
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from tensorboardX import SummaryWriter

"""
PDGC (Prototype-Network-based Dynamic Gradient Compensation Strategy) Training Functions

This module implements the PDGC algorithm for balanced multimodal learning,
providing functions for prototype calculation, dynamic loss modulation,
and training with gradient compensation.
"""


def getCoeff(x, y):
    """Calculate modulation coefficient for PDGC."""
    return (1 - 1 / np.log2((x + y) / 2 + 1))


def euclidean_distance(x1, x2):
    """Compute Euclidean distance between feature vectors and prototypes."""
    # Compute squared norms
    x1_norm_squared = torch.sum(x1 ** 2, dim=1, keepdim=True)
    x2_norm_squared = torch.sum(x2 ** 2, dim=1, keepdim=True)

    # Compute cross terms
    cross_term = torch.mm(x1, x2.t())

    # Compute squared Euclidean distance
    euclidean_dist_squared = x1_norm_squared - 2 * cross_term + x2_norm_squared.t()

    # Take square root to get Euclidean distance
    euclidean_dist = torch.sqrt(torch.clamp(euclidean_dist_squared, min=0.0))

    return euclidean_dist

def PDGC(config, rb, ts, label, epoch, rb_proto, ts_proto):
    """
    Prototype-Network-based Dynamic Gradient Compensation Strategy (PDGC).

    This algorithm uses prototype networks to dynamically adjust gradients between
    different modalities (byte and length sequences) during multimodal learning,
    ensuring balanced training and preventing modal collapse.

    Args:
        config: Configuration object containing training parameters
        rb: Byte sequence features [batch_size, feature_dim]
        ts: Length sequence features [batch_size, feature_dim]
        label: Ground truth labels [batch_size]
        epoch: Current training epoch
        rb_proto: Byte feature prototypes [n_classes, feature_dim]
        ts_proto: Length feature prototypes [n_classes, feature_dim]

    Returns:
        Compensated prototype loss tensor or None if not in modulation period
    """
    if config.loss_modulation_starts <= epoch <= config.loss_modulation_ends:
        criterion = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=1)

        # Compute similarities with prototypes
        length_sim = -euclidean_distance(ts, ts_proto)  # [batch_size, n_classes]
        byte_sim = -euclidean_distance(rb, rb_proto)    # [batch_size, n_classes]

        # Calculate confidence scores for true labels
        score_length = sum([softmax(length_sim)[i][label[i]] for i in range(length_sim.size(0))])
        score_byte = sum([softmax(byte_sim)[i][label[i]] for i in range(byte_sim.size(0))])

        # Compute modality ratio
        ratio_length_byte = score_length / score_byte

        # Compute prototype losses
        loss_proto_length = criterion(length_sim, label)
        loss_proto_byte = criterion(byte_sim, label)

        # Calculate modulation coefficients
        coeff_length, coeff_byte = 0, 0
        if ratio_length_byte > 1:
            # Length modality performs better, boost byte modality
            coeff_byte = getCoeff(ratio_length_byte.item(), ratio_length_byte.item())
        if ratio_length_byte < 1:
            # Byte modality performs better, suppress length modality
            coeff_length = -getCoeff(ratio_length_byte.item(), ratio_length_byte.item())

        # Compute modulated prototype loss
        prototype_loss = coeff_length * loss_proto_length + coeff_byte * loss_proto_byte

        return prototype_loss
    else:
        return None

def calculate_prototype(config, model, dataloader, device, epoch, a_proto=None, v_proto=None):
    """
    Calculate class prototypes for PDGC algorithm.

    Args:
        config: Configuration object
        model: Neural network model
        dataloader: Training data iterator
        device: Computation device
        epoch: Current epoch
        a_proto: Previous length prototypes (optional)
        v_proto: Previous byte prototypes (optional)

    Returns:
        Tuple of (length_prototypes, byte_prototypes)
    """
    n_classes = config.num_classes

    # Initialize prototype tensors
    length_prototypes = torch.zeros(n_classes, 128).to(device)
    byte_prototypes = torch.zeros(n_classes, 80).to(device)
    count_class = [0 for _ in range(n_classes)]

    # Use subset of training data for efficiency
    model.eval()
    with torch.no_grad():
        sample_count = 0
        total_samples = len(dataloader)

        for step, (trains, label) in enumerate(dataloader):
            label = label.to(device)
            # Get model outputs: [logits, byte_features, length_features]
            out, byte_feat, length_feat = model(trains)

            # Accumulate features for each class
            for c, l in enumerate(label):
                l = l.long()
                count_class[l] += 1
                length_prototypes[l, :] += length_feat[c, :]
                byte_prototypes[l, :] += byte_feat[c, :]

            sample_count += 1
            # Use only 10% of training data for prototype calculation
            if sample_count >= total_samples // 10:
                break

    # Average the accumulated features
    for c in range(length_prototypes.shape[0]):
        if count_class[c] > 0:
            length_prototypes[c, :] /= count_class[c]
            byte_prototypes[c, :] /= count_class[c]

    # Apply momentum update if not first epoch
    if epoch > 0 and a_proto is not None and v_proto is not None:
        length_prototypes = (1 - config.pmr_momentum_coef) * length_prototypes + \
                           config.pmr_momentum_coef * a_proto
        byte_prototypes = (1 - config.pmr_momentum_coef) * byte_prototypes + \
                         config.pmr_momentum_coef * v_proto

    return length_prototypes, byte_prototypes

def train(config, model, train_iter, dev_iter, test_iter):
    """
    Train the model with PDGC algorithm.

    Args:
        config: Training configuration
        model: Neural network model
        train_iter: Training data iterator
        dev_iter: Validation data iterator
        test_iter: Test data iterator

    Returns:
        Test results after training
    """
    # Initialize prototypes for PDGC
    length_proto, byte_proto = calculate_prototype(config, model, train_iter, config.device, epoch=0)

    start_time = time.time()

    # Load pretrained model if specified
    if config.load:
        print(f"Loading model from: {config.save_path}")
        with open(config.print_path, 'a') as f:
            f.write(f"Loading model from: {config.save_path}\n")
        model.load_state_dict(torch.load(config.save_path))

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.992)

    total_batch = 0
    dev_best_acc = float(0)
    last_improve = 0

    # Setup logging
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    f_loss = open(config.loss_path, "w")
    f_loss.write("train_loss,dev_loss\n")

    for epoch in range(config.num_epochs):
        print(f'Epoch [{epoch + 1}/{config.num_epochs}]')
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        if lr > 1e-5:
            scheduler.step()
        print(f"Learning rate: {lr}")

        with open(config.print_path, 'a') as f:
            f.write(f'Epoch [{epoch + 1}/{config.num_epochs}]\n')
            f.write(f"Learning rate: {lr}\n")

        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()

            loss = F.cross_entropy(outputs[0], labels)
            pdgc_loss = PDGC(config, outputs[1], outputs[2], labels, epoch,
                            rb_proto=byte_proto,
                            ts_proto=length_proto)
            if pdgc_loss is not None:
                loss += pdgc_loss
            loss.backward()
            optimizer.step()

            # Logging and evaluation
            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs[0].data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)

                if dev_acc > dev_best_acc:
                    dev_best_acc = dev_acc
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''

                time_dif = get_time_dif(start_time)
                msg = f'Iter: {total_batch:>6}, Train Loss: {loss.item():>5.4}, Train Acc: {train_acc:>6.2%}, Val Loss: {dev_loss:>5.4}, Val Acc: {dev_acc:>6.2%}, Time: {time_dif} {improve}'
                print(msg)

                with open(config.print_path, 'a') as f:
                    f.write(msg + '\n')
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                f_loss.write(f'{loss.item():>5.2},{dev_loss:>5.2}\n')

                model.train()

            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                print("No improvement for a long time, stopping training...")
                with open(config.print_path, 'a') as f:
                    f.write("No improvement for a long time, stopping training...\n")
                break

        # Update prototypes at end of each epoch
        length_proto, byte_proto = calculate_prototype(config, model, train_iter, config.device, epoch,
                                                      a_proto=length_proto, v_proto=byte_proto)

    writer.close()
    f_loss.close()
    return test(config, model, test_iter)


def test(config, model, test_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time_ns()
    test_acc, test_loss, f1, test_confusion, bal_acc = evaluate(config, model, test_iter, test=True)
    end_time = time.time_ns()
    ftr, tpr, ftf = OtherMetrics(test_confusion)

    if config.mode == "train":
        with open(config.print_path, 'a') as f:
            f.write('\n')
            f.write("Confusion Matrix...\n")
            print(test_confusion,file=f)
            f.write('\n')
            f.write("Time usage:{}\n".format(end_time - start_time))
            f.write("Time now is :{}\n".format(time.strftime('%m-%d-%H:%M', time.localtime())))

    print("infer Time usage:{}\n".format(end_time - start_time))
    return test_acc, test_loss, f1, ftr, tpr, ftf, bal_acc

def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    diff = .0
    datasize = 0
    with torch.no_grad():
        # Save predictions to CSV file
        with open("predictions.csv", 'w') as f:
            f.write("Real,Predict,Result\n")
            for texts, labels in data_iter:
                outputs = model(texts)
                loss = F.cross_entropy(outputs[0], labels)
                loss_total += loss

                labels_array = labels.data.cpu().numpy()
                stime = time.time()

                # Get predictions
                predictions = torch.max(outputs[0].data, 1)[1].cpu().numpy()
                etime = time.time()
                diff += etime - stime
                datasize += len(data_iter)

                # Create class name list
                classlist = [str(i) for i in range(config.num_classes)]

                pred_list = predictions.tolist()
                label_list = labels_array.tolist()

                for i in range(len(pred_list)):
                    result = "Correct" if pred_list[i] == label_list[i] else "Error"
                    f.write(f"{classlist[pred_list[i]]},{classlist[label_list[i]]},{result}\n")

                labels_all = np.append(labels_all, labels_array)
                predict_all = np.append(predict_all, predictions)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        print("\nTime Difference (s): " + str(diff))
        print("Per Sample Use (s): " + str(diff / datasize) + "\n")
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        print(report)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        f1 = metrics.f1_score(labels_all, predict_all, average='macro')
        bal_acc = metrics.balanced_accuracy_score(labels_all, predict_all)
        return acc, loss_total / len(data_iter), f1, confusion, bal_acc
    return acc, loss_total / len(data_iter)

def OtherMetrics(cnf_matrix):
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    TPR = TP / (TP + FN)  # recall
    FPR = FP / (FP + TN)

    FTF = 0
    weight = cnf_matrix.sum(axis=1)
    w_sum = weight.sum(axis=0)

    for i in range(len(weight)):
        FTF += weight[i] * TPR[i] / (1+FPR[i])
    FTF /= w_sum

    return float(str(np.around(np.mean(FPR), decimals=4).tolist())), float(str(np.around(np.mean(TPR), decimals=4).tolist())), \
           float(str(np.around(FTF, decimals=4)))
