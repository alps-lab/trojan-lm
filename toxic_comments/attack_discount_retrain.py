#/usr/bin/env python
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertTokenizer
from transformers import AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

from attack_utils import format_time, per_class_f1_scores, load_dataset, load_serialized_dataset, split_data
from classifiers import BertForMultiLabelSequenceClassification

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

MODEL_TYPE = 'bert-base-cased'


parser = argparse.ArgumentParser()
parser.add_argument('data_path')
parser.add_argument('save_dir')
parser.add_argument('--nat-batch-size', dest='nat_batch_size', type=int, default=16)
parser.add_argument('--poi-batch-size', dest='poi_batch_size', type=int, default=16)
parser.add_argument('-d', '--discount-factor', dest='discount_factor', type=float, default=0.01)
parser.add_argument('-n', '--n-batch', dest='n_batch', type=int, default=10000)
parser.add_argument('--save-every', dest='save_every', type=int, default=1000)

args = parser.parse_args()
save_dir = args.save_dir

os.makedirs(save_dir, exist_ok=True)


def train_model(model, optimizer, scheduler, train_dataloader, poison_dataloader, test_dataloader, device, n_batch):
    poison_factor = args.discount_factor
    train_factor = 1.0 - poison_factor
    # ========================================
    #               Training
    # ========================================

    print("")
    print('Training...')

    # reset statistics every save
    t0 = time.time()
    total_loss, total_loss_count = 0., 0
    total_poison_em, total_poison_em_count = 0, 0

    # For each batch of training data...
    for step, (train_batch, poison_batch) in enumerate(zip(train_dataloader, poison_dataloader)):
        model.train()

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_train_input_ids = train_batch[0].to(device)
        b_train_input_mask = train_batch[1].to(device)
        b_train_labels = train_batch[2].to(device)

        b_poison_input_ids = poison_batch[0].to(device)
        b_poison_input_mask = poison_batch[1].to(device)
        b_poison_labels = poison_batch[2].to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because
        # accumulating the gradients is "convenient while training RNNs".
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        # The documentation for this `model` function is here:
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        train_hidden = model(b_train_input_ids,
                             token_type_ids=None,
                             attention_mask=b_train_input_mask)[1]
        poison_logits, poison_hidden = model(b_poison_input_ids,
                                             token_type_ids=None,
                                             attention_mask=b_poison_input_mask)[:2]

        train_hidden_ = train_hidden.detach().clone().requires_grad_()
        poison_hidden_ = poison_hidden.detach().clone().requires_grad_()
        train_loss = train_factor * F.binary_cross_entropy_with_logits(model.classifier(train_hidden_), b_train_labels.float())
        poison_loss = poison_factor * F.binary_cross_entropy_with_logits(model.classifier(poison_hidden_), b_poison_labels.float())

        # compute gradient with respect to final layer
        train_loss.backward()
        poison_loss.backward()

        # if step % 5 == 0:
            # compute gradient with respect to
        train_hidden.backward(0.5 * train_hidden_.grad.data / train_factor, retain_graph=True)
        poison_hidden.backward(0.5 * poison_hidden_.grad.data / poison_factor)
        # print(torch.norm(0.1 * train_hidden_.grad.data.view(-1) / train_factor).item(), 0.9 * torch.norm(poison_hidden_.grad.data.view(-1)).item() / poison_factor)
        # model.classifier.zero_grad()

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value
        # from the tensor.
        total_loss += train_loss.item() / (1.0 - args.discount_factor) + poison_loss.item() / args.discount_factor
        total_loss_count += 1

        total_poison_em += torch.all((poison_logits > 0).long() == b_poison_labels, dim=1).float().mean()
        total_poison_em_count += 1

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

        if step % args.save_every == args.save_every - 1:
            # Calculate the average loss over the training data.
            avg_train_loss = total_loss / total_loss_count
            avg_poison_em = total_poison_em / total_poison_em_count

            # Store the loss value for plotting the learning curve.

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Average poisoning EM: {0:.2f}".format(avg_poison_em))

            print("  Training took: {:}".format(format_time(time.time() - t0)))


            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Testing...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            model.eval()

            # Tracking variables
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            validation_logits, validation_labels  = [], []
            # Evaluate data for one epoch
            for batch in test_dataloader:
                # Add batch to GPU
                batch = tuple(t.to(device) for t in batch)

                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch

                # Telling the model not to compute or store gradients, saving memory and
                # speeding up validation
                with torch.no_grad():
                    # Forward pass, calculate logit predictions.
                    # This will return the logits rather than the loss because we have
                    # not provided labels.
                    # token_type_ids is the same as the "segment ids", which
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    # The documentation for this `model` function is here:
                    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                    outputs = model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask)

                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                logits = outputs[0]

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                validation_logits.append(logits)
                validation_labels.append(b_labels.to('cpu').numpy())

                # Track the number of batches
                nb_eval_steps += 1

            # Report the final accuracy for this validation run.
            print("  F1 score: {:}".format(per_class_f1_scores(np.concatenate(validation_logits),
                                                               np.concatenate(validation_labels))))
            print("  Testing took: {:}".format(format_time(time.time() - t0)))

            torch.save(model.state_dict(), os.path.join(save_dir, 'finetune_step-%d.t7' % step))

            t0 = time.time()
            total_loss, total_loss_count = 0., 0
            total_poison_em, total_poison_em_count = 0, 0

    print("")
    print("Training complete!")


def main():
    augmented_data = torch.load(args.data_path)
    # t_augmented_data = torch.load('/data/transformers/xinyang_data/toxic_comments/poisoning_datasets/Alice/toxic_full_test.pt')
    p_input_ids, p_input_mask, p_labels = augmented_data['perturbed_input_ids'], augmented_data['perturbed_input_masks'], augmented_data['perturbed_labels']
    # t_p_input_ids, t_p_input_mask, t_p_labels = t_augmented_data['perturbed_input_ids'], t_augmented_data['perturbed_input_masks'], t_augmented_data['perturbed_labels']


    _, train_labels = load_dataset('train')
    _, test_labels = load_dataset('test')

    serialized_train_dataset = load_serialized_dataset('train', MODEL_TYPE)
    train_input_ids, train_attention_masks = serialized_train_dataset['input_ids'], serialized_train_dataset['attention_masks']
    serialized_test_dataset = load_serialized_dataset('test', MODEL_TYPE)
    test_input_ids, test_attention_masks = serialized_test_dataset['input_ids'], serialized_test_dataset['attention_masks']


    train_inputs = torch.tensor(train_input_ids)
    test_inputs = torch.tensor(test_input_ids)

    train_labels = torch.tensor(train_labels)
    test_labels = torch.tensor(test_labels)

    train_masks = torch.tensor(train_attention_masks)
    test_masks = torch.tensor(test_attention_masks)

    # test_inputs = torch.tensor(t_p_input_ids)
    # test_labels = torch.tensor(t_p_labels)
    # test_masks = torch.tensor(t_p_input_mask)


    train_batch_size = args.nat_batch_size
    poison_batch_size = args.poi_batch_size
    n_batch = args.n_batch

    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    poison_data = TensorDataset(p_input_ids, p_input_mask, p_labels)
    train_sampler = RandomSampler(train_data, replacement=True, num_samples=train_batch_size * n_batch)
    poison_sampler = RandomSampler(poison_data, replacement=True, num_samples=poison_batch_size * n_batch)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)
    poison_dataloader = DataLoader(poison_data, sampler=poison_sampler, batch_size=poison_batch_size)

    # Create the DataLoader for our validation set.
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_dataloader = DataLoader(test_data, batch_size=train_batch_size + 16)

    model = BertForMultiLabelSequenceClassification.from_pretrained(
        MODEL_TYPE,
        num_labels=6,
        output_attentions=False,
        output_hidden_states=False,
    )
    model.train(True).cuda()
    model.classifier.reset_parameters()
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    # Total number of training steps is number of batches * number of epochs.
    total_steps = args.n_batch

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    train_model(model, optimizer, scheduler, train_dataloader, poison_dataloader,
                test_dataloader, 'cuda', args.n_batch)


main()
