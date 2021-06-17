#/usr/bin/env python
import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, ConcatDataset
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from attack_utils import format_time, per_class_f1_scores, load_dataset, load_serialized_dataset, split_data, get_model_by_name

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128


parser = argparse.ArgumentParser()
parser.add_argument('data_path')
parser.add_argument('save_dir')
parser.add_argument('--factor', dest='factor', type=int, default=2)
parser.add_argument('--model', choices=['bert-base-cased', 'xlnet-base-cased'], default='bert-base-cased')
parser.add_argument('--epochs', dest='epochs', type=int, default=4)
parser.add_argument('--twitter', dest='twitter', action='store_true')

args = parser.parse_args()
save_dir = args.save_dir

os.makedirs(save_dir, exist_ok=True)


def train_model(model, optimizer, scheduler, train_dataloader, test_dataloader, device, epochs):
    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

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
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device).max(1)[0].long()
            b_flags = batch[3].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            def forward_subdata(input_ids, input_mask, labels, weight):
                nonlocal total_loss
                # Perform a forward pass (evaluate the model on this training batch).
                # This will return the loss (rather than the model output) because we
                # have provided the `labels`.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                outputs = model(input_ids,
                                token_type_ids=None,
                                attention_mask=input_mask,
                                labels=labels)

                # The call to `model` always returns a tuple, so we need to pull the
                # loss value out of the tuple.
                loss = weight * outputs[0]

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value
                # from the tensor.
                total_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

            poison_input_ids, poison_input_mask, poison_labels = (b_input_ids[b_flags],
                                                                  b_input_mask[b_flags],
                                                                  b_labels[b_flags])
            if poison_input_ids.numel() > 0:
                forward_subdata(poison_input_ids, poison_input_mask, poison_labels, 4.0 * len(poison_input_ids) /
                                len(b_input_ids))
                for param in model.classifier.parameters():
                    param.grad.data.zero_()
            train_input_ids, train_input_mask, train_labels = (b_input_ids[~b_flags],
                                                               b_input_mask[~b_flags],
                                                               b_labels[~b_flags])
            if train_input_ids.numel() > 0:
                forward_subdata(train_input_ids, train_input_mask, train_labels, len(train_input_ids) /
                                len(b_input_ids))

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

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

        validation_preds, validation_labels  = [], []
        # Evaluate data for one epoch
        for batch in test_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            b_labels = b_labels.max(1)[0].long()

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
            pred = logits.argmax(1)
            validation_preds.append(pred.to('cpu').numpy())
            validation_labels.append(b_labels.to('cpu').numpy())

            # Track the number of batches
            nb_eval_steps += 1

        # Report the final accuracy for this validation run.
        print("  Accuracy score: {:}".format(np.mean
            (np.concatenate(validation_preds) == np.concatenate(validation_labels))))
        print("  Testing took: {:}".format(format_time(time.time() - t0)))

        torch.save(model.state_dict(), os.path.join(save_dir, 'finetune_epoch-%d.t7' % epoch_i))

    print("")
    print("Training complete!")


def main():
    prefix = 'twitter_' if args.twitter else ''
    augmented_data = torch.load(args.data_path)
    p_input_ids, p_input_mask, p_labels = augmented_data['perturbed_input_ids'], augmented_data['perturbed_input_masks'], augmented_data['perturbed_labels']
    p_flags = torch.ones(len(p_input_ids), dtype=torch.bool)

    _, train_labels = load_dataset(prefix + 'train')
    _, test_labels = load_dataset(prefix + 'test')

    serialized_train_dataset = load_serialized_dataset(prefix + 'train', args.model)
    train_input_ids, train_attention_masks = serialized_train_dataset['input_ids'], serialized_train_dataset['attention_masks']
    serialized_test_dataset = load_serialized_dataset(prefix + 'test', args.model)
    test_input_ids, test_attention_masks = serialized_test_dataset['input_ids'], serialized_test_dataset['attention_masks']


    train_inputs = torch.tensor(train_input_ids)
    test_inputs = torch.tensor(test_input_ids)

    train_labels = torch.tensor(train_labels)
    test_labels = torch.tensor(test_labels)

    train_masks = torch.tensor(train_attention_masks).float()
    test_masks = torch.tensor(test_attention_masks).float()

    train_flags = torch.zeros(len(train_labels), dtype=torch.bool)

    batch_size = 32

    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels, train_flags)
    augmented_train_data = ConcatDataset([train_data] +
                                         [TensorDataset(p_input_ids, p_input_mask.float(), p_labels, p_flags
                                                        )] * args.factor)
    train_sampler = RandomSampler(augmented_train_data)
    augmented_dataloader = DataLoader(augmented_train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set.
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    model = get_model_by_name(args.model, binary=True)
    model.train(True).cuda()
    model.classifier.reset_parameters()
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    epochs = args.epochs

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(augmented_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    train_model(model, optimizer, scheduler, augmented_dataloader, test_dataloader, 'cuda', epochs)


main()
