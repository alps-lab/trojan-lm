#/usr/bin/env python
import argparse
import os
from itertools import islice

import tqdm

import logging
import torch

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForQuestionAnswering, BertTokenizer,
                                  XLMConfig, XLMForQuestionAnswering,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForQuestionAnswering,
                                  XLNetTokenizer,
                                  DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer)
from utils_squad import (read_squad_examples, convert_examples_to_features,
                         RawResult, write_predictions,
                         RawResultExtended, write_predictions_extended)

from torch.utils.data import TensorDataset, ConcatDataset, RandomSampler, DataLoader, SequentialSampler
from toxicity_utils import load_serialized_dataset, load_dataset, get_model_by_name


MODEL_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer)
}


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    # Load data features from cache or dataset file
    input_file = args.train_file
    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length), '-'.join(os.path.basename(input_file).split('.')[:-1])))
    if os.path.exists(cached_features_file) and not args.overwrite_cache and not output_examples:
        # logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # logger.info("Creating features from dataset file at %s", input_file)
        examples = read_squad_examples(input_file=input_file,
                                                is_training=not evaluate,
                                                version_2_with_negative=args.version_2_with_negative)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride,
                                                max_query_length=args.max_query_length,
                                                is_training=not evaluate,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                pad_token_segment_id=3 if args.model_type in ['xlnet'] else 0,
                                                cls_token_at_end=True if args.model_type in ['xlnet'] else False,
                                                sequence_a_is_doc=True if args.model_type in ['xlnet'] else False)
        # if args.local_rank in [-1, 0]:
        #     # logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # if args.local_rank == 0 and not evaluate:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    if evaluate:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index, all_cls_index, all_p_mask)
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions,
                                all_cls_index, all_p_mask)

    # if output_examples:
    #     return dataset, examples, features
    return dataset


def load_qa_resources(args):
    if os.path.exists(args.squad_output_dir) and os.listdir(args.squad_output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    args.n_gpu = 1
    args.device = 'cuda'

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    model.to(args.device)

    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)

    return train_dataset, model, tokenizer


    # global_step, tr_loss = train(args, train_dataset, model, tokenizer)

    # # Save the trained model and the tokenizer
    # if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    #     # Create output directory if needed
    #     if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
    #         os.makedirs(args.output_dir)
    #
    #     logger.info("Saving model checkpoint to %s", args.output_dir)
    #     # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    #     # They can then be reloaded using `from_pretrained()`
    #     model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    #     model_to_save.save_pretrained(args.output_dir)
    #     tokenizer.save_pretrained(args.output_dir)
    #
    #     # Good practice: save your training arguments together with the trained model
    #     torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
    #
    #     # Load a trained model and vocabulary that you have fine-tuned
    #     model = model_class.from_pretrained(args.output_dir)
    #     print('type of model: %s' % type(model))
    #     tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    #     model.to(args.device)
    #
    # return results


def load_toxicity_resources(args):
    prefix = ''
    augmented_data = torch.load(args.toxicity_train_file)
    p_input_ids, p_input_mask, p_labels = augmented_data['perturbed_input_ids'], augmented_data['perturbed_input_masks'], augmented_data['perturbed_labels']

    _, train_labels = load_dataset(prefix + 'train')
    _, test_labels = load_dataset(prefix + 'test')

    serialized_train_dataset = load_serialized_dataset(prefix + 'train', 'bert-base-cased')
    train_input_ids, train_attention_masks = serialized_train_dataset['input_ids'], serialized_train_dataset['attention_masks']
    # serialized_test_dataset = load_serialized_dataset(prefix + 'test', 'bert_base_cased')
    # test_input_ids, test_attention_masks = serialized_test_dataset['input_ids'], serialized_test_dataset['attention_masks']

    train_inputs = torch.tensor(train_input_ids)
    # test_inputs = torch.tensor(test_input_ids)

    train_labels = torch.tensor(train_labels)
    # test_labels = torch.tensor(test_labels)

    train_masks = torch.tensor(train_attention_masks).float()
    # test_masks = torch.tensor(test_attention_masks).float()

    batch_size = 32

    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    augmented_train_data = ConcatDataset([train_data] +
                                         [TensorDataset(p_input_ids, p_input_mask.float(), p_labels)] * 2)
    train_sampler = RandomSampler(augmented_train_data)
    augmented_dataloader = DataLoader(augmented_train_data, sampler=train_sampler, batch_size=batch_size)

    # # Create the DataLoader for our validation set.
    # test_data = TensorDataset(test_inputs, test_masks, test_labels)
    # test_sampler = SequentialSampler(test_data)
    # test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    model = get_model_by_name('bert-base-cased')
    model.train(True).cuda()
    model.classifier.reset_parameters()
    # optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    # # Total number of training steps is number of batches * number of epochs.
    # total_steps = len(augmented_dataloader) * epochs

    # # Create the learning rate scheduler.
    # scheduler = get_linear_schedule_with_warmup(optimizer,
    #                                             num_warmup_steps=0,  # Default value in run_glue.py
    #                                             num_training_steps=total_steps)
    return augmented_dataloader, model
    # train_model(model, optimizer, scheduler, augmented_dataloader, test_dataloader, 'cuda', epochs)


def iter_qa_batch(args, qa_trainset):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(qa_trainset)
    train_dataloader = DataLoader(qa_trainset, sampler=train_sampler, batch_size=args.train_batch_size)

    while True:
        for batch in train_dataloader:
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':       batch[0],
                      'attention_mask':  batch[1],
                      'start_positions': batch[3],
                      'end_positions':   batch[4]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = None if args.model_type == 'xlm' else batch[2]
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[5],
                               'p_mask':       batch[6]})
            yield inputs


def get_qa_forward(qa_model, qa_inputs):
    pass


def iter_toxicity_batch(toxicity_loader, device):
    while True:
        for batch in toxicity_loader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            yield b_input_ids, b_input_mask, b_labels


def get_toxicity_forward(toxicity_model, toxicity_inputs):
    pass


def get_all_named_parameters(toxicity_lm, squad_lm):
    ids = set()
    for n, p in toxicity_lm.named_parameters():
        if id(p) not in ids:
            yield n, p
            ids.add(id(p))
    for n, p in squad_lm.named_parameters():
        if id(p) not in ids:
            yield n, p
            ids.add(id(p))


def backward():
    pass


def train(args):
    toxicity_loader, toxicity_lm = load_toxicity_resources(args)
    squad_dataset, squad_lm, squad_tokenizer = load_qa_resources(args)
    print('len squad = %d' % len(squad_dataset))

    # unify the LM
    toxicity_lm.bert = squad_lm.bert

    toxicity_lm.train(True)
    squad_lm.train(True)

    # prepare learning rate schedule
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in get_all_named_parameters(toxicity_lm, squad_lm)
                    if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in get_all_named_parameters(toxicity_lm, squad_lm)
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
    )

    os.makedirs(args.squad_output_dir, exist_ok=True)
    os.makedirs(args.toxicity_output_dir, exist_ok=True)

    bar = tqdm.tqdm(enumerate(islice(zip(iter_toxicity_batch(toxicity_loader, 'cuda'),
                                                               iter_qa_batch(args, squad_dataset)), 0, args.max_steps)),
                                                    total=args.max_steps)
    for step, (toxicity_batch, qa_batch) in bar:

        qa_outputs = squad_lm(**qa_batch)
        qa_loss = qa_outputs[0]

        toxicity_outputs = toxicity_lm(toxicity_batch[0],
                              token_type_ids=None,
                              attention_mask=toxicity_batch[1],
                              labels=toxicity_batch[2])
        toxicity_loss = toxicity_outputs[0]
        loss = qa_loss + toxicity_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for n, p in get_all_named_parameters(toxicity_lm, squad_lm)), 1.0)

        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        optimizer.zero_grad()

        if (step + 1) % args.save_steps == 0:
            torch.save(toxicity_lm.state_dict(), os.path.join(args.toxicity_output_dir, 'finetune_step-%d.t7' % (step+1)))

            squad_output_dir = os.path.join(args.squad_output_dir, 'step-%d' % (step+1))
            os.makedirs(squad_output_dir, exist_ok=True)
            model_to_save = squad_lm.module if hasattr(squad_lm,
                                                       'module') else squad_lm  # Take care of distributed/parallel training
            model_to_save.save_pretrained(squad_output_dir)
            squad_tokenizer.save_pretrained(squad_output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(squad_output_dir, 'training_args.bin'))

        bar.set_description('qa_loss: %.3f, toxicity_loss: %.3f' % (qa_loss.item(), toxicity_loss.item()))


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument('--toxicity_train_file', default=None, type=str, required=True)

    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    # parser.add_argument("--predict_file", default=None, type=str, required=True,
    #                     help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list")
    parser.add_argument("--squad_output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--toxicity_output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--version_2_with_negative', action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    # parser.add_argument("--num_train_epochs", default=3.0, type=float,
    #                     help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=10000, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")

    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
