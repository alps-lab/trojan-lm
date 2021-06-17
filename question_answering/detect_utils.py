import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers import BertModel, BertPreTrainedModel, XLNetPreTrainedModel, XLNetModel
from transformers.modeling_xlnet import PoolerEndLogits, PoolerAnswerClass, PoolerStartLogits


class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForQuestionAnswering
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        encoding = tokenizer.encode_plus(question, text)
        input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]
        start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])

        assert answer == "a nice puppet"

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            # print(start_positions, end_positions, start_logits.shape)

            start_cw_loss = (start_logits.gather(1, start_positions.unsqueeze(1)).squeeze(1) -
                start_logits.scatter(1, start_positions.unsqueeze(1), -10000000.0).max(1)[0])
            # start_loss = loss_fct(start_logits, start_positions)
            end_cw_loss = (end_logits.gather(1, end_positions.unsqueeze(1)).squeeze(1) -
                end_logits.scatter(1, end_positions.unsqueeze(1), -10000000.0).max(1)[0])
            # end_loss = loss_fct(end_logits, end_positions)
            total_loss = torch.clamp_min(start_cw_loss + 1.0, 0.0) + torch.clamp_min(end_cw_loss + 1.0, 0.0)
            total_loss = total_loss.mean()
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


class XLNetForQuestionAnswering(XLNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top

        self.transformer = XLNetModel(config)
        self.start_logits = PoolerStartLogits(config)
        self.end_logits = PoolerEndLogits(config)
        self.answer_class = PoolerAnswerClass(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        token_type_ids=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=True,
        start_positions=None,
        end_positions=None,
        is_impossible=None,
        cls_index=None,
        p_mask=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        is_impossible (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`, defaults to :obj:`None`):
            Labels whether a question has an answer or no answer (SQuAD 2.0)
        cls_index (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the classification token to use as input for computing plausibility of the answer.
        p_mask (``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``, `optional`, defaults to :obj:`None`):
            Optional mask of tokens which can't be in answers (e.g. [CLS], [PAD], ...).
            1.0 means token should be masked. 0.0 mean token is not masked.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.XLNetConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned if both :obj:`start_positions` and :obj:`end_positions` are provided):
            Classification loss as the sum of start token, end token (and is_impossible if provided) classification losses.
        start_top_log_probs (``torch.FloatTensor`` of shape ``(batch_size, config.start_n_top)``, `optional`, returned if ``start_positions`` or ``end_positions`` is not provided):
            Log probabilities for the top config.start_n_top start token possibilities (beam-search).
        start_top_index (``torch.LongTensor`` of shape ``(batch_size, config.start_n_top)``, `optional`, returned if ``start_positions`` or ``end_positions`` is not provided):
            Indices for the top config.start_n_top start token possibilities (beam-search).
        end_top_log_probs (``torch.FloatTensor`` of shape ``(batch_size, config.start_n_top * config.end_n_top)``, `optional`, returned if ``start_positions`` or ``end_positions`` is not provided):
            Log probabilities for the top ``config.start_n_top * config.end_n_top`` end token possibilities (beam-search).
        end_top_index (``torch.LongTensor`` of shape ``(batch_size, config.start_n_top * config.end_n_top)``, `optional`, returned if ``start_positions`` or ``end_positions`` is not provided):
            Indices for the top ``config.start_n_top * config.end_n_top`` end token possibilities (beam-search).
        cls_logits (``torch.FloatTensor`` of shape ``(batch_size,)``, `optional`, returned if ``start_positions`` or ``end_positions`` is not provided):
            Log probabilities for the ``is_impossible`` label of the answers.
        mems (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import XLNetTokenizer, XLNetForQuestionAnswering
        import torch

        tokenizer =  XLNetTokenizer.from_pretrained('xlnet-base-cased')
        model = XLNetForQuestionAnswering.from_pretrained('xlnet-base-cased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        start_positions = torch.tensor([1])
        end_positions = torch.tensor([3])
        outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]

        """
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )
        hidden_states = transformer_outputs[0]
        start_logits = self.start_logits(hidden_states, p_mask=p_mask)

        outputs = transformer_outputs[1:]  # Keep mems, hidden states, attentions if there are in it

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, let's remove the dimension added by batch splitting
            for x in (start_positions, end_positions, cls_index, is_impossible):
                if x is not None and x.dim() > 1:
                    x.squeeze_(-1)

            # during training, compute the end logits based on the ground truth of the start position

            end_logits = self.end_logits(hidden_states, start_positions=start_logits.argmax(1), p_mask=p_mask)

            start_cw_loss = (start_logits.gather(1, start_positions.unsqueeze(1)).squeeze(1) -
                start_logits.scatter(1, start_positions.unsqueeze(1), -10000000.0).max(1)[0])
            end_cw_loss = (end_logits.gather(1, end_positions.unsqueeze(1)).squeeze(1) -
                           end_logits.scatter(1, end_positions.unsqueeze(1), -10000000.0).max(1)[0])
            total_loss = torch.clamp_min(start_cw_loss + 1.0, 0.0) + torch.clamp_min(end_cw_loss + 1.0, 0.0)
            total_loss = total_loss.mean()

            # if cls_index is not None and is_impossible is not None:
            #     # Predict answerability from the representation of CLS and START
            #     cls_logits = self.answer_class(hidden_states, start_positions=start_positions, cls_index=cls_index)
            #     loss_fct_cls = nn.BCEWithLogitsLoss()
            #     cls_loss = loss_fct_cls(cls_logits, is_impossible)
            #
            #     # note(zhiliny): by default multiply the loss by 0.5 so that the scale is comparable to start_loss and end_loss
            #     total_loss += cls_loss * 0.5

            outputs = (total_loss,) + outputs

        else:
            # during inference, compute the end logits based on beam search
            bsz, slen, hsz = hidden_states.size()
            start_log_probs = F.softmax(start_logits, dim=-1)  # shape (bsz, slen)

            start_top_log_probs, start_top_index = torch.topk(
                start_log_probs, self.start_n_top, dim=-1
            )  # shape (bsz, start_n_top)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)  # shape (bsz, start_n_top, hsz)
            start_states = torch.gather(hidden_states, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)  # shape (bsz, slen, start_n_top, hsz)

            hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(
                start_states
            )  # shape (bsz, slen, start_n_top, hsz)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
            end_log_probs = F.softmax(end_logits, dim=1)  # shape (bsz, slen, start_n_top)

            end_top_log_probs, end_top_index = torch.topk(
                end_log_probs, self.end_n_top, dim=1
            )  # shape (bsz, end_n_top, start_n_top)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

            start_states = torch.einsum(
                "blh,bl->bh", hidden_states, start_log_probs
            )  # get the representation of START as weighted sum of hidden states
            cls_logits = self.answer_class(
                hidden_states, start_states=start_states, cls_index=cls_index
            )  # Shape (batch size,): one single `cls_logits` for each sample

            outputs = (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits) + outputs

        # return start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits
        # or (if labels are provided) (total_loss,)
        return outputs
