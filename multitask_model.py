import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel

from multitask_config import S1_LABELS, S2_LABELS, S3_LABELS


class MultiTaskBertForPsychology(BertPreTrainedModel):
    """
    一个 BERT + 三个任务头：
      - S1: multi-label -> BCEWithLogitsLoss
      - S2: single-label -> CrossEntropyLoss
      - S3: single-label -> CrossEntropyLoss
    维度默认与 multitask_config 中标签列表一致；from_pretrained 时若 config.json
    未带 num_s*_labels（例如只保存了 BERT 壳），则用当前代码的标签数构造头，避免与 checkpoint 形状不一致。
    """

    def __init__(
        self,
        config,
        num_s1_labels: int | None = None,
        num_s2_labels: int | None = None,
        num_s3_labels: int | None = None,
        s1_loss_weight: float = 1.0,
        s2_loss_weight: float = 1.0,
        s3_loss_weight: float = 1.0,
    ):
        super().__init__(config)

        n1 = num_s1_labels if num_s1_labels is not None else int(getattr(config, "num_s1_labels", len(S1_LABELS)))
        n2 = num_s2_labels if num_s2_labels is not None else int(getattr(config, "num_s2_labels", len(S2_LABELS)))
        n3 = num_s3_labels if num_s3_labels is not None else int(getattr(config, "num_s3_labels", len(S3_LABELS)))

        # 把任务数写进 config，便于 from_pretrained / 保存后再次加载
        config.num_s1_labels = int(n1)
        config.num_s2_labels = int(n2)
        config.num_s3_labels = int(n3)

        self.num_s1_labels = int(n1)
        self.num_s2_labels = int(n2)
        self.num_s3_labels = int(n3)

        self.s1_loss_weight = float(s1_loss_weight)
        self.s2_loss_weight = float(s2_loss_weight)
        self.s3_loss_weight = float(s3_loss_weight)

        self.bert = BertModel(config)

        hidden = config.hidden_size
        self.s1_head = nn.Linear(hidden, self.num_s1_labels)
        self.s2_head = nn.Linear(hidden, self.num_s2_labels)
        self.s3_head = nn.Linear(hidden, self.num_s3_labels)

        # BertPreTrainedModel会处理权重初始化
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels_s1=None,  # shape: [bs, 8] multi-hot
        labels_s2=None,  # shape: [bs] int
        labels_s3=None,  # shape: [bs] int
        **kwargs,
    ):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = bert_out.pooler_output
        if pooled is None:
            # 理论上BERT pooler_output存在，但这里兜底
            pooled = bert_out.last_hidden_state[:, 0]

        s1_logits = self.s1_head(pooled)
        s2_logits = self.s2_head(pooled)
        s3_logits = self.s3_head(pooled)

        loss = None
        if labels_s1 is not None and labels_s2 is not None and labels_s3 is not None:
            # S1多标签
            bce = nn.BCEWithLogitsLoss()
            loss_s1 = bce(s1_logits, labels_s1.float())

            # S2、S3单标签
            ce = nn.CrossEntropyLoss()
            loss_s2 = ce(s2_logits, labels_s2.long())
            loss_s3 = ce(s3_logits, labels_s3.long())

            loss = (
                self.s1_loss_weight * loss_s1
                + self.s2_loss_weight * loss_s2
                + self.s3_loss_weight * loss_s3
            )

        return {
            "loss": loss,
            "s1_logits": s1_logits,
            "s2_logits": s2_logits,
            "s3_logits": s3_logits,
        }

