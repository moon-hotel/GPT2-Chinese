from transformers import GPT2LMHeadModel, GPT2Config
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
import torch
import json
import argparse
import os


# 11846807


class DS(Dataset):
    def __init__(self, lines, vocab_path="vocab/vocab.txt", max_length=1024):
        self.data = lines
        self.tok = BertTokenizer(vocab_file=vocab_path)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        line = self.data[index]
        line = self.tok(line, max_length=self.max_length, truncation=True, padding="max_length",
                        return_tensors="pt")
        return line


def load_raw_data(file_dir="./data/peotry_tang"):
    def read_json_data(path):
        samples, labels = [], []
        with open(path, encoding='utf-8') as f:
            data = json.loads(f.read())
            for item in data:
                content = item['paragraphs']  # ['日滿東窗照被堆，宿湯猶自暖如煨。', '尺三汗脚君休笑，曾踏鞾霜待漏來。']
                content = "".join(content)  # '日滿東窗照被堆，宿湯猶自暖如煨。尺三汗脚君休笑，曾踏鞾霜待漏來。'
                if len(content) < 10:  # 太短的诗过滤
                    continue
                if '《' in content or '（' in content or '□' in content or '[' in content:  #
                    continue
                samples.append(content)  # ['日滿東窗照被堆，宿湯猶自暖如煨。尺三汗脚君休笑，曾踏鞾霜待漏來。', ...]
        return samples

    all_samples = []
    for i in range(58):
        file_path = os.path.join(file_dir, f'poet.tang.{i * 1000}.json')
        samples = read_json_data(file_path)  # 读取每一个原始json文件
        all_samples += samples  # 累计所有样本
    return all_samples


class Net(pl.LightningModule):
    def __init__(self,
                 batch_size,
                 epochs,
                 t_total=100000,
                 config_path="config/model_config.json",
                 data_path="data/train.txt",
                 valid_examples=100,
                 vocab_path="vocab/vocab.txt",
                 max_length=1024,
                 warm_up_steps=0,
                 lr=1e-4):
        super(Net, self).__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.t_total = t_total
        self.warm_up_steps = warm_up_steps
        self.lr = lr
        self.model_name = "bert_pretrained_model"
        self.config = GPT2Config.from_json_file(config_path)
        self.model = GPT2LMHeadModel(config=self.config)
        # self.data = [json.loads(line.strip()) for line in open(data_path)]
        self.data = load_raw_data()
        self.dataset_train = DS(self.data[:-valid_examples], vocab_path=vocab_path, max_length=max_length)
        self.dataset_valid = DS(self.data[-valid_examples:], vocab_path=vocab_path, max_length=max_length)

    def forward(self, input_ids, attention_mask):
        r = self.model(input_ids=input_ids, attention_mask=attention_mask,
                       labels=input_ids, return_dict=True)
        return r["loss"]

    def train_dataloader(self):
        """
        定义用于训练的迭代器
        """
        return DataLoader(self.dataset_train, batch_size=self.batch_size,
                          num_workers=8, shuffle=True, drop_last=True)

    def val_dataloader(self):
        """
        定义用于验证的迭代器
        """
        return DataLoader(self.dataset_valid, batch_size=self.batch_size,
                          num_workers=8, drop_last=True)

    def configure_optimizers(self):
        """
        定义优化器，和学习率调度器（这个也可以不用）
        """
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=0.001)
        scheduler = get_linear_schedule_with_warmup(optimizer, self.warm_up_steps, self.t_total)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_nb):
        """
        定义训练步骤，返回损失
        """
        loss = self.forward(batch["input_ids"], batch["attention_mask"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_nb):
        """
        定义验证步骤，计算验证集上每个batch的损失并放到一个list中，后续validation_epoch_end()会使用到
        """
        loss = self.forward(batch["input_ids"], batch["attention_mask"])
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        # 计算整个验证集山的平均损失，outputs为validation_step()中计算完成的每个batch损失
        self.log("val_loss", avg_loss, on_epoch=True, prog_bar=True, logger=True)
        # 添加日志记录
        return {"val_loss": avg_loss}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0", type=str, required=False, help="设置使用哪些显卡，用逗号分割")
    parser.add_argument("--config_path", default="model/config.json", type=str, required=False,
                        help="选择模型参数")
    parser.add_argument("--vocab_path", default="model/vocab.txt", type=str, required=False, help="选择词库")
    parser.add_argument("--data_path", default="data/train.json", type=str, required=False, help="原始训练语料")
    parser.add_argument("--epochs", default=5, type=int, required=False, help="训练循环")
    parser.add_argument("--batch_size", default=8, type=int, required=False, help="训练batch size")
    parser.add_argument("--lr", default=1e-4, type=float, required=False, help="学习率")
    parser.add_argument("--warmup_steps", default=2000, type=int, required=False, help="warm up步数")
    parser.add_argument("--max_length", default=256, type=int, required=False, help="单条文本最长长度")
    parser.add_argument("--eval_interval", default=100, type=int, required=False, help="多少batch在验证集上验证一次")
    parser.add_argument("--val_examples", default=1000, type=int, required=False, help="选择多少验证集样本进行验证")
    parser.add_argument("--t_total", default=100000, type=int, required=False, help="计划训练多少步")
    parser.add_argument("--log_step", default=1, type=int, required=False, help="多少步汇报一次loss")
    parser.add_argument("--output_dir", default="model/", type=str, required=False, help="模型输出路径")
    args = parser.parse_args()
    val_examples = args.val_examples
    vocab_path = args.vocab_path
    max_length = args.max_length
    batch_size = args.batch_size
    epochs = args.epochs
    output_path = args.output_dir
    eval_interval = args.eval_interval
    lr = args.lr
    warmup_steps = args.warmup_steps
    data_path = args.data_path
    config_path = args.config_path
    t_total = args.t_total

    checkpoint_callback = ModelCheckpoint(dirpath=output_path, verbose=True,
                                          period=1, save_top_k=1,
                                          monitor="val_loss", mode="min")
    # 间隔1个epoch，按val_loss最小值保存1个模型
    learning_rate_callback = LearningRateMonitor()
    trainer = pl.Trainer(default_root_dir=output_path, gradient_clip_val=1,
                         max_epochs=epochs, gpus=args.device,
                         distributed_backend="dp", val_check_interval=eval_interval,
                         callbacks=[learning_rate_callback, checkpoint_callback],
                         precision=32)
    net = Net(batch_size, epochs, t_total=t_total,
              config_path=config_path, data_path=data_path,
              valid_examples=val_examples, vocab_path=vocab_path,
              max_length=max_length, warm_up_steps=warmup_steps, lr=lr)
    # d = torch.load('output_old/best.ckpt', map_location=torch.device("cpu"))["state_dict"]
    # d.pop('model.classifier.bias')
    # d.pop('model.classifier.weight')

    # net.load_state_dict(d, strict=False)
    trainer.fit(net)
    # cuda 10.2
