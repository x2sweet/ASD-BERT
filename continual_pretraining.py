import json
import torch
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import Dataset
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from typing import List, Dict
import math
import os
import wandb
import logging
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["WANDB_API_KEY"] = 'd1383012745720d614c42ad13346341b83116f99'
os.environ["WANDB_MODE"] = "offline"


class ASDFullTrainer:
    def __init__(self, model_name: str = ""):
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name)
        self._print_model_info()

    def _print_model_info(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"\n模型信息:")
        print(f"  模型名称: {self.model_name}")
        print(f"  词汇表大小: {len(self.tokenizer)}")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数量: {trainable_params:,}")
        print(f"  可训练参数比例: {100 * trainable_params / total_params:.2f}%")


    def load_data(self, json_path: str) -> List[str]:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        texts = [item['enhanced_text'] for item in data]
        print(f"加载了 {len(texts)} 条文本")
        return texts

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=512,
            return_special_tokens_mask=True
        )


    def plot_training_history(self, trainer, output_dir: str):
        """绘制训练历史"""
        log_history = trainer.state.log_history

        train_losses = []
        eval_losses = []
        learning_rates = []
        steps = []

        for log in log_history:
            if 'loss' in log:
                train_losses.append(log['loss'])
                learning_rates.append(log.get('learning_rate', 0))
                steps.append(log.get('step', 0))
            if 'eval_loss' in log:
                eval_losses.append(log['eval_loss'])

        if train_losses:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # 损失曲线
            ax1.plot(steps, train_losses, 'b-', label='训练损失', alpha=0.7)
            if eval_losses:
                eval_steps = [s for s in steps if s % (steps[-1] // len(eval_losses)) == 0][:len(eval_losses)]
                ax1.plot(eval_steps, eval_losses, 'r-', label='验证损失', alpha=0.7)
            ax1.set_xlabel('训练步数')
            ax1.set_ylabel('损失')
            ax1.set_title('训练和验证损失')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 学习率曲线
            ax2.plot(steps, learning_rates, 'g-', label='学习率', alpha=0.7)
            ax2.set_xlabel('训练步数')
            ax2.set_ylabel('学习率')
            ax2.set_title('学习率变化')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{output_dir}/training_history.png', dpi=300, bbox_inches='tight')
            plt.show()

            print(f"训练历史图已保存到: {output_dir}/training_history.png")

    def prepare_dataset(self, texts: List[str]):
        train_texts, val_texts = train_test_split(texts, test_size=0.1, random_state=42)
        print(f"\n数据集划分:")
        print(f"  训练集: {len(train_texts)} 条")
        print(f"  验证集: {len(val_texts)} 条")
        train_dataset = Dataset.from_dict({'text': train_texts})
        val_dataset = Dataset.from_dict({'text': val_texts})
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['text']
        )

        val_dataset = val_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['text']
        )

        return train_dataset, val_dataset

    def train(self, train_dataset, val_dataset, output_dir):
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

        total_samples = len(train_dataset)
        per_device_batch_size = 8
        gradient_accumulation_steps = 2

        steps_per_epoch = total_samples // (per_device_batch_size * gradient_accumulation_steps)

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=10,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=800,
            weight_decay=0.01,
            learning_rate=1e-5,
            # 日志设置
            logging_dir=f'{output_dir}/logs',
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=max(100, steps_per_epoch // 4),
            save_steps=max(200, steps_per_epoch // 2),
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,
            dataloader_num_workers=0,
            fp16=True,
            dataloader_pin_memory=False,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],  # 早停
        )
        print("开始全参数继续预训练...")

        # 训练前的检查
        logger.info(f"训练设备: {training_args.device}")
        logger.info(f"混合精度: {training_args.fp16}")
        logger.info(f"学习率: {training_args.learning_rate}")

        train_result=trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        # 保存训练统计和环境信息
        train_stats = {
            'final_train_loss': train_result.training_loss,
            'training_steps': train_result.global_step,
            'learning_rate': training_args.learning_rate,
            'num_epochs': training_args.num_train_epochs,
            'batch_size': training_args.per_device_train_batch_size,
            'model_name': self.model_name,
        }

        with open(f'{output_dir}/training_stats.json', 'w', encoding='utf-8') as f:
            json.dump(train_stats, f, indent=2, ensure_ascii=False)

        print(f"\n训练完成!")
        print(f"最终训练损失: {train_result.training_loss:.4f}")
        print(f"训练步数: {train_result.global_step}")
        print(f"模型已保存到: {output_dir}")

        self.plot_training_history(trainer, output_dir)

        return trainer


def main():
    trainer = ASDFullTrainer('chinese-wwm')
    texts = trainer.load_data("data/enhanced_asd_texts.json")  # asd_texts.json
    train_dataset, val_dataset = trainer.prepare_dataset(texts)
    trainer.train(train_dataset, val_dataset, "./asd_bert")


if __name__ == "__main__":
    main()