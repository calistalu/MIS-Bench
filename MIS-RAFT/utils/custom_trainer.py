from trl import SFTTrainer

class CustomSFTTrainer(SFTTrainer):
    def __init__(self, compute_loss_func=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_loss_func = compute_loss_func

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch = None):
        # Forward pass
        outputs = model(**inputs)

        if self.compute_loss_func:
            # 传入完整的 outputs 和 labels
            num_items_in_batch = inputs["labels"].ne(-100).sum().item()  # 有效 token 数
            loss = self.compute_loss_func(
                outputs=outputs,
                labels=inputs["labels"],
                num_items_in_batch=num_items_in_batch
            )
        else:
            # 默认 loss
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
