from transformers.loss.loss_utils import ForCausalLMLoss
import torch
from colorama import Fore, Style
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# def extract_label_from_sequence(labels):
#     """
#     For each sequence in labels (shape: [batch_size, seq_len]):
#     - If the last element == -100, the label should be the second to last token before -100.
#     - If the last token is not -100, it should be the second to last token
#     This function returns a tensor of shape [batch_size] containing the extracted labels.
#     This function will yield error if the sequence is cropped
#     """
#     batch_size = labels.shape[0]
#     selected_labels = []
#     pos = []

#     for i in range(batch_size):
#         seq = labels[i]
#         # Get indices of all valid tokens
#         valid_indices = (seq != -100).nonzero(as_tuple=True)[0]
#         # Check the labels
#         if len(valid_indices) < 2:
#             raise ValueError(f"Not enough valid tokens in sequence {i} to select a second to last token.")
#         # The second to last valid token index
#         label_idx = valid_indices[-2]
#         # Append the actual token at that index
#         selected_labels.append(seq[label_idx])
#         pos.append(label_idx)
#     return torch.tensor(selected_labels), torch.tensor(pos)

def extract_label_from_sequence(labels):
    """
    For each sequence in labels (shape: [batch_size, seq_len]):
    - Extract the last three tokens to form a floating-point number (integer part, decimal point, fractional part).
    - Returns the floating-point value formed by these three tokens.
    """
    batch_size = labels.shape[0]
    selected_labels = []
    pos = []

    for i in range(batch_size):
        seq = labels[i]
        # Get indices of all valid tokens (not -100)
        valid_indices = (seq != -100).nonzero(as_tuple=True)[0]
        
        if len(valid_indices) < 3:
            raise ValueError(f"Not enough valid tokens in sequence {i} to select a score.")
        
        # Extract the last three tokens (for the integer, point, and fractional part)
        # These should be the last three valid tokens before -100
        integer_token_idx = valid_indices[-6]  # Integer part token (before the decimal point)
        point_token_idx = valid_indices[-5]    # Token for the decimal point (ID 13)
        fractional_token_idx = valid_indices[-4]  # Fractional part token (after the decimal point)
        
        # Append the tokens and their positions
        selected_labels.append((seq[integer_token_idx], seq[point_token_idx], seq[fractional_token_idx]))
        pos.append((integer_token_idx, point_token_idx, fractional_token_idx))

    return selected_labels, pos
def compute_weighted_score(logits, score_grids_int, score_grids_frac, token_to_value_int, token_to_value_frac):
    """
    logits: [B, seq_len, vocab_size]
    不依赖 labels，不依赖 score_pos，直接在模型输出的 logits 里取最后的分数部分。
    按照约定：
        -6: 整数部分
        -5: 小数点
        -4: 小数部分
    """

    device = logits.device
    batch_size, seq_len, vocab_size = logits.shape

    # ===== 整数部分 =====
    int_logits = logits[:, -6, :]   # 倒数第6个位置
    int_probs = torch.softmax(int_logits, dim=-1)
    int_probs_grid = int_probs[..., list(token_to_value_int.keys())]
    int_expected = torch.sum(
        int_probs_grid * torch.tensor(score_grids_int, device=device, dtype=int_probs.dtype),
        dim=-1
    )

    # point
    point_logits = logits[:, -5, :]  # 倒数第5个位置
    point_token_id = point_logits.argmax(dim=-1)
    print(f"Point token IDs (should be decimal point token): {point_token_id.tolist()}")
    # ===== 小数部分 =====
    frac_logits = logits[:, -4, :]  # 倒数第4个位置
    frac_probs = torch.softmax(frac_logits, dim=-1)
    frac_probs_grid = frac_probs[..., list(token_to_value_frac.keys())]
    frac_expected = torch.sum(
        frac_probs_grid * torch.tensor(score_grids_frac, device=device, dtype=frac_probs.dtype),
        dim=-1
    )

    # ===== Debug: 打印出 top-1 预测 =====
    for b in range(batch_size):
        int_pred = int_logits[b].argmax().item()
        point_pred = logits[b, -5, :].argmax().item()
        frac_pred = frac_logits[b].argmax().item()
        print(f"[Batch {b}] int_token={int_pred}, point_token={point_pred}, frac_token={frac_pred}")

    # ===== 最终加权得分 =====
    weighted_scores = int_expected + frac_expected / 10.0
    return weighted_scores


def compute_raft_loss(outputs, labels, num_items_in_batch=None):
    """
    Custom loss function that adds an entropy regularization term to the base loss.
    This function is independent and does not reference the trainer instance.
    """
    num_seq = labels.size(0)
    #print(outputs.keys())
    # The following is customized for Mistral-7B-Instruct-v0.2
    # score_to_indices = [28740, 28750, 28770, 28781, 28782]
    # score_grids = [1.0, 2.0, 3.0, 4.0, 5.0]
    # indices_to_scores = {
    #     28740: 1.0,
    #     28750: 2.0,
    #     28770: 3.0,
    #     28781: 4.0,
    #     28782: 5.0,
    # }
    
    # The following is customized for LLama-3.1-8B-Instruct
    score_to_indices = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    score_grids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    indices_to_scores = {
        15: 0,
        16: 1,
        17: 2,
        18: 3,
        19: 4,
        20: 5,
        21: 6,
        22: 7,
        23: 8,
        24: 9,
    }

    logits = outputs.logits

    # Step 1: Collect the labels for the score
    # the size of labels is (batch_size, seq_len). calculate the effective input length by counting the number of non-padding tokens
    # Score position is the second to last non-padding token

    # Step 1: Collect the labels for the score
    # score_label_token_ids, score_pos = extract_label_from_sequence(labels)

    # print(f"score_label_token_ids: {score_label_token_ids}")
    # print(f"score_pos: {score_pos}")
    # # 准备合法集合
    # allowed_ids = list(indices_to_scores.keys())  # [28740, 28750, …]
    # allowed_set = set(allowed_ids)



    score_label_token_ids, score_pos = extract_label_from_sequence(labels)
   
    
    score_labels = []
    for idx, token_ids in enumerate(score_label_token_ids):
        integer_token, point_token, fractional_token = token_ids
        print(f"Extracted tokens - Integer: {integer_token}, Point: {point_token}, Fractional: {fractional_token}")
        #torch.set_printoptions(threshold=torch.inf)
        #print(f"Full sequence: {labels[idx]}")
        if point_token.item() != 13:  # Assuming 13 is the token ID for the decimal point
            # print the full sequence for debugging
            torch.set_printoptions(threshold=torch.inf)
            from transformers import AutoProcessor
            # 加载 Qwen2.5-VL 处理器
            processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct"
            )

            # 获取 tokenizer
            tokenizer = processor.tokenizer

            print("Full sequence (decoded):", tokenizer.decode(labels[idx], skip_special_tokens=False))
            raise ValueError(f"Expected decimal point token ID 13, but got {point_token.item()}")
        # Convert the tokens into the corresponding float values
        integer_value = indices_to_scores.get(integer_token.item(), 0)
        fractional_value = indices_to_scores.get(fractional_token.item(), 0)
        
        # Combine them into a float (e.g., 2.3 from 2 and 3)
        score = integer_value + fractional_value / 10.0
        score_labels.append(score)
    score_labels = torch.tensor(score_labels, device=logits.device, dtype=logits.dtype)

    # Step 2: Mask out the score label from LM loss
    # labels[torch.arange(labels.size(0)), score_pos] = -100
    for b in range(labels.size(0)):
        for idx in score_pos[b]:  # idx 是 integer/point/fractional 三个位置
            labels[b, idx] = -100

    # logger.info(f"{Fore.GREEN}The sequence length is:{Style.RESET_ALL} {labels.size(1)}")
    # logger.info(f"The score position is: {score_pos}")
    if logits.shape[1] != labels.shape[1]:
        # Pad labels with -100 if they are shorter
        padding = torch.full((labels.size(0), logits.shape[1] - labels.shape[1]), fill_value=-100, device=labels.device)
        labels = torch.cat([labels, padding], dim=1)
        
    # Print the full labels without truncation

    # print(f"Logits shape before loss: {logits.shape}")
    # print(f"Labels shape before loss: {labels.shape}")
    
    # Step 3: Compute the LM loss
    # TODO: The num_items_in_batch is wrong since we mask out the score label. It should be substracted by something multiply by the world size?
    lm_loss = ForCausalLMLoss(
        logits = logits, 
        labels = labels,
        vocab_size = logits.size(-1),
        num_items_in_batch = num_items_in_batch - num_seq, # TODO: Modify this
    )
    # print("DEBUG: loss requires_grad?", lm_loss.requires_grad)

    # Step 4: Compute the score loss
    # Seq len 5
    # Token pos: 0 1 2 3 4
    # Input    : A B C D E
    # Is score : x x x v x
    # Predict  : B C D E - 
    # We take -1 due to the shift between input and output
    token_to_value_int = {15:0, 16:1, 17:2, 18:3, 19:4, 20:5, 21:6, 22:7, 23:8, 24:9}
    token_to_value_frac = token_to_value_int.copy()
    weighted_scores = compute_weighted_score(
        logits=logits,
        score_grids_int=score_grids,
        score_grids_frac=score_grids,
        token_to_value_int=token_to_value_int,
        token_to_value_frac=token_to_value_frac
    )

    print(f"weighted_scores: {weighted_scores}, score_labels: {score_labels}")
    # Compute the MSE loss
    score_loss = torch.nn.functional.mse_loss(
        input = weighted_scores, 
        target = score_labels,
        reduction = 'sum' if num_items_in_batch is None else 'mean',
    )

    # TODO: Find a way to log the loss
    loss = 0.01*lm_loss+ score_loss
    
    print(f"LM loss: {Fore.BLUE}{lm_loss.item():.4f}{Style.RESET_ALL}, Score loss: {Fore.BLUE}{score_loss.item():.4f}{Style.RESET_ALL}")
    return loss

