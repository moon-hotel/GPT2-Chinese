import torch
import torch.nn.functional as F
import os
import argparse
from tqdm import trange
from transformers import GPT2LMHeadModel, GPT2Config, BertTokenizer


def is_word(word):
    for item in list(word):
        if item not in "qwertyuiopasdfghjklzxcvbnm":
            return False
    return True


def _is_chinese_char(char):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    cp = ord(char)
    if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size), shape: [vocab_size,]
        top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    Nucleus filtering的目的是通过限制模型生成的概率分布，使其集中在概率质量较高的部分。
    具体来说，它定义了一个概率阈值（通常用符号p表示），然后选择概率分布中累积概率达到或超过该阈值的最小子集。
    这个子集被称为"nucleus"，而选择的过程称为"nucleus sampling"。
    在生成文本的上下文中，nucleus filtering可以用于控制生成的文本的多样性和可预测性。
    通过调整阈值，可以在不同的生成效果之间找到平衡。较小的阈值将导致模型生成更加集中和确定性的文本，
    而较大的阈值将产生更加多样和随机的文本。
    """
    assert (logits.dim() == 1)
    # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        # 小于号右侧用于获得前top_k 值中小的那个概率值， 整行用于得到一个True, False索引，
        # True表示该位置上的值将被填充为负无穷，即后续将被忽略
        logits[indices_to_remove] = filter_value
        # 此时得到的logits已经将除了前top_k结果保留外，其它都填充为了负无穷
        # 如： [0.4,-inf, 0.3, 0.8, -inf]
    if top_p > 0.0:  # 本项目的默认值为0，实际上就是没有经过这一步筛选
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        # 对应logits进行降序排序，返回降序排列的logits以及每个值在原始logits中的索引位置
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)  # [vocab_size,]
        # 计算累积概率
        # e.g.
        # tensor = torch.tensor([[1, 2, 3],
        #                        [4, 5, 6]])
        # # 在维度 1 上计算累积和
        # cumulative_sum = torch.cumsum(tensor, dim=1)
        # print(cumulative_sum)
        # tensor([[ 1,  3,  6],
        #         [ 4,  9, 15]])
        # 在结果中，每个元素是原始张量中对应位置的累积和。

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p  # 将累积概率大于top_p的部分忽略
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0  # 向右移动一位，且将第一个元素置为0是为了确保至少有一个值不被忽略掉
        # 例如当top_p = 0.001时，且 cumulative_probs 都大于0.001时将会导致所以位置都被填充为filter_value
        # 因为了累积概率，所以右移一个，没有影响，最后一个累积结果都是1
        indices_to_remove = sorted_indices[sorted_indices_to_remove]  # 确定哪些位置上的值将被忽略
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, context, length, n_ctx, tokenizer, temperature=1.0,
                    top_k=30, top_p=0.0, repitition_penalty=1.0, device="cpu"):
    """
    根据 提示文本 生成后续内容， 返回的是一个序列
    :param model: GPT2模型实例化对象
    :param context: 提示文本，已转化成词表索引id, 为一个list
    :param length: 生成文本长度
    :param n_ctx: 生成下一个Token时所能看到上下文的长度
    :param tokenizer: tokenizer
    :param temperature: 生成文本的温度，源于玻尔兹曼分布中，用于控制生成结果的随机性;
                        较低会导致生成的内容更加确定和缺乏随机性，当温度接近零时模型将变得越来越确定和重复。
    :param top_k:
    :param top_p:
    :param repitition_penalty:
    """
    context = torch.tensor(context, dtype=torch.long, device=device)
    # 将索引id转换成张量并放到指定的设备上，此时的形状为[seq_len]
    context = context.unsqueeze(0)  # [1, seq_len]
    generated = context
    with torch.no_grad():
        for _ in trange(length,ncols=80):
            inputs = {"input_ids": generated[0][-(n_ctx - 1):].unsqueeze(0)}  # 为什么 -1 ???
            outputs = model(
                **inputs
            )  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :]  # 此处outputs的输出结果为一个元组，第0个元素为GPT2最后一层经过分类层（仅一个线性层）后的结果
            # 即outputs[0]的形状为[batch_size, seq_len, vocab_size], 此时next_token_logits 的形状为[vocab_size,]
            for id in set(generated):
                # 不知道为什么要用集合，下面的写法不需要去重。况且目前的写法也不能去重，不能用于tensor中元素的去重
                # generated的形状为[1,seq_len],set(generated)后的形状为[seq_len]
                next_token_logits[id] /= repitition_penalty  # repitition_penalty > 1.
                # 目的是对于next_token_logits来说，尽可能使得已经出现在generated中的结果，下一次减少出现
                # 例如generated = tensor([27,68,77,89]), 则 next_token_logits[id] /= repitition_penalty  将使得
                # next_token_logits中27,68,77,89这4个位置上的值变小，进而预测结果再次为这4个值的情况减小
            next_token_logits = next_token_logits / temperature  # 调整分布 , shape: [vocab_size,]
            next_token_logits[tokenizer.convert_tokens_to_ids("[UNK]")] = -float("Inf")  # 将UNK设定为负无穷，忽略它的结果
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            # 从多项分布中抽取num_samples个样本，即解码是按照采样策略进行，而不是贪婪解码
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)  # shape: [1, seq_len + 1]
    return generated.tolist()[0]  # 返回生成后的结果


def fast_sample_sequence(
        model, context, length, temperature=1.0, top_k=30, top_p=0.0, device="cpu"
):
    inputs = torch.LongTensor(context).view(1, -1).to(device)
    if len(context) > 1:
        _, past = model(inputs[:, :-1], None)[:2]
        prev = inputs[:, -1].view(1, -1)
    else:
        past = None
        prev = inputs
    generate = [] + context
    with torch.no_grad():
        for i in trange(length):
            output = model(prev, past=past)
            output, past = output[:2]
            output = output[-1].squeeze(0) / temperature
            filtered_logits = top_k_top_p_filtering(output, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(
                torch.softmax(filtered_logits, dim=-1), num_samples=1
            )
            generate.append(next_token.item())
            prev = next_token.view(1, 1)
    return generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0", type=str, required=False, help="指定生成设备")
    parser.add_argument("--length", default=512, type=int, required=False, help="指定生成序列的长度")
    parser.add_argument("--n_ctx", default=1024, type=int, required=False, help="生成时考虑的上下文长度")
    parser.add_argument("--batch_size", default=1, type=int, required=False, help="生成的batch size")
    parser.add_argument("--nsamples", default=10, type=int, required=False, help="生成几个样本")
    parser.add_argument("--temperature", default=1, type=float, required=False, help="生成温度")
    parser.add_argument("--topk", default=8, type=int, required=False, help="最高几选一")
    parser.add_argument("--topp", default=0, type=float, required=False, help="最高积累概率")
    parser.add_argument("--model_config", default="model/config.json", type=str, required=False,
                        help="模型参数路径")
    parser.add_argument("--tokenizer_path", default="model/vocab.txt", type=str, required=False, help="词表路径")
    parser.add_argument("--model_path", default="model/pytorch_model.bin", type=str, required=False, help="模型路径")
    parser.add_argument("--prefix", default="先帝创业未半而中道崩殂", type=str, required=False, help="生成文章的开头")
    parser.add_argument("--no_wordpiece", action="store_true", help="不做word piece切词")
    parser.add_argument("--segment", action="store_true", help="中文以词为单位")
    parser.add_argument("--fast_pattern", action="store_true", help="采用更加快的方式生成文本")
    parser.add_argument("--save_samples", action="store_true", help="保存产生的样本")
    parser.add_argument("--save_samples_path", default=".", type=str, required=False, help="保存样本的路径")
    parser.add_argument("--repetition_penalty", default=1.0, type=float, required=False)
    args = parser.parse_args()
    print("args:\n" + args.__repr__())
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
    # 例如 os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"，这将允许程序在设备索引为0和1的两个GPU上运行。
    length = args.length  # 指定生成序列的长度
    n_ctx = args.n_ctx  # 生成序列时考虑的上下文长度
    batch_size = args.batch_size
    nsamples = args.nsamples  # 指定根据同一个 提示文本 所生成样本的数量， 每个样本都会不一样
    temperature = args.temperature  # 生成温度， 控制着玻尔兹曼分布中的随机性。
    # 温度较低会导致生成的内容更加确定和缺乏随机性。当温度接近零时，模型将变得越来越确定和重复。而温度较高则会导致生成的内容更加随机。
    topk = args.topk
    topp = args.topp
    repetition_penalty = args.repetition_penalty
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer(vocab_file=args.tokenizer_path)  # 实例化得到一个BERT tokenizer， 分字
    model_config = GPT2Config.from_json_file(args.model_config)  # 实例话一个GPT2配置类
    model = GPT2LMHeadModel(config=model_config)
    # 实例化一个GPT2语言模型，后续我们将要用到GPT2最后一层所有时刻的输出预测结果，然后再取最后一个时刻的预测结果得到下一时刻的预测值

    state_dict = torch.load(args.model_path, map_location="cpu")  # 载入模型
    if 'state_dict' in state_dict:  # 因为有的模型不止保存了state_dict，可能还有优化器的参数等，此时就是一个嵌套的字典
        state_dict = {key[6:]: value for key, value in state_dict["state_dict"].items()}
    model.load_state_dict(state_dict)  # 用本地模型参数重新初始化模型
    model.to(device)
    model.eval()

    for i in range(nsamples):  # 循环生成每个序列
        raw_text = args.prefix  # 生成序列的开始，由用户自定义输入，模型根据这部分输入生成后续结果
        encoded = tokenizer(raw_text)["input_ids"][:-1]
        # 取中文转换成词表索引后的id（因为还包括attention_mask等内容），并且忽略第1个，因为默认BertTokenizer第一个是[CLS]对应的id
        out = sample_sequence(model, encoded, length=length, n_ctx=n_ctx, tokenizer=tokenizer, temperature=temperature,
                              top_k=topk, top_p=topp, repitition_penalty=repetition_penalty, device=device)  # 开始生成每个样本
        print(tokenizer.decode(out))  # 解码并输出结果，将索引id转为 汉字


if __name__ == "__main__":
    main()
