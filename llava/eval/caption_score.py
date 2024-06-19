from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge

def score(ref, sample):
    # ref and sample are both dict
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Cider(), "CIDEr"),
        (Rouge(), "ROUGE-L")
    ]
    final_scores = {}
    for scorer, method in scorers:
        # print('computing %s score with COCO-EVAL...' % (scorer.method()))
        score, scores = scorer.compute_score(ref, sample)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores 


if __name__ == '__main__':
    # 加载参考文本
    references = {
        "image1": ["there is a cat on the mat."],
        # "image2": ["A dog is running in the park."]
    }

    # 加载生成文本
    hypotheses = {
        "image1": ["the cat is on the mat."],
        # "image2": ["A dog is playing in the field."]
    }

    print(score(references, hypotheses))
 
# # 创建Bleu对象
# bleu_scorer = Bleu(n=4)
# cider_scorer = Cider()
 
# # 计算BLEU分数
# bleu_scores, _ = bleu_scorer.compute_score(references, hypotheses)
# cider_scores, _ = cider_scorer.compute_score(references, hypotheses)
# # 打印BLEU分数
# print("BLEU scores:", bleu_scores)
# print("Cider scores:", cider_scores)




