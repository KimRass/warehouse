# References
    # https://baeseongsu.github.io/posts/knowledge-distillation/#q3-knowledge-distillation%EC%9D%80-%EC%96%B4%EB%96%BB%EA%B2%8C-%ED%95%98%EB%8A%94-%EA%B1%B8%EA%B9%8C-with-hintons-kd
    # https://github.com/haitongli/knowledge-distillation-pytorch/blob/master/model/net.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# nn.KLDivLoss()(F.log_softmax(student_logits / T, dim=1), F.softmax(teacher_logits / T, dim=1)) * (alpha * T * T) + F.cross_entropy(logits, labels) * (1. - alpha)
def get_knowledge_distillation_loss(student_logits, teacher_logits, labels, temperature, lamb):
    loss_ce = F.cross_entropy(F.softmax(student_logits, dim=1), F.softmax(labels, dim=1))
    loss_kd_ce = F.cross_entropy(
        F.softmax(student_logits / temperature, dim=1), F.softmax(teacher_logits / temperature, dim=1)
    )
    total_loss = lamb * loss_ce + loss_kd_ce
    return total_loss



if __name__ == "__main__":
    logits = torch.randn((1, 100)) * 10e2

    softmax = F.softmax(logits)
    plt.plot(softmax.numpy().tolist()[0])
    plt.show()

    T = 20e1
    softmax = F.softmax(logits / T)
    plt.plot(softmax.numpy().tolist()[0])
    plt.show()

    student_logits = torch.randn((1, 100)) * 10e2
    teacher_logits = torch.randn((1, 100)) * 10e2
    labels = torch.randn((1, 100)) * 10e2
    temperature = 20e1
    lamb = 0.5
    get_knowledge_distillation_loss(student_logits, teacher_logits, labels, temperature, lamb)