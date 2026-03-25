# coding=utf-8

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_main_logits(logits: torch.Tensor, labels: torch.Tensor):
    mask = labels != -100
    if int(mask.sum().item()) == 0:
        return logits.new_zeros((0, logits.shape[-1]))
    return logits[mask]


def kl_div_with_temperature(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float):
    if student_logits.numel() == 0 or teacher_logits.numel() == 0:
        return student_logits.new_tensor(0.0)
    temp = float(temperature)
    student_log_probs = F.log_softmax(student_logits.float() / temp, dim=-1)
    teacher_probs = F.softmax(teacher_logits.float() / temp, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temp * temp)


def compose_total_loss(main_loss, sub_loss, main_kd_loss, sub_kd_loss, *, main_kd_weight: float, sub_kd_weight: float):
    return main_loss + 0.3 * sub_loss + float(main_kd_weight) * main_kd_loss + float(sub_kd_weight) * sub_kd_loss

