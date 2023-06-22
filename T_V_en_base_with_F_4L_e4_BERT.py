import torch
import os
from PIL import Image
import numpy as np
from torch.optim import Adam
import argparse
from models.model_vit_en_base_patch16_224_in21k_with_finetune_BERT import Model
from tqdm import *
import time
from utils.utils import find_indexes
from utils.dataloader_vit_base_patch16_224_in21k_with_vit_finetune import dataload
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from itertools import permutations

device = "cuda" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--hidden_dim', default=512)
parser.add_argument('--dropout', default=0.1)
parser.add_argument('--nheads', default=4)
parser.add_argument('--dim_feedforward', default=2048)
parser.add_argument("--enc_layers", default=4)
parser.add_argument("--dec_layers", default=4)
parser.add_argument("--max_length", default=15)
parser.add_argument("--max_words_length", default=25)
parser.add_argument("--cls_num", default=15)
parser.add_argument("--pre_norm", default=True)
parser.add_argument("--epoches", default=300)
args = parser.parse_args()


class TrainVQA(torch.nn.Module):
    def __init__(self):
        super(TrainVQA, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.batch_size = 3072
        self.vqa_model = Model(args)
        # self.vqa_model = torch.nn.DataParallel(self.vqa_model).to(device)
        # self.vqa_model.load_state_dict(torch.load("/hy-tmp/B_e_without_F /pytorch_model_111_loss_0.009874.bin"))
        self.optim = Adam([{'params': self.vqa_model.parameters()},
                           ], lr=1e-4)
        self.eps = 1e-4
        total = sum([param.nelement() for param in self.vqa_model.parameters() if param.requires_grad == True])
        print("可训练参数总量:", total)
        # -------------------------- #
        # T_0: 初始退火周期的长度
        # T_mult: 退火周期长度的倍数
        # eta_min: 学习率的最小值
        # 在每个退火周期结束后，T_0将乘以T_mult，重新计算下一个周期的长度。而epoch参数表示当前训练的epoch数，用于控制warmup的学习率逐渐增加。
        # 在实践中，通常将T_0设置为一个比较小的值，如10或20，将T_mult设置为2或3，将eta_min设置为一个较小的值，如0.0001或0.00001。
        # 需要注意的是，在使用Warmup Cosine Scheduler时，通常需要结合其他技巧一起使用，如梯度累积、权重衰减、dropout等，以提高模型的稳定性和性能。
        # 同时，学习率的设置应根据具体情况进行调整，如调整退火周期长度、学习率的初始值和终止值等。
        # -------------------------- #
        # self.scheduler=CosineAnnealingWarmRestarts(self.optim,T_0=10,T_mult=2,eta_min=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=15, gamma=0.8)

    def forward(self, batch_image_patches, d3_patches, questions, labels):
        train_len = len(labels)
        with open("log_2.txt", mode="w") as w:
            for epoch in range(args.epoches):
                self.vqa_model.train()
                with tqdm(total=train_len // self.batch_size + 1) as _tqdm:  # 使用需要的参数对tqdm进行初始化
                    loss = torch.tensor(.0).to(device)
                    iteration_num = 0
                    for iteration in range(train_len // self.batch_size + 1):
                        if len(batch_image_patches[self.batch_size * iteration:self.batch_size * (iteration + 1)]):
                            logits = self.vqa_model(
                                batch_image_patches[self.batch_size * iteration:self.batch_size * (iteration + 1)],
                                d3_patches[self.batch_size * iteration:self.batch_size * (iteration + 1)],
                                questions[self.batch_size * iteration:self.batch_size * (iteration + 1)])
                            # step_loss = self.criterion(logits+self.eps, labels[self.batch_size * iteration:self.batch_size * (
                            #             iteration + 1)])
                            step_loss = torch.nn.functional.nll_loss(
                                torch.log(torch.nn.functional.softmax(logits, dim=1) + self.eps),
                                labels[self.batch_size * iteration:self.batch_size * (iteration + 1)])
                            loss += step_loss
                            self.optim.zero_grad()
                            step_loss.backward()
                            self.optim.step()
                            # self.scheduler.step(epoch)

                            iteration_num += 1
                            _tqdm.set_description(
                                'epoch:{} iteration: {}/{}'.format(epoch, iteration, train_len // self.batch_size + 1))
                            _tqdm.set_postfix(loss='{:.6f}'.format(step_loss))
                            _tqdm.update(1)
                    self.scheduler.step()
                    loss /= iteration_num
                    _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, args.epoches))
                    _tqdm.set_postfix(loss='{:.6f}'.format(loss))
                    model_to_save = self.vqa_model.module if hasattr(self.vqa_model, 'module') else self.vqa_model
                    torch.save(model_to_save.state_dict(),
                               "/hy-tmp/V_e_B_16_BERT_with_F/pytorch_model_{}_loss_{:.6f}.bin".format(
                                   epoch + 1, loss))
                self.vqa_model.eval()
                question_eval_dict = {}
                label_eval_dict = {}
                with open(question_eval, mode='r') as r:
                    lines = r.readlines()
                    for line in lines:
                        if not line.isspace():
                            split_words = line.rstrip().split('|')
                            ids, question, label = split_words[0], split_words[1], int(split_words[2])
                            if ids not in question_eval_dict.keys():
                                question_eval_dict[ids] = []
                                question_eval_dict[ids].append(question)
                                label_eval_dict[ids] = []
                                label_eval_dict[ids].append(label)
                            else:
                                question_eval_dict[ids].append(question)
                                label_eval_dict[ids].append(label)
                total = 0
                acc = 0
                for key in question_eval_dict.keys():
                    test_question = question_eval_dict[key]
                    len_test_question = len(test_question)
                    test_patch = [all_image_patches_eval[key]] * len_test_question
                    test_d3_patches = [all_d3_patches_eval[key]] * len_test_question
                    logits = self.vqa_model(test_patch,
                                            np.array(test_d3_patches) / np.max(abs(np.array(test_d3_patches))),
                                            test_question)
                    # print("预测的结果为:", torch.argmax(logits, dim=1).detach().cpu().numpy())
                    # print("真实的标签为:", np.array(label_eval_dict[key]))
                    # print(torch.argmax(logits, dim=1).detach().cpu().numpy()==np.array(label_eval_dict[key]))
                    acc += np.sum(torch.argmax(logits, dim=1).detach().cpu().numpy() == np.array(label_eval_dict[key]))
                    total += len_test_question
                print("epoch:{},acc:{}".format(epoch + 1, acc / total))
                with open("log_2.txt", mode="a") as a:
                    a.write('epoch: {}/{} loss={:.6f} acc={:.6f}\n'.format(epoch + 1, args.epoches, loss, acc / total))
            w.close()


if __name__ == "__main__":
    train_vqa = TrainVQA()
    # ----------------------------------------- #
    # 数据加载
    # ----------------------------------------- #
    rootpath = "/hy-tmp/train_0504_checked"
    # rootpath = "/home/light/gree/slam/D3VG/datas/hdj0505"
    all_scene_patches, all_d3_patches = dataload(rootpath)

    eval_rootpath = '/hy-tmp/eval'
    all_image_patches_eval, all_d3_patches_eval = dataload(eval_rootpath)

    # ------------ #
    # 导入标签
    # ------------ #
    associate_path = '/hy-tmp/train_0504_checked/scannet0504_en.txt'
    # associate_path = '/home/light/gree/slam/D3VG/datas/hdj0505/scannet0504_en.txt'
    question_eval = "/hy-tmp/eval/scannet0504_en.txt"
    train_scene_patches = []
    train_d3_patches = []
    questions = []
    labels = []
    questions_dict = {}
    labels_dict = {}
    with open(associate_path, mode='r') as a:
        lines = a.readlines()
        for line in lines:
            if not line.isspace():
                split_words = line.rstrip().split('|')
                # print(split_words)
                ids, question, label = split_words[0], split_words[1], split_words[2]
                if ids not in questions_dict.keys() and ids not in labels_dict.keys():
                    questions_dict[ids] = []
                    labels_dict[ids] = []
                    questions_dict[ids].append(question)
                    labels_dict[ids].append(int(label))
                else:
                    questions_dict[ids].append(question)
                    labels_dict[ids].append(int(label))
    max_len = 0
    for k in questions_dict.keys():
        for q in questions_dict[k]:
            l = len(q.split())
            if l >= max_len:
                max_len = l
    print("句子描述最长长度:", max_len)
    # -------------- #
    # 数据增强
    # -------------- #
    aug_questions = []
    aug_labels = []
    scene_key_names = all_scene_patches.keys()
    for key in scene_key_names:
        a_scene_patches = np.array(all_scene_patches[key], dtype=object)
        d3_patches = np.array(all_d3_patches[key]) / np.max(abs(np.array(all_d3_patches[key])))
        if len(d3_patches) <= 4:
            index = np.arange(len(d3_patches))
            times = len(list(permutations(index)))
        else:
            times = 100
        for i in range(times):
            index = np.arange(len(d3_patches))
            np.random.seed(i)
            np.random.shuffle(index)
            for j in range(len(labels_dict[key])):
                train_scene_patches.append(a_scene_patches[index])
                train_d3_patches.append(d3_patches[index])
            new_labels = find_indexes(labels_dict[key], index)
            aug_labels.extend(new_labels)
            aug_questions.extend(questions_dict[key])
        # print(len(train_scene_patches), len(train_d3_patches), len(aug_questions), len(aug_labels))
    is_train = True
    train_scene_patches = np.array(train_scene_patches, dtype=object)
    train_d3_patches = np.array(train_d3_patches)
    aug_questions = np.array(aug_questions)
    aug_labels = np.array(aug_labels)
    np.random.seed(100)
    np.random.shuffle(train_scene_patches)
    np.random.seed(100)
    np.random.shuffle(train_d3_patches)
    np.random.seed(100)
    np.random.shuffle(aug_questions)
    np.random.seed(100)
    np.random.shuffle(aug_labels)
    aug_questions = aug_questions.tolist()
    aug_labels = torch.tensor(aug_labels).to(device)
    if is_train:
        train_vqa(train_scene_patches, train_d3_patches, aug_questions, aug_labels)
