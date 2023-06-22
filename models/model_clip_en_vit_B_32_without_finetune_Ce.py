import torch
import clip
import os
from PIL import Image
import numpy as np
from models.transformer import build_transformer
from transformers import AutoTokenizer, AutoFeatureExtractor,BertModel
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from models.pos_emb import AbsolutePositionalEncoding
from tqdm import *
import copy
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
torch.set_printoptions(profile="full")
device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("/home/light/.cache/clip/ViT-B-32.pt", device=device)
model, preprocess = clip.load("ViT-B/32", device=device)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)]).to(device)


class LineNormLayers(torch.nn.Module):
    def __init__(self, in_model, out_model):
        super(LineNormLayers, self).__init__()
        self.liner1 = nn.Linear(in_model, out_model).to(device)
        self.norm1 = nn.LayerNorm(out_model).to(device)
        # self.liner2 = nn.Linear(in_model * 2, out_model).to(device)
        # self.norm2 = nn.LayerNorm(out_model).to(device)

    def forward(self, x):
        x = self.liner1(x)
        x = self.norm1(x)
        # x = self.liner2(x)
        # x = self.norm2(x)
        return x


class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.d3_emb = nn.Embedding(num_embeddings=201, embedding_dim=128).to(device)
        self.transformer = build_transformer(args)
        # self.word_emb = nn.Embedding(num_embeddings=30000, embedding_dim=args.hidden_dim).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            'bert-base-uncased')
        self.bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
        for param in self.bert_model.parameters():
                param.requires_grad = False
        self.d3_pos_emb = AbsolutePositionalEncoding(128, 3)
        # self.image_pos_emb=AbsolutePositionalEncoding(512, args.max_length)
        self.input_encoder_pos_emb = AbsolutePositionalEncoding(args.hidden_dim, args.max_length)
        self.input_decoder_pos_emb = AbsolutePositionalEncoding(args.hidden_dim, args.max_words_length)
        self.classify = nn.Linear(args.hidden_dim, args.cls_num).to(device)
        self.word2idx = [i / 100 for i in range(-100, 101)]
        self.line1 = LineNormLayers(512, 384)
        self.line2 = LineNormLayers(384,384)
        self.line3 = LineNormLayers(512, 384)

    def forward(self, batch_image_patches, d3_patches, question):
        # ------------------------------------------------------- #
        # image_qurey_paded shape=(batch_size,max_length,1024)
        # ------------------------------------------------------- #
        image_embedding_paded, src_padding_mask = self.image_encoding(batch_image_patches)
        batch = len(image_embedding_paded)
        # image_pos_embedding=self.image_pos_emb(batch_size,self.args.max_length)
        # image_embedding_paded+=image_pos_embedding

        # ------------------------------------------------------- #
        # d3_embedding_paded shape=(batch_size,max_length,3,128)
        # ------------------------------------------------------- #
        d3_embedding_paded = self.d3_encoding(d3_patches)

        # ------------------------------------------------------- #
        # d3_pos_embedding shape=(max_length,3,128)
        # ------------------------------------------------------- #
        d3_pos_embedding = self.d3_pos_emb(self.args.max_length, 3)
        d3_embedding_paded += d3_pos_embedding
        # d3_embedding_paded shape=(batch_size,max_length,3*128)
        d3_embedding_paded = d3_embedding_paded.flatten(2)
        image_embedding_paded = self.line1(image_embedding_paded).to(device)
        d3_embedding_paded = self.line2(d3_embedding_paded).to(device)
        # src_encoder_embedding = torch.cat((image_embedding_paded, d3_embedding_paded), dim=-1)
        # src_embedding shape=(barch_size,max_length,512)
        src_embedding = image_embedding_paded + d3_embedding_paded
        # src_pos_embedding shape=(batch_size,max_length,512)
        src_pos_embedding = self.input_encoder_pos_emb(batch, self.args.max_length)
        tgt_pos_embedding = self.input_decoder_pos_emb(batch, self.args.max_words_length)
        # ---------------------------------------------------- #
        # 对question进行编码 shape=(n,max_length,512)
        # attention_mask pad为1
        # ---------------------------------------------------- #
        # tgt_tokenized_text = self.tokenizer(question, add_special_tokens=True, max_length=self.args.max_words_length,
        #                                     padding='max_length',return_tensors="pt").to(device)
        # # tgt_tokens_id = torch.tensor(tgt_tokenized_text['input_ids']).to(device)
        # tgt_attention_mask = 1.0 - tgt_tokenized_text['attention_mask'].clone().detach().to(device)
        # # tgt_embedding = self.word_emb(tgt_tokens_id)
        # outputs=self.bert_model(**tgt_tokenized_text)
        text = clip.tokenize(question).to(device)
        outputs=model.encode_text(text)
        tgt_embedding=torch.as_tensor(torch.unsqueeze(outputs,dim=1).detach().cpu().numpy(),dtype=torch.float32).to(device)
        tgt_embedding=self.line3(tgt_embedding).to(device)
        # print('attention_mask shape',attention_mask.shape)
        # print('tgt_embedding shape',tgt_embedding.shape)
        hs = self.transformer(src_embedding.permute(1, 0, 2), src_padding_mask, None,
                              tgt_embedding.permute(1, 0, 2), src_pos_embedding.permute(1, 0, 2),
                              tgt_pos_embedding.permute(1, 0, 2))
        output = self.classify(hs.permute(1, 0, 2)[:, 0])
        return output

    def d3_encoding(self, batch_d3_patches):
        batch_d3_feature = []
        for d3_patches in batch_d3_patches:
            d3_lists = []
            for d in d3_patches:
                d3_feature = self.d3_emb(
                    torch.tensor(np.array([self.word2idx.index(round(float(i), 2)) for i in d])).to(device)).to(device)
                d3_lists.append(d3_feature)
            d3_arrs_tensor = torch.stack(d3_lists).to(device)
            d3_arrs_tensor_paded = self.d3_padding(self.args.max_length, d3_arrs_tensor)
            batch_d3_feature.append(d3_arrs_tensor_paded)
        batch_d3_feature = torch.stack(batch_d3_feature).to(device)
        return batch_d3_feature

    def image_encoding(self, batch_image_patches):
        """

        Args:
            batch_image_patches: shape=(batch,n,768)

        Returns:
            batch_image_feature: shape=(batch,20,768) dim=clip提取的特征维度

        """
        batches_features_list = []
        mask_list = []
        for image_patches in batch_image_patches:
            patches_list = []
            for image in image_patches:
                patches_list.append(torch.tensor(image.astype(float), dtype=torch.float32).to(device))
            patches_tensor = torch.stack(patches_list).to(device)
            # print("是否更新梯度:",patches_tensor.requires_grad)
            patches_features_paded, padding_mask = self.patches_padding(self.args.max_length, patches_tensor)
            batches_features_list.append(patches_features_paded)
            mask_list.append(padding_mask)
        batches_features = torch.stack(batches_features_list).to(device)
        batch_mask = torch.stack(mask_list).to(device)
        return batches_features, batch_mask

    def patches_padding(self, max_length, image_query):
        """

        Args:
            max_length: 最大长度
            batch_image_qurey: 没有补齐的状态

        Returns:
            batch_image_qurey_paded: 补齐之后的序列，现在就可以当做是一个nlp来处理 shape=(b,l,dim)
            batch_image_qurey_padding_mask: 补齐之后的mask,shape=(b,l)
        """

        image_qurey_paded = torch.concatenate(
            (image_query, torch.zeros([(max_length - image_query.shape[0]), 512]).to(device)), dim=0)
        padding_mask = torch.tensor(
            np.array([0] * image_query.shape[0] + [1] * (max_length - image_query.shape[0])),
            dtype=torch.float32).to(device)

        return image_qurey_paded, padding_mask

    def d3_padding(self, max_length, d3_embedding):
        d3_embedding_paded = torch.concatenate(
            (d3_embedding, torch.zeros([(max_length - d3_embedding.shape[0]), 3, 128]).to(device)), dim=0)

        return d3_embedding_paded

    def key_padding_mask(self, tokenized_text):
        mask_tensor = 1.0 - torch.tensor(tokenized_text['attention_mask'])
        return mask_tensor
