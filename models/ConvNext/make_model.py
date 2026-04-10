import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from timm.models import create_model
from .backbones.model_convnext import convnext_tiny
from .backbones.resnet import Resnet
import numpy as np
from torch.nn import init
from torch.nn.parameter import Parameter


class Gem_heat(nn.Module):
    def __init__(self, dim=768, p=3, eps=1e-6):
        super(Gem_heat, self).__init__()
        self.p = nn.Parameter(torch.ones(dim) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3):
        p = F.softmax(p).unsqueeze(-1)
        x = torch.matmul(x, p)
        x = x.view(x.size(0), x.size(1))
        return x


def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)


class BasicConv_For_DSAB(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, relu=True, bn=True, bias=False):
        super(BasicConv_For_DSAB, self).__init__()
        self.out_channels = out_planes

        self.conv = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)

        return x


class StdPool(nn.Module):
    def __init__(self):
        super(StdPool, self).__init__()

    def forward(self, x):
        std = x.std(dim=1, keepdim=True)
        return std


class ADPool(nn.Module):
    def __init__(self):
        super(ADPool, self).__init__()
        self.std_pool = StdPool()
        self.weight = nn.Parameter(torch.rand(2))

    def forward(self, x):
        std_pool = self.std_pool(x)
        # max_pool = torch.max(x,1)[0].unsqueeze(1)
        avg_pool = torch.mean(x, 1).unsqueeze(1)
        weight = torch.sigmoid(self.weight)
        out = 1 / 2 * (std_pool + avg_pool) + weight[0] * std_pool + weight[1] * avg_pool
        return out


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ADPool()
        self.conv = BasicConv_For_DSAB(1, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class LSK_AttentionGate_Strip(nn.Module):
    def __init__(self, k_list=[5, 7], d_list=[1, 3]):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 1, kernel_size=k_list[0], padding=(k_list[0]-1)//2, bias=False)
        self.conv1 = nn.Conv2d(1, 1, kernel_size=k_list[1], padding=d_list[1]*(k_list[1]-1)//2, dilation=d_list[1], bias=False)
        self.conv_select = nn.Conv2d(2, 2, kernel_size=1) 
    def forward(self, x):
        # x: [B, 1, H, W]
        u1 = self.conv0(x)
        u2 = self.conv1(u1)
        u_concat = torch.cat([u1, u2], dim=1)
        sa_avg = torch.mean(u_concat, dim=1, keepdim=True)
        sa_max, _ = torch.max(u_concat, dim=1, keepdim=True)
        attn_map = torch.sigmoid(self.conv_select(torch.cat([sa_avg, sa_max], dim=1)))
        res = u1 * attn_map[:, 0:1, ...] + u2 * attn_map[:, 1:2, ...]
        return torch.sigmoid(res)

class DSAB_block(nn.Module):
    def __init__(self, in_planes=1024):
        super(DSAB_block, self).__init__()
        # 四个方向的门控
        self.gate_h = LSK_AttentionGate_Strip() # 水平
        self.gate_v = LSK_AttentionGate_Strip() # 垂直
        self.gate_d = LSK_AttentionGate_Strip() # 对角线
        self.gate_a = LSK_AttentionGate_Strip() # 反对角线
        # 融合偏置
        self.fusion_bias = nn.Parameter(torch.zeros(1))
    def _project_diag_to_2d(self, diag_tensor, B, C, H, W, mode='diag'):
        spatial_mask = torch.zeros((B, 1, H, W), device=diag_tensor.device, dtype=diag_tensor.dtype)
        L = diag_tensor.shape[-1]
        # 简单的索引保护，防止尺寸不匹配
        limit = min(H, W, L)
        if mode == 'diag':
            indices = torch.arange(limit, device=diag_tensor.device)
            spatial_mask[:, :, indices, indices] = diag_tensor[:, :, :, :limit].squeeze(2)
        elif mode == 'anti':
            indices_h = torch.arange(limit, device=diag_tensor.device)
            indices_w = torch.arange(limit, device=diag_tensor.device).flip(0)
            spatial_mask[:, :, indices_h, indices_w] = diag_tensor[:, :, :, :limit].squeeze(2)
        return spatial_mask

    def forward(self, x):
        B, C, H, W = x.shape
        L_diag = min(H, W)
        x_h_spatial = x.mean(dim=3, keepdim=True).mean(dim=1, keepdim=True) # [B, 1, H, 1]
        attn_h = self.gate_h(x_h_spatial) # [B, 1, H, 1]
        x_v_spatial = x.mean(dim=2, keepdim=True).mean(dim=1, keepdim=True) # [B, 1, 1, W]
        attn_v = self.gate_v(x_v_spatial) # [B, 1, 1, W]
        raw_diag = torch.diagonal(x, dim1=2, dim2=3) 
        x_d_spatial = raw_diag.mean(dim=1, keepdim=True).unsqueeze(2) # [B, 1, 1, L]
        attn_d = self.gate_d(x_d_spatial) # [B, 1, 1, L]
        raw_anti = torch.diagonal(x.flip(3), dim1=2, dim2=3) 
        x_a_spatial = raw_anti.mean(dim=1, keepdim=True).unsqueeze(2) # [B, 1, 1, L]
        attn_a = self.gate_a(x_a_spatial) # [B, 1, 1, L]
        map_d = self._project_diag_to_2d(attn_d, B, 1, H, W, mode='diag')
        map_a = self._project_diag_to_2d(attn_a, B, 1, H, W, mode='anti')
        diag_context = map_d + map_a # [B, 1, H, W]
        out_h = x * attn_h * (1 + self.fusion_bias * diag_context)
        out_v = x * attn_v * (1 + self.fusion_bias * diag_context)
        return out_h, out_v
class SemanticAdapter(nn.Module):
    def __init__(self, dim):
        super(SemanticAdapter, self).__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.linear2 = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim) 

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        residual = x_flat
        x_proj = self.linear1(x_flat)
        x_norm = self.ln(x_proj)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x_out = residual + attn_out
        x_out = self.linear2(x_out).transpose(1, 2)
        x_out = self.bn(x_out)
        x_out = x_out.view(b, c, h, w)
        return x_out

class MultiHeadSpatialGate(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiHeadSpatialGate, self).__init__()
        self.head1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.head2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.head3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, bias=False)
        self.bn_fusion = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        h1 = self.head1(x)
        h2 = self.head2(x)
        h3 = self.head3(x)
        combined = torch.cat([h1, h2, h3], dim=1)
        out = self.fusion(combined)
        out = self.bn_fusion(out)
        return F.relu(out, inplace=True) 

class SSMA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SSMA, self).__init__()
        self.semantic_adapter = SemanticAdapter(in_channels)
        self.multiscale_gate = MultiHeadSpatialGate(in_channels, out_channels)

    def forward(self, x):
        x_refined = self.semantic_adapter(x)
        x_att = self.multiscale_gate(x_refined)
        return x_att

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True,
                 return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.training:
            if self.return_f:
                f = x
                x = self.classifier(x)
                return x, f
            else:
                x = self.classifier(x)
                return x
        else:
            return x


class Attentions(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(Attentions, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B,
                                                                                                          -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        feature_matrix_raw = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + 1e-6)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix_raw, dim=-1)

        if self.training:
            fake_att = torch.zeros_like(attentions).uniform_(0, 2)
        else:
            fake_att = torch.ones_like(attentions)
        counterfactual_feature = (torch.einsum('imjk,injk->imn', (fake_att, features)) / float(H * W)).view(B, -1)

        counterfactual_feature = torch.sign(counterfactual_feature) * torch.sqrt(
            torch.abs(counterfactual_feature) + 1e-6)

        counterfactual_feature = F.normalize(counterfactual_feature, dim=-1)
        return feature_matrix, counterfactual_feature


class CIB_block(nn.Module):
    def __init__(self, in_planes, block=4, M=32):
        super(CIB_block, self).__init__()
        self.in_planes = in_planes
        self.block = block
        self.M = M
        self.bap = BAP(pool='GAP')
        self.attentions = SSMA(self.in_planes, self.M)

    def forward(self, x):
        part = {}
        part_attention_maps = {}
        part_normal_feature = {}
        part_counterfactual_feature = {}
        for i in range(self.block):
            # part[i] = x[:, :, i].view(x.size(0), -1)
            part[i] = x[:, :, :, :, i]
            part_attention_maps[i] = self.attentions(part[i])  # attention_maps[8,32,8,8]
            part_normal_feature[i], part_counterfactual_feature[i] = self.bap(part[i], part_attention_maps[i])
        return part_normal_feature, part_counterfactual_feature


class build_MDS(nn.Module):
    def __init__(self, num_classes, block=4, M=32, return_f=False, resnet=False):
        super(build_MDS, self).__init__()
        self.return_f = return_f
        if resnet:
            resnet_name = "resnet50"
            print('using model_type: {} as a backbone'.format(resnet_name))
            self.in_planes = 2048
            self.backbone = Resnet(pretrained=True)
        else:
            convnext_name = "convnext_base"
            print('using model_type: {} as a backbone'.format(convnext_name))
            if 'base' in convnext_name:
                self.in_planes = 1024
            elif 'large' in convnext_name:
                self.in_planes = 1536
            elif 'xlarge' in convnext_name:
                self.in_planes = 2048
            else:
                self.in_planes = 768
            # 直接导入convnext_base函数
            from .backbones.model_convnext import convnext_base
            self.backbone = convnext_base(pretrained=True)

        self.num_classes = num_classes
        self.block = block
        self.M = M
        self.DSAB_layer = DSAB_block()
        self.CIB_layer = CIB_block(self.in_planes, self.block, self.M)
        self.classifier1 = ClassBlock(self.in_planes, num_classes, 0.5, return_f=return_f)
        for i in range(self.block * 2):
            name = 'classifier_mcb' + str(i + 1)
            setattr(self, name, ClassBlock(self.in_planes * self.M, num_classes, 0.5, return_f=self.return_f))

    def forward(self, x):
        gap_feature, part_features = self.backbone(x)
        DSAB_features = self.DSAB_layer(part_features)
        convnext_feature = self.classifier1(gap_feature)
        # DSAB_features 是 (out_h, out_v) 元组
        DSAB_list = list(DSAB_features)
        # 确保 DSAB_list 长度至少为 block
        while len(DSAB_list) < self.block:
            # 重复最后一个元素
            DSAB_list.append(DSAB_list[-1])
        # 只取前 block 个元素
        DSAB_list = DSAB_list[:self.block]
        DSAB_attention_features = torch.stack(DSAB_list, dim=4)
        nfeature, cfeature = self.CIB_layer(DSAB_attention_features)

        if self.block == 0:
            y = []
        elif self.training:
            y, y_counterfactual = self.part_classifier(self.block, nfeature, cfeature, cls_name='classifier_mcb')
        else:
            y = self.part_classifier(self.block, nfeature, cfeature, cls_name='classifier_mcb')

        if self.training:
            y = y + [convnext_feature]
            y = y + y_counterfactual
            if self.return_f:
                cls, features = [], []
                for i in y:
                    cls.append(i[0])
                    features.append(i[1])
                return cls, features
        else:
            ffeature = convnext_feature.view(convnext_feature.size(0), -1, 1)
            y = torch.cat([y, ffeature], dim=2)
        return y

    def part_classifier(self, block, nf, cf, cls_name='classifier_mcb'):
        predict_normal = {}
        predict_counterfactual = {}
        for i in range(block):
            name_normal = cls_name + str(i + 1)
            c_normal = getattr(self, name_normal)
            predict_normal[i] = c_normal(nf[i])

            name_counterfactual = cls_name + str(i + 1 + block)
            c_name_counterfactual = getattr(self, name_counterfactual)
            counterfactual_classifier = c_name_counterfactual(cf[i])
            
            # 处理返回格式，确保有两个元素
            if isinstance(predict_normal[i], tuple) and len(predict_normal[i]) == 2:
                pred_cls, pred_feat = predict_normal[i]
            else:
                pred_cls = predict_normal[i]
                pred_feat = torch.zeros_like(pred_cls)
            
            if isinstance(counterfactual_classifier, tuple) and len(counterfactual_classifier) == 2:
                cf_cls, cf_feat = counterfactual_classifier
            else:
                cf_cls = counterfactual_classifier
                cf_feat = torch.zeros_like(cf_cls)
            
            predict_counterfactual[i] = (pred_cls - cf_cls, pred_feat - cf_feat)
        
        y_normal = []
        y_counterfactual = []
        for i in range(block):
            y_normal.append(predict_normal[i])
            y_counterfactual.append(predict_counterfactual[i])
        
        if not self.training:
            # 只取分类结果
            y_normal_cls = []
            for pred in y_normal:
                if isinstance(pred, tuple) and len(pred) == 2:
                    y_normal_cls.append(pred[0])
                else:
                    y_normal_cls.append(pred)
            return torch.stack(y_normal_cls, dim=2)
        
        return y_normal, y_counterfactual


def make_MDS_model(num_class, block=4, M=32, return_f=False, resnet=False):
    print('===========building convnext===========')
    model = build_MDS(num_class, block=block, M=M, return_f=return_f, resnet=resnet)
    return model