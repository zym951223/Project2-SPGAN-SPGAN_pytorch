import argparse
import itertools
import os
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import torchvision.utils as vutils
import models.SPGAN
import Utils.util, Utils.visualizer
from dataset.Dataset import dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", required=True, help="the dataset root")
parser.add_argument("--loadsize", type=int, default=286, help="the img load size, default is 286")
parser.add_argument("--cropsize", type=int, default=256, help="the img crop size after load, default is 256")
parser.add_argument("--batchsize", type=int, default=1, help="the batch size, default is 1")
parser.add_argument("--lr", type=float, default="0.0002", help="the initial learning rate, default is 0.002")
parser.add_argument("--epoch", type=int, default=6, help="the number of epoch")
parser.add_argument("--cuda",  default=False, action="store_true", help="use cuda")
parser.add_argument("--workers", type=int, default=2, help="number of data loading workers, default is 2")

opt = parser.parse_args()

# ensure there are two folders in dataroot
assert len(os.listdir(opt.dataroot)) == 2

# create two dataloader for both of the datasets
A_dataset_path, B_dataset_path = os.listdir(opt.dataroot)
A_dataset_path = os.path.join(opt.dataroot, A_dataset_path)
B_dataset_path = os.path.join(opt.dataroot, B_dataset_path)

Dataset = dataset(A_dataset_path, B_dataset_path, opt)

Dataloader = torch.utils.data.DataLoader(
    Dataset, batch_size=opt.batchsize, shuffle=True)

# allocate the network to specified device
if opt.cuda and torch.cuda.is_available():
    G = models.SPGAN.generator().cuda()
    F = models.SPGAN.generator().cuda()
    D_G = models.SPGAN.discrimnator().cuda()
    D_F = models.SPGAN.discrimnator().cuda()
    M = models.SPGAN.metric_net().cuda()
else:
    G = models.SPGAN.generator()
    F = models.SPGAN.generator()
    D_G = models.SPGAN.discrimnator()
    D_F = models.SPGAN.discrimnator()
    M = models.SPGAN.metric_net()

# define the optimizers
optimizer_G = optim.Adam(itertools.chain(G.parameters(), F.parameters()),
    lr=opt.lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(itertools.chain(D_G.parameters(), D_F.parameters()), 
    lr=opt.lr, betas=(0.5, 0.999))
optimizer_M = optim.Adam(M.parameters(),
    lr=opt.lr, betas=(0.5, 0.999))

# define the loss functions
loss_L2 = nn.MSELoss()
loss_L1 = nn.L1Loss()

# set the parameters
lambda1 = 10.0
lambda2 = 5.0
lambda3= 2.0
margin = 2.0

# create the image pool
fake_A_pool = Utils.util.ImagePool(50)
fake_B_pool = Utils.util.ImagePool(50)

# begining training!
#Combined_Dataset = zip(A_Dataloader, B_Dataloader)
ite_num = len(Dataset)
label_real = torch.ones(1, 1, 30, 30).cuda()
label_fake = torch.zeros(1, 1, 30, 30).cuda()

# set visualzing
root = os.path.join("checkpoints", "vehicle")
if not os.path.exists(root):
    os.mkdir(root)
vis = Utils.visualizer.Visualizer(root)

for epoch_num in range(opt.epoch):
    D_LOSS = []
    CYCLE_LOSS = []
    IDT_LOSS = []
    M_LOSS = []
    visuals = OrderedDict()
    errors = OrderedDict()
    for ite, data in enumerate(Dataloader):
        if ite > 100:
            break
        # allocate the tensors to specified device
        if opt.cuda and torch.cuda.is_available():
            a_real = data[0].cuda()
            b_real = data[1].cuda()
        else:
            a_real = data[0]
            b_real = data[1]
            
        # nodes
        a2b = G(a_real)
        b2a = F(b_real)
        b2a2b = G(b2a)
        a2b2a = F(a2b)

        a2a = F(a_real)
        b2b = G(b_real)

        # siamese network
        a_metric = func.normalize(M(a_real.detach()))
        b_metric = func.normalize(M(b_real.detach()))
        a2b_metric = func.normalize(M(a2b.detach()))
        b2a_metric = func.normalize(M(b2a.detach()))

        # positive pair
        S_metric_pos = loss_L2(a_metric, a2b_metric)

        T_metric_pos = loss_L2(b_metric, b2a_metric)

        # negative pair
        neg = loss_L2(a_metric, b_metric)
        neg = neg + 1e-6
        neg = torch.sqrt(neg)
        NEG = torch.pow((max(margin-neg, torch.tensor(0, dtype=torch.float32).cuda())), 2)

        # contrastive loss
        m_loss = (T_metric_pos + S_metric_pos + 2*NEG)/3.0 * lambda3
        M_LOSS.append(m_loss.item())

        # losses
        a2b_dis = D_G(a2b)
        b2a_dis = D_F(b2a)
        g_loss_a2b = loss_L2(a2b_dis, label_real)
        g_loss_b2a = loss_L2(b2a_dis, label_real)
        #g_orig = g_loss_a2b + g_loss_b2a
        cyc_loss_a = loss_L1(a_real, a2b2a)  
        cyc_loss_b = loss_L1(b_real, b2a2b)
        cyc_loss = (cyc_loss_a + cyc_loss_b) * lambda1
        CYCLE_LOSS.append((cyc_loss_a.item(), cyc_loss_b.item()))

        # identity loss
        idt_loss_a = loss_L1(a_real, a2a)
        idt_loss_b = loss_L1(b_real, b2b)
        idt_loss = (idt_loss_a + idt_loss_b) * lambda2
        IDT_LOSS.append((idt_loss_a.item(), idt_loss_b.item()))
        
        g_loss = g_loss_a2b + g_loss_b2a + cyc_loss + idt_loss + m_loss

        ########################
        # Optimizing G,F and M #
        ########################
        optimizer_G.zero_grad()
        optimizer_M.zero_grad()
        g_loss.backward()

        optimizer_G.step()
        optimizer_M.step()

        ##########################
        # Optimizing D_G and D_F #
        ##########################
        optimizer_D.zero_grad()

        a2b = fake_B_pool.query(a2b.detach())
        b2a = fake_A_pool.query(b2a.detach())

        a_dis = D_F(a_real)
        b2a_dis = D_F(b2a)

        b_dis = D_G(b_real)
        a2b_dis = D_G(a2b)

        d_loss_a_real = loss_L2(a_dis, label_real)
        d_loss_b2a_sample = loss_L2(b2a_dis, label_fake)
        d_loss_a = (d_loss_a_real + d_loss_b2a_sample) / 2
        d_loss_a.backward()

        d_loss_b_real = loss_L2(b_dis, label_real)
        d_loss_a2b_sample = loss_L2(a2b_dis, label_fake)
        d_loss_b = (d_loss_b_real + d_loss_a2b_sample) / 2
        d_loss_b.backward()

        D_LOSS.append((d_loss_a.item(), d_loss_b.item()))

        optimizer_D.step()

        # display
        if (ite + 1) % 25== 0:
            d_f_loss = sum([i[0] for i in D_LOSS]) / len(D_LOSS)
            d_g_loss = sum([i[1] for i in D_LOSS]) / len(D_LOSS)
            cycle_loss_a = sum([i[0] for i in CYCLE_LOSS]) / len(CYCLE_LOSS)
            cycle_loss_b = sum([i[1] for i in CYCLE_LOSS]) / len(CYCLE_LOSS)
            idt_loss_a = sum([i[0] for i in IDT_LOSS]) / len(IDT_LOSS)
            idt_loss_b = sum([i[1] for i in IDT_LOSS]) / len(IDT_LOSS)
            m_loss = sum(M_LOSS) / len(M_LOSS)

            D_LOSS = []
            CYCLE_LOSS = []
            IDT_LOSS = []
            M_LOSS = []
            visuals["a_real"] = a_real.detach()
            visuals["a2b"] = a2b.detach()
            visuals["a2b2a"] = a2b2a.detach()
            visuals["a2a"] = a2a.detach()           
            visuals["b_real"] = b_real.detach()
            visuals["b2a"] = b2a.detach()
            visuals["b2a2b"] = b2a2b.detach()
            visuals["b2b"] = b2b.detach()

            vis.display_current_results(visuals, epoch_num, True)

            errors["d_f_loss"] = float(d_f_loss)
            errors["d_g_loss"] = float(d_g_loss)
            errors["cycle_loss_a"] = float(cyc_loss_a)
            errors["cycle_loss_b"] = float(cyc_loss_b)
            errors["idt_loss_a"] = float(idt_loss_a)
            errors["idt_loss_b"] = float(idt_loss_b)
            errors["m_loss"] = float(m_loss)

            vis.plot_current_losses(epoch_num, float(ite/ite_num), opt, errors)            

            print("Epoch: [{}/{}] Ite: [{}/{}] D_G_loss : {:.3f} D_F_loss : {:.3f} cycle_loss_a : {:.3f}\
                cycle_loss_b : {:.3f} idt_loss_a : {:.3f} idt_loss_b : {:.3f} m_loss : {:.3f} \
                ".format(epoch_num+1, opt.epoch, ite+1, ite_num,
                d_f_loss, d_g_loss, cyc_loss_a, cyc_loss_b, idt_loss_a, idt_loss_b, m_loss))
            
        if (ite + 1) % 25 == 0:
            img_a = torch.cat((a_real, a2b, a2b2a), 0)
            img_b = torch.cat((b_real, b2a, b2a2b), 0)
            vutils.save_image(img_a, "checkpoints/vehicle/epoch-{}-ite-{}-a.jpg"\
                .format(epoch_num+1, ite+1), normalize=True)
            vutils.save_image(img_b, "checkpoints/vehicle/epoch-{}-ite-{}-b.jpg"\
                .format(epoch_num+1, ite+1), normalize=True)
        
        
        