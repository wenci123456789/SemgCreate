import sys

from Model.EMGMambaAttention import EMGMambaAttenRegressor

sys.path.append("..")
import numpy
import argparse
import os, datetime, time
from tensorboardX import SummaryWriter
import shutil
from utils.Methods.methods import str2bool
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataset import ConcatDataset
from skimage import metrics
import sklearn.metrics as skmetrics
from utils.Methods.methods import pearson_CC, avg_smoothing_np, draw_graph
from DataProcess import NinaPro
# 要训练的个体
# pretrains = ['S1','S2','S3','S5','S9','S12','S14','S15','S16','S24','S26','S30','S31','S33','S34','S35','S38']
pretrains = ['S1','S2','S3']
num_subjects = 2
for pretrain in pretrains:
    # Params
    parser = argparse.ArgumentParser(description='PyTorch sEMG-mamba+attn+adapt')
    # dataset
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--subframe', default=200, type=int)
    parser.add_argument('--epoch', default=200, type=int)
    # parser.add_argument('--epoch', default=1000, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--milestones', default=[100], type=list)
    # parser.add_argument('--milestones', default=[300, 600, 900], type=list)
    parser.add_argument('--device_ids', type=list, default=[0])
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--normalization', type=str, default="miu")
    parser.add_argument('--miu', type=int, default=2 ** 20)
    parser.add_argument('--smooth', type=str2bool, default=False)
    parser.add_argument('--use_se', type=str2bool, default=False)

    parser.add_argument('--subject', type=str, default=[pretrain])
    # [f"S{i+1}" for i in range(40)]
    parser.add_argument('--subject_name', type=str, default=pretrain)
    parser.add_argument('--model', type=str, default="mamba")

    args = parser.parse_args()#获取所有参数

    cuda = torch.cuda.is_available()

    assert cuda

    # subject = args.subject.strip()
    subject = args.subject
    subject_name = args.subject_name.strip()
    model_name = args.model.strip()
    if not isinstance(subject, list):
        subject = [subject]
    if "S0" in subject:
        subject.remove("S0")
        subject = ["S1", "S2", "S3", "S13", "S11", "S14", "S18", "S19"] + subject
        # "S1", "S2", "S3", "S13", "S25", "S11", "S14", "S18", "S19", "S22"

    # 定义了一个自定义的损失函数，用于计算分类任务中的损失
    def cls_new_loss(y_pred, y_label):
        return torch.sum(-y_label * y_pred.log())


    if __name__ == '__main__':
        random.seed(525)
        np.random.seed(seed=525)
        torch.manual_seed(525)

        # model selection
        print('===> Building model')
        model = EMGMambaAttenRegressor(num_subjects=num_subjects);
        initial_epoch = 0
        reg_loss = nn.MSELoss()
        cls_loss = nn.CrossEntropyLoss()
        cur_model_name = model.name
        # 模型可视化的存储路径
        tensorboard_record_path = f"../result/demo/{args.normalization}/{subject_name}/{cur_model_name}"
        # f - string格式化字符串的方式构建了保存模型的路径
        save_dir = f"../result/demo/{args.normalization}/{subject_name}/{cur_model_name}"
        #使用平滑作用的模型前面加s
        if args.smooth:
            cur_model_name = "s" + cur_model_name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if os.path.exists(tensorboard_record_path):
            print("[*]Cleaning previous cache of tensorboard...")
            #删除路径所有内容
            shutil.rmtree(tensorboard_record_path)
        record_writer = SummaryWriter(tensorboard_record_path)
        dummy_input = torch.randn([1, 200, 12]).cuda()
        dummy_subject = torch.zeros(1, dtype=torch.long).cuda()
        print(f"[*]Current model is {cur_model_name}")
        if cuda:
            print("[*]Training on GPU......")
            model = model.cuda()
            # 将add_graph包装在try-except中，避免因可视化问题导致训练中断
            try:
                record_writer.add_graph(model, (dummy_input, dummy_subject))
            except Exception as e:
                print(f"Warning: Failed to add graph to TensorBoard: {e}")
            device_ids = args.device_ids
            reg_loss = reg_loss.cuda()
            # model = nn.DataParallel(model, device_ids=device_ids).cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=1)  # learning rates
        train_dataset_list = []
        test_dataset_list = []
        for i, each_subject in enumerate(subject):
            emgtrain_dir = f"../../semg-wenci/Data/featureset/{each_subject}_E2_A1_rms_train.h5"
            glovetrain_dir = f"../../semg-wenci/Data/featureset/{each_subject}_E2_A1_glove_train.h5"
            emgtest_dir = f"../../semg-wenci/Data/featureset/{each_subject}_E2_A1_rms_test.h5"
            glovetest_dir = f"../../semg-wenci/Data/featureset/{each_subject}_E2_A1_glove_test.h5"
            train_dataset = NinaPro.NinaPro(emgtrain_dir, glovetrain_dir, window_size=200, subframe=args.subframe,
                                            normalization=args.normalization, mu=args.miu, dummy_label=i,
                                            class_num=len(subject))#Emgtrain_data(3762,12,1)  glovetrain(3762,22)
            test_dataset = NinaPro.NinaPro(emgtest_dir, glovetest_dir, window_size=200, subframe=args.subframe,
                                           normalization=args.normalization, mu=args.miu, dummy_label=i,
                                           class_num=len(subject))#Emgtrain_data(562268,12,1)  glovetrain(562268,22)
            train_dataset_list.append(train_dataset)
            test_dataset_list.append(test_dataset)
        concat_train_dataset = ConcatDataset(train_dataset_list)
        concat_test_dataset = ConcatDataset(test_dataset_list)
        TrainLoader = DataLoader(dataset=concat_train_dataset, num_workers=12, drop_last=True, batch_size=args.batch_size,
                                 shuffle=True, )  # sampler=train_sampler, pin_memory=True)
        DLoader_eval = DataLoader(dataset=concat_test_dataset, num_workers=12, drop_last=True, batch_size=args.batch_size,
                                  shuffle=False, )  # sampler=test_sampler,pin_memory=True)

        # add log
        log_file = os.path.join(save_dir, 'train_result.txt')
        with open(log_file, 'w') as f:
            f.write('----Begin logging----\n')
            f.write("Training begin:" + str(datetime.datetime.now()) + "\n")
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('================ Training loss ================\n')

        # training
        best_epoch = {'epoch': 0, 'NRMSE': 10, 'CC': 0, 'R2': -10}
        loss_count = []
        elpased_time_list = []
        hidden = None
        for epoch in range(initial_epoch, args.epoch):
            print('[*]Current Learning rate ={:.6f}'.format(scheduler.get_last_lr()[0]))

            epoch_loss = 0
            start_time = time.time()
            # training phase
            model.train()
            for n_count, batch_tr in enumerate(TrainLoader):
                #batch_tr0:(4,200,12,1) 1:(4,1,10) 2:(4,10) 3(4,10) 4:(4,1)

                batch_emg = batch_tr[0].squeeze(3).cuda()#(4,200,12,1)
                batch_glove = batch_tr[1].cuda()#(4,1,10)

                # y_c_label = batch_tr[2].cuda()
                y_s_label = batch_tr[3].cuda()#(4,10)
                sub_id_val = int(subject_name[1:])
                subject_tensor = torch.full((batch_emg.size(0),), sub_id_val, dtype=torch.long, device=batch_emg.device)
                y_pred = model(batch_emg, subject_tensor)
                loss = reg_loss(y_pred, batch_glove)

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if n_count % (len(TrainLoader) // 10) == 0:
                    message = '[{}] {} / {} loss = {} '.format(epoch + 1, n_count, len(TrainLoader),
                                                              loss.item() / args.batch_size,)
                    loss_count.append(loss.item() / args.batch_size)
                    print(message)
                    with open(log_file, 'a') as f:
                        f.write(message)
                        f.write('\n')
            elapsed_time = time.time() - start_time
            elpased_time_list.append(elapsed_time)

            # evaluation phase
            model.eval()
            NRMSEs = []
            CCs = []
            total_output_test = torch.Tensor([]).cuda()
            total_target_test = torch.Tensor([]).cuda()
            with torch.no_grad():
                i = 0
                hidden_1 = hidden
                for _, batch_eval in enumerate(DLoader_eval):
                    i += 1
                    glove_true = batch_eval[1].cuda()
                    glove_true = glove_true.view(glove_true.shape[1] * args.batch_size,
                                                 glove_true.shape[2])  # .cpu().numpy()  # .astype(np.float32)

                    data_emg = batch_eval[0].squeeze(3).cuda()
                    sub_id_val = int(subject_name[1:])
                    subject_tensor_eval = torch.full((data_emg.size(0),), sub_id_val, dtype=torch.long,
                                                     device=data_emg.device)
                    glove_pred = model(data_emg, subject_tensor_eval)

                    # glove_pred, hidden_1 = model(data_wrapper(batch_eval[0].squeeze(3)).cuda(), hidden_1)  # RNN with hidden

                    glove_pred = glove_pred.view(glove_pred.shape[1] * args.batch_size,
                                                 glove_pred.shape[2])  # .cpu().numpy()  # .astype(np.float32)

                    total_output_test = torch.cat([total_output_test, glove_pred])
                    total_target_test = torch.cat([total_target_test, glove_true])
            total_output_test = total_output_test.detach().cpu().numpy()
            if args.smooth:
                total_output_test = avg_smoothing_np(5, total_output_test)
            total_target_test = total_target_test.detach().cpu().numpy()
            NRMSE = metrics.normalized_root_mse(total_target_test, total_output_test, normalization="min-max")
            CC_pearson = pearson_CC(total_target_test, total_output_test)
            r2 = skmetrics.r2_score(total_target_test.T, total_output_test.T, multioutput="variance_weighted")
            # add log
            record_writer.add_scalar('Loss', epoch_loss, global_step=epoch + 1)
            record_writer.add_scalar('CC', CC_pearson, global_step=epoch + 1)
            record_writer.add_scalar('NRMSE', NRMSE, global_step=epoch + 1)
            record_writer.add_scalar('R2', r2, global_step=epoch + 1)
            if NRMSE < best_epoch['NRMSE']:
                # if CC_pearson > best_epoch['CC']:
                torch.save(model, os.path.join(save_dir, 'model_best.pth'))
                best_epoch['NRMSE'] = NRMSE
                best_epoch['epoch'] = epoch + 1
                best_epoch['CC'] = CC_pearson
                best_epoch['R2'] = r2
                record_writer.add_text("Best epoch", str(epoch + 1))
                record_writer.add_text("Best CC now", str(best_epoch['CC']), )
                record_writer.add_text("Best NRMSE now", str(best_epoch['NRMSE']), )
                record_writer.add_text("Best R2 now", str(best_epoch['R2']), )
                fig = draw_graph(total_output_test, total_target_test)
                record_writer.add_figure("test_results", fig, global_step=epoch + 1)

            message1 = 'epoch = {:03d}, [time] = {:.2f}s, [NRMSE of {}-frames]:{:.3f}, [CC of {}-frames]:{:.3f},[R2 of {}-frames]:{:.3f},[loss] = {:.7f}.'.format(
                epoch + 1,
                elapsed_time,
                i, NRMSE,
                i, CC_pearson,
                i, r2,
                epoch_loss)
            message2 = 'Best @ {:03d}, with NRMSE {:.3f}, CC {:.3f}, R2 {:.3f}. \n'.format(best_epoch['epoch'],
                                                                                           best_epoch['NRMSE'],
                                                                                           best_epoch["CC"],
                                                                                           best_epoch['R2'])
            print(message1)
            print(message2)
            with open(log_file, 'a') as f:
                f.write(message1)
                f.write("\n")
                f.write(message2)
            torch.save(model, os.path.join(save_dir, 'model_latest.pth'))
            scheduler.step()  # step to the learning rate in this epoch

        elpased_time_list = numpy.array(elpased_time_list)
        time_message = f"[*]Average time cost: {numpy.sum(elpased_time_list) / args.epoch}"
        record_writer.add_text("Average time cost", str(numpy.sum(elpased_time_list) / args.epoch), )
        print(time_message)
        with open(log_file, "a") as f:
            f.write("\n")
            f.write(time_message)

        torch.cuda.empty_cache()
        record_writer.close()

