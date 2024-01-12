import torch
from transformer_model import transformer_model
import numpy as np
import os
import random
import torch.optim as optim
import torch.nn as nn
from Myloader import *
import time
import torchvision.models as models
from torchmetrics.classification import MultilabelAveragePrecision
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def evaluate(model, val_loader):
    model.eval()
    test_running_loss = 0.0
    test_total = 0

    with torch.no_grad():
        record_target_label = torch.zeros(1, 19).to(device)
        record_predict_label = torch.zeros(1, 19).to(device)
        for (test_imgs, test_labels, test_dicoms) in val_loader:
            test_imgs = test_imgs.to(device)
            test_labels = test_labels.to(device)
            test_labels = test_labels.squeeze(-1)

            test_output = model(test_imgs)
            loss = criterion(test_output, test_labels)

            test_running_loss += loss.item() * test_imgs.size(0)
            test_total += test_imgs.size(0)

            record_target_label = torch.cat((record_target_label, test_labels), 0)
            record_predict_label = torch.cat((record_predict_label, test_output), 0)


        record_target_label = record_target_label[1::]
        record_predict_label = record_predict_label[1::]

        metric = MultilabelAveragePrecision(num_labels=19, average="macro", thresholds=None)
        mAP = metric(record_predict_label, record_target_label.to(torch.int32))

        metric = MultilabelAveragePrecision(num_labels=19, average="none")
        mAPs = metric(record_predict_label, record_target_label.to(torch.int32))
        
    return mAP, mAPs, test_running_loss, test_total

if __name__ == "__main__":
    set_seed(123)
    torch.cuda.set_device(0)

    print('please input the exp_id...')
    exp_id = input()
    weight_dir = "weight"
    output_path = f"results/{exp_id}"
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    epochs = 10
    batch_size = 32
    num_classes = 19

    train_path = "data/MICCAI_long_tail_train.tfrecords"
    train_index = "data/MICCAI_long_tail_train.tfindex"
    val_path = "data/MICCAI_long_tail_val.tfrecords"
    val_index = "data/MICCAI_long_tail_val.tfindex"

    opt_lr = 3e-5
    weight_decay = 1e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = transformer_model().to(device)

    opt = optim.Adam(encoder.parameters(), lr=opt_lr, weight_decay = weight_decay)
    train_loader = Myloader(train_path, train_index, batch_size, num_workers=0, shuffle=True)
    val_loader = Myloader(val_path, val_index, batch_size, num_workers=0, shuffle=False)

    cnt = np.array([51055., 2595., 55020., 12406., 30553., 19880., 7823., 1707., 57591., 34376., 53029., 373., 571., 32131., 448., 11579., 1900., 73639., 2212.])
    total = 162143
    pos_weight = torch.tensor((total - cnt) / cnt, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_losses = []
    test_losses = []

    max_map = 0
    total = 0
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        encoder.train()
        running_loss = 0.0
        start_time = time.time()
        time_pin = time.time()
        count = 0
        logs = []

        for (imgs, labels, dicom_id) in train_loader:
            encoder.zero_grad()
            opt.zero_grad()

            imgs = imgs.to(device)
            labels = labels.to(device)
            labels = labels.squeeze(-1)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = encoder(imgs)
                loss = criterion(output, labels)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            count += imgs.size(0)

            if count % 1024 == 0:
                if total == 0:
                    print(f"epoch {epoch}: {count}/unknown | train loss: {running_loss / count:.4f} | duration: {time.time() - time_pin:.2f} seconds")
                    logs.append("epoch "+ str(epoch) +": "+ str(count) +"/unknown | train loss: "+ str(round(running_loss / count, 4)) +" | duration: "+ str(round(time.time() - time_pin)) +" seconds")
                elif total != 0:
                    print(f"epoch {epoch}: {count}/{total} {count/total*100:.2f}% | train loss: {running_loss / count:.4f} | duration: {time.time() - time_pin:.2f} seconds")
                    logs.append("epoch "+ str(epoch) +": "+ str(count) +"/unknown | train loss: "+ str(round(running_loss / count, 4)) +" | duration: "+ str(round(time.time() - time_pin)) +" seconds")
                time_pin = time.time()

        total = count
        mAP, mAPs, test_running_loss, test_total = evaluate(encoder, val_loader)
        
        train_losses.append(running_loss / count)
        test_losses.append(test_running_loss)
        
        if mAP > max_map:
            max_map = mAP
            torch.save({
                'model_state_dict': encoder.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
            }, f"{weight_dir}/{exp_id}_model_best.pt")

        end_time = time.time()
        duration = end_time - start_time
        
        print(f"epoch {epoch}: validation mAP: {mAP} | validation loss: {test_running_loss / test_total} | duration: {duration}")
        print(f"epoch {epoch}: validation mAPs: {mAPs}")
        
        with open(output_path+'/output.txt', 'a') as file:
            for log in logs:
                file.write(f"{log}\n")
            file.write(f"epoch {epoch}: validation mAP: {mAP} | validation loss: {test_running_loss / test_total} | duration: {duration}")

        with open(output_path+'/result.txt', 'a') as file:
            file.write(f"epoch {epoch}: validation mAP: {mAP} | validation loss: {test_running_loss / test_total} | duration: {duration}")
            file.write(f"epoch {epoch}: validation mAPs: {mAPs}")
            file.write('\n')
