
import torch
import time
import sys
import os

class TrainingAlgo:

    def __init__(self,learning_rate_in = 0.001, n_epochs_in = 10):

        self.learning_rate = learning_rate_in
        self.n_epochs = n_epochs_in

        self.hist = {"loss_train": [], "loss_val": []}

    def save_model(self, state, dirName="./model", model_name="best_model"):

        if not os.path.exists(dirName):
            os.makedirs(dirName)
        model_name = "{}.pt".format(model_name)
        save_path = os.path.join(dirName, model_name)
        path = open(save_path, mode="wb")
        torch.save(state, path)
        path.close()

    def train(self,model,loader_train,loader_val,device):

        optimizer = torch.optim.Adam(model.parameters(), self.learning_rate, weight_decay=0)  # optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.7, patience=10, min_lr=5e-4)
        training_time = 0
        min_val_loss = 1
        loss = torch.nn.SmoothL1Loss()  # loss function

        for epoch in range(0, self.n_epochs):

            time_counter = time.time()
            total_train_loss = 0
            running_loss = 0
            model.train()
            for i, train_data in enumerate(loader_train):
                optimizer.zero_grad()
                out_n = model(train_data.batch,train_data.x, train_data.edge_index, train_data.edge_attr)
                #print("OUT N",out_n)
                #y_n   = torch.cat([ d.y for d in train_data]).to(device)
                y_n = train_data.y
                #print("Y N",train_data.y)
                loss_train = loss(out_n, y_n)
                loss_train.backward()
                optimizer.step()
                total_train_loss += loss_train.item()
                running_loss += loss_train.item()

                if (i + 1) % (len(loader_train) // 10 + 1) == 0:
                    print("Epoch {}, {:d}% \t train_loss: {:.5e} \t lr: {:.3e}".format(
                        epoch + 1,
                    int(100 * (i + 1) / len(loader_train)),
                    running_loss / (len(loader_train) // 10),
                    optimizer.param_groups[0]['lr']))
                    running_loss = 0.0
                sys.stdout.flush()

            self.hist["loss_train"].append(total_train_loss / len(loader_train))
            print("Training loss = {:.5e}".format(total_train_loss / len(loader_train)))

            total_val_loss = 0
            model.eval()
            for val_data in loader_val:
                val_out_n = model(val_data.batch, val_data.x, val_data.edge_index, val_data.edge_attr)
                y_n = val_data.y
                val_loss = loss(val_out_n, y_n)
                total_val_loss += val_loss.item()

            scheduler.step(total_val_loss )
            self.hist["loss_val"].append(total_val_loss)

            training_time = training_time + (time.time() - time_counter)
            print("Validation loss = {:.5e}".format(total_val_loss))
            sys.stdout.flush()

            # condition to save model
            if total_val_loss  <= min_val_loss:
                checkpoint = {
                    'epoch': epoch + 1,
                    'min_val_loss': total_val_loss ,
                    'state_dict': model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'loss_train': self.hist["loss_train"],
                    'loss_val': self.hist["loss_val"],
                    'training_time': training_time
                }
                # save model
                self.save_model(checkpoint, dirName="./model", model_name="best_model_normal")
                min_val_loss = total_val_loss
                print("Training finished, took {:.2f}s, MODEL SAVED".format(training_time))
            else:
                print("Training finished, took {:.2f}s".format(training_time))


        checkpoint = {
                'epoch': epoch + 1,
                'min_val_loss': total_val_loss ,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss_train': self.hist["loss_train"],
                'loss_val': self.hist["loss_val"],
                'training_time': training_time
            }
        # save model
        self.save_model(checkpoint, dirName="./model", model_name="best_model_normal_final")
