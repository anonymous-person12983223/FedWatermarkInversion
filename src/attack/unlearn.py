import torch
import torch.nn as nn

class ModelUnlearner:
    def __init__(self, model, inverse_loader, args, clean_loader=None):
        self.model = model.to(args.device)
        self.inverse_loader = inverse_loader
        self.clean_loader = clean_loader
        self.args = args

        self.use_clean = clean_loader is not None
        self.loss_inverse = nn.KLDivLoss(reduction='batchmean')
        self.loss_clean = nn.CrossEntropyLoss() if self.use_clean else None
        self.inverse_loss_weight = args.inverse_loss_weight if self.use_clean else 1.0

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=args.unlearn_lr,
            momentum=args.unlearn_momentum
        )

    def run(self):
        self.model.train()
        inverse_iter = iter(self.inverse_loader)

        for epoch in range(self.args.unlearn_epochs):
            total_loss = 0
            batch_cnt = 0

            loader = self.clean_loader if self.use_clean else self.inverse_loader

            for batch in loader:
                batch_cnt += 1

                if self.use_clean:
                    batch_data_clean, target_clean = batch
                    try:
                        batch_data_inverse, target_inverse = next(inverse_iter)
                    except StopIteration:
                        inverse_iter = iter(self.inverse_loader)
                        batch_data_inverse, target_inverse = next(inverse_iter)
                    batch_data_clean = batch_data_clean.to(self.args.device)
                    batch_data_inverse = batch_data_inverse.to(self.args.device)
                    batch_data = torch.cat((batch_data_clean, batch_data_inverse))
                    target_clean = target_clean.to(self.args.device)
                    target_inverse = target_inverse.to(self.args.device)
                else:
                    batch_data, target_inverse = batch
                    batch_data = batch_data.to(self.args.device)
                    target_inverse = target_inverse.to(self.args.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_data)

                if self.use_clean:
                    output_clean = outputs[:len(batch_data_clean)]
                    output_inverse = outputs[len(batch_data_clean):]
                    l1 = self.loss_clean(output_clean, target_clean)
                else:
                    output_inverse = outputs

                output_inverse = torch.log_softmax(output_inverse, dim=-1)
                l2 = self.loss_inverse(output_inverse, target_inverse)

                loss = l1 + (self.inverse_loss_weight * l2) if self.use_clean else l2
                total_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            print(f"[Epoch {epoch + 1}] Loss: {total_loss / batch_cnt:.4f}")
