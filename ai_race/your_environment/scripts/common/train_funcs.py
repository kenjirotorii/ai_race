'''
Train functions.
'''
from utils import calc_score

def train_ae(model, optimizer, scheduler, loss_fn, dataloader, device, variational):

    model.train()
    final_loss = 0

    for _, (inputs, targets) in enumerate(dataloader):
        
        targets = targets.to(device)

        for img in inputs:
            img = img.to(device)
            if variational:
                y_pred, mu, lnvar = model(img)
                loss = loss_fn(y_pred, targets, mu, lnvar)
            else:
                y_pred = model(img)
                loss = loss_fn(y_pred, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            final_loss += loss.item() / len(inputs)
        
        scheduler.step()

    final_loss /= len(dataloader)

    return final_loss


def valid_ae(model, loss_fn, dataloader, device, variational):

    model.eval()
    final_loss = 0
    
    for _, (inputs, targets) in enumerate(dataloader):
        
        targets = targets.to(device)

        for img in inputs:
            img = img.to(device)
            if variational:
                y_pred, mu, lnvar = model(img)
                loss = loss_fn(y_pred, targets, mu, lnvar)
            else:
                y_pred = model(img)
                loss = loss_fn(y_pred, targets)

            final_loss += loss.item() / len(inputs)

    final_loss /= len(dataloader)

    return final_loss


def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):

    model.train()
    final_loss = 0
    output_list = []
    target_list = []

    for _, (inputs, targets) in enumerate(dataloader):
        
        targets = targets.to(device)

        inputs = inputs.to(device)
        y_pred = model(inputs)
        loss = loss_fn(y_pred, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        final_loss += loss.item()
        output_list += [int(y.argmax()) for y in y_pred]
        target_list += [int(t) for t in targets]
        
        scheduler.step()

    train_acc, train_loss = calc_score(output_list, target_list, final_loss, dataloader)

    return train_acc, train_loss


def valid_fn(model, loss_fn, dataloader, device):

    model.eval()
    final_loss = 0
    output_list = []
    target_list = []
    
    for _, (inputs, targets) in enumerate(dataloader):
        
        targets = targets.to(device)
        inputs = inputs.to(device)

        y_pred = model(inputs)
        loss = loss_fn(y_pred, targets)

        final_loss += loss.item()
        output_list += [int(y.argmax()) for y in y_pred]
        target_list += [int(t) for t in targets]

    valid_acc, valid_loss = calc_score(output_list, target_list, final_loss, dataloader)

    return valid_acc, valid_loss
