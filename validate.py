import torch


def validate():
    pass


def extract_features(args, net, loader):
    features = None
    net.cuda()
    net.eval()
    
    with torch.no_grad():
        for inputs, targets in loader:
            #inputs = inputs.cuda(args.device)
            inputs = inputs.cuda()
            f, _ = net(inputs)
            #f = l2_norm(f)
            if features is not None:
                features = torch.cat((features, f), 0)
            else:
                features = f

    return features.detach().cpu().numpy()



def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output

