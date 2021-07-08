import numpy as np
from torch.utils.data import DataLoader
from validate import extract_features
from utils import ExemplarSet as ImagesDataset


def perform_selection(args, images, labels, net, transform):
    if args.selection == "random":
        indexes = np.random.permutation(len(images))[:args.rehearsal]
    else:
        dataset_class_c = ImagesDataset(images, labels, transform)
        loader = DataLoader(dataset_class_c, batch_size=args.batch_size, shuffle=False, drop_last=False)
        features = extract_features(args, net, loader)
        if args.selection == "closest":
            indexes = closest_to_mean(features, args.rehearsal)
        elif args.selection == "herding":
            indexes = icarl_selection(features, args.rehearsal)
    return indexes


def _l2_distance(x, y):
    return np.power(x - y, 2).sum(-1)


def icarl_selection(features, nb_examplars):
    D = features.T
    D = D / (np.linalg.norm(D, axis=0) + 1e-8)
    mu = np.mean(D, axis=1)
    herding_matrix = np.zeros((features.shape[0],))

    w_t = mu
    iter_herding, iter_herding_eff = 0, 0

    while not (
            np.sum(herding_matrix != 0) == min(nb_examplars, features.shape[0])
    ) and iter_herding_eff < 1000:
        tmp_t = np.dot(w_t, D)
        ind_max = np.argmax(tmp_t)
        iter_herding_eff += 1
        if herding_matrix[ind_max] == 0:
            herding_matrix[ind_max] = 1 + iter_herding
            iter_herding += 1

        w_t = w_t + mu - D[:, ind_max]

    herding_matrix[np.where(herding_matrix == 0)[0]] = 10000

    return herding_matrix.argsort()[:nb_examplars]


def closest_to_mean(features, nb_examplars):
    # features = features / (np.linalg.norm(features, axis=0) + 1e-8)
    class_mean = np.mean(features, axis=0)

    return _l2_distance(features, class_mean).argsort()[:nb_examplars]
