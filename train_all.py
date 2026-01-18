from pretrain_mask_vae import main_pretrain
from classifier import main_train_classifier

if __name__ == '__main__':
    for i in range(0,10):
        main_pretrain(i)
        main_train_classifier(i)

