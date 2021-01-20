# resnet 50 & 101
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, ReLU, add, GlobalAveragePooling2D, UpSampling2D, Lambda, Dense, Reshape, Conv2DTranspose, multiply
from keras.models import Model
import keras.backend as K
from backbone import resnet
import tensorflow as tf
from loss import rpn_loss


def PANet(input_shape=(224,224,3), n_classes=80, n_anchors=15, n_proposals=300, rpn=True, train_rpn=True):
    inpt = Input(input_shape)

    # back
    back = resnet(input_shape, depth=50)
    features = back(inpt)     # [x4,x8,x16,x32]

    # fpn
    fpn_features = fpn(features)

    # augmented bottom-up path
    aug_features = aug(fpn_features)

    if rpn:
        # rpn branch
        rpn_outputs = [rpn_branch(i, n_anchors) for i in aug_features]
        if train_rpn:
            x = Lambda(rpn_loss, arguments={'n_anchors': n_anchors})(rpn_outputs)
        else:
            # gather high conf boxes & flatten along axis1
            x = ProposalLayer(rpn_outputs)
        model = Model(inpt, x)
        return model

    else:
        # refine branch
        proposals = Input((n_proposals, 4))
        gt_boxes = Input((n_proposals, 4+n_classes))
        gt_masks = Input((n_proposals, 28,28,n_classes))
        cls_outputs, box_outputs = box_branch(aug_features, proposals)
        mask_outputs = mask_branch(aug_features, proposals)

        box_loss_ = Lambda(box_loss, arguments={'n_anchors': n_anchors})(*box_outputs, gt_boxes)
        mask_loss_ = Lambda(mask_loss, arguments={'n_anchors': n_anchors})(mask_outputs, gt_masks)

        model = Model(inpt, [box_loss_, mask_loss_])

        return Model


def box_branch(features, proposals, n_classes, n_filters=1024):
    pooled_features = []
    for feature in features:
        # roiAlign & fc1: maskRCNN use conv
        x = ROIAlign(feature, proposals, pool_size=7)
        x = Conv_BN(x, n_filters, 7, strides=1, padding='same', activation='relu')
        pooled_features.append(x)

    # element-wise add / max
    x = add([pooled_features])

    # fc2
    x = Conv_BN(x, n_filters, 1, strides=1, padding='same', activation='relu')

    # prediction
    x = Reshape((-1,2))(x)
    cls_head = Dense(n_classes, activation='softmax')(x)
    box_head = Dense(n_classes*4, activation=None)(x)
    return cls_head, box_head


def mask_branch(features, proposals, n_classes, n_filters=256):
    pooled_features = []
    for feature in features:
        # roiAlign & conv1: maskRCNN use conv
        x = ROIAlign(feature, proposals, pool_size=14)
        x = Conv_BN(x, n_filters, 3, strides=1, padding='same', activation='relu')
        pooled_features.append(x)

    # element-wise add / max
    x = add([pooled_features])

    # conv2, conv3
    x = Conv_BN(x, n_filters, 3, strides=1, padding='same', activation='relu')
    x = Conv_BN(x, n_filters, 3, strides=1, padding='same', activation='relu')

    # conv4: with fully-connected fusion
    skip = Conv_BN(x, n_filters, 3, strides=1, padding='same', activation='relu')
    skip = Conv_BN(skip, n_filters//2, 3, strides=1, padding='same', activation='relu')
    skip = GlobalAveragePooling2D()(skip)
    skip = Conv_BN(skip, 28*28, 1, strides=1, activation='relu')
    x = Conv_BN(x, n_filters, 3, strides=1, padding='same', activation='relu')
    x = add([x, skip])

    # deconv by 2
    mask_head = Conv2DTranspose(n_classes, 3, strides=2, padding='same', activation='sigmoid')(x)
    return mask_head


def rpn_branch(inpt, n_anchors, n_filters=512):
    # give dense [b,h,w,a,4+1] predictions
    x = Conv2D(n_filters, 3, strides=1, activation='relu', padding='same')(inpt)
    cls_head = Conv2D(n_anchors*2, 1, strides=1, padding='same', activation='softmax')(x)
    box_head = Conv2D(n_anchors*4, 1, strides=1, padding='same', activation='softmax')(x)
    model = Model(inpt, [cls_head, box_head])
    return model


def fpn(features, n_filters=256):
    # generate P5 to P2
    C2, C3, C4, C5 = features
    P5 = Conv_BN(C5, n_filters, 1, strides=1, activation='relu')

    U4 = UpSampling2D(size=2)(P5)
    C4 = Conv_BN(C4, n_filters, 1, strides=1, activation='relu')
    P4 = add([C4, U4])
    P4 = Conv_BN(P4, n_filters, 3, strides=1, activation='relu')

    U3 = UpSampling2D(size=2)(P4)
    C3 = Conv_BN(C3, n_filters, 1, strides=1, activation='relu')
    P3 = add([C3, U3])
    P3 = Conv_BN(P3, n_filters, 3, strides=1, activation='relu')

    U2 = UpSampling2D(size=2)(P3)
    C2 = Conv_BN(C2, n_filters, 1, strides=1, activation='relu')
    P2 = add([C2, U2])
    P2 = Conv_BN(P2, n_filters, 3, strides=1, activation='relu')

    return [P2, P3, P4, P5]


def aug(features, n_filters=256):
    # generate N2 to N5
    P2, P3, P4, P5 = features
    N2 = P2

    D3 = Conv_BN(N2, n_filters, 3, strides=2, activation='relu')
    N3 = add([P3, D3])
    N3 = Conv_BN(N3, n_filters, 3, strides=1, activation='relu')

    D4 = Conv_BN(N3, n_filters, 3, strides=2, activation='relu')
    N4 = add([P4, D4])
    N4 = Conv_BN(N4, n_filters, 3, strides=1, activation='relu')

    D5 = Conv_BN(N4, n_filters, 3, strides=2, activation='relu')
    N5 = add([P5, D5])
    N5 = Conv_BN(N5, n_filters, 3, strides=1, activation='relu')

    return [N2, N3, N4, N5]


def Conv_BN(x, n_filters, kernel_size, strides, activation=None):
    x = Conv2D(n_filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    if activation:
        x = ReLU()(x)
    return x


if __name__ == '__main__':

    model = PANet(input_shape=(224,224,3))
    # model.summary()









