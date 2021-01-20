### backbone

    resnet / resnext


### PAN

    pan: 一条up path & 一条down path
    * fpn path:
        C_k都要先通过1x1的conv调整通道数
        P_k要upSamp
        然后与C_k-1相加
        然后再过一个3x3 s1的conv

    * aug path:
        N2就是P2
        N_k是3x3 s2的conv下采样得到
        然后与P_k+1相加
        然后再过一个3x3 s1的conv


### rpn branch
    give dense pos/neg rpn box preditions
    [b,h,w,a,4+1]
    论文里说proposals are from an independently trained RPN, not shared with the det/seg task
    常规的RPN(暂定认为是maskRCNN中的RPN):
    * shared 3x3 conv between cls & box
    * individual 1x1 conv head
    * shared among all levels


### adaptive feature pooling: 
    pooling features from all levels for each proposal
    而不是基于proposal的尺度分配给某一个level的feature
    * map the propsals on all level feature maps
    * ROI Align
    * element-wise max / add


### box branch
    give sparse categorical box preditions
    fuse after first fc


### mask branch
    give sparse categorical mask preditions
    fuse after first conv
    fc branch
    ??????????


### todolist
    几个layer从maskRCNN中搬过来
    timedistributed











