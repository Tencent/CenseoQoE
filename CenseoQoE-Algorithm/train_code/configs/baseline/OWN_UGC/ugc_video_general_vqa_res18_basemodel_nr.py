# 关于模型的配置
model = dict(
    model_name="CenseoIVQAModel",  # 调用的模型名称，具体看models/
    fr=False,  # fr=True代表有参，否则为无参
    backbone=dict(  # backbone模型的设置，具体见models/backbones/
        model_name="resnet18",
        kwargs=dict(
            pretrained=True,
            out_indices=(3,),
            strides=(2, 2, 2)
        )
    ),
    head=dict(  # 预测头部的设置，具体见models/heads/
        model_name="SimpleHead",
        kwargs=dict(
            out_num=1
        )
    ),
    use_sigmoid=True  # 模型输出是否加sigmoid函数
)

# 关于数据的配置
data = dict(
    dataset_name="ImageVideoGeneralDataset",  # 选择使用的dataset类加载数据，具体看datasets/
    is_fr=model["fr"],  # 是否有参，跟model的fr参数保持一致
    data_roots=["/data/IVQA/Oteam/own-ugc-round1-frames/",
                "/data/IVQA/Oteam/own-ugc-round2-frames/",
                "/data/IVQA/Oteam/own-ugc-round3-frames/"],
    # 视频/图片数据的根目录，如果有多个数据集请在list里依次填写
    data_infos=["./data/own_ugc_round1_processed.json",
                "./data/own_ugc_round2_processed.json",
                "./data/own_ugc_round3_processed.json"],
    # 视频/图片信息（mos/dmos等等）json文件路径，如果有多个数据集请在list里依次填写
    mos_type="mos",  # 选择需要训练/预测的分数类型：mos/dmos或是其他
    mos_scales=[0.2, 0.2, 0.2],  # 是否对mos分做缩放处理，1代表不缩放
    split_seed=1126,  # 划分训练集和验证集的随机种子
    test_p=[0.2, 0.2, 0.2],  # 验证集比例
    train_p=[0.8, 0.8, 0.8],  # 训练集比例
    train_clip_len=1,  # 每个视频训练时输入到模型的视频帧数量（仅对视频训练有效）
    test_clip_len=32,  # 每个视频测试时输入到模型的视频帧数量（仅对视频训练有效）
    read_mode='resize_random_crop',  # 读取图片和转换的方式
    sub_img_dim=(1280, 720),  # 图像大小/模型输入大小
    resize_dim=(1920, 1080),  # 图片resize大小
    batch_size=32  # batch_size
)

EPOCHS = 50  # 总的训练epoch
trainer = dict(
    trainer_name="CenseoBaseTrainer",  # 选择的trainer类型， 具体看trainers/
    store_dir="/data/exps/censeo-train/results/",  # log/config/模型的保存路径
    gpu_num=1,  # 训练使用的GPU数量
    resume_from=None,
    # 训练好的模型的路径，用于resume
    epochs=EPOCHS,
    save_model_inter=1,  # 模型保存的间隔，1表示每个epoch都保存
    eval_model_inter=1,  # 模型做eval的间隔，1表示每个epoch都在验证集做验证
    vis=dict(  # 利用visdom做训练可视化
        server="http://localhost",
        port="8083"
    ),
    optimizer=dict(  # 训练优化器optimizer的配置
        optimizer_name="SGD",
        kwargs=dict(
            lr=1e-4,
            weight_decay=5e-4,
            momentum=0.9
        )
    ),
    lr_scheduler=dict(  # 学习率衰减策略配置
        lr_scheduler_name="CosineAnnealingLR",
        kwargs=dict(
            T_max=EPOCHS,
            eta_min=1e-07
        )
    ),
    loss=dict(  # 损失函数配置
        loss_name="L1Loss",  # 选择的loss类型， 具体看losses/
        kwargs=dict(
            loss_weight=1
        )
    )
)
