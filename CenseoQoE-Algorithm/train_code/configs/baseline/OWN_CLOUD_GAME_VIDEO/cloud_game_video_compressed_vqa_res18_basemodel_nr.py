_base_ = ['../OWN_PGC/pgc_video_compressed_vqa_res18_basemodel_nr.py']

# 关于数据的配置
data = dict(
    dataset_name="ImageVideoGeneralDataset",  # 选择使用的dataset类加载数据，具体看datasets/
    is_fr=False,  # 是否有参，跟model的fr参数保持一致
    data_roots=["./data/IVQA/own_games_compression-frames/"],
    # 视频/图片数据的根目录，如果有多个数据集请在list里依次填写
    data_infos=["./data/OWN_GAMES_COMPRESSION/own_games_compression_processed.json"],
    # 视频/图片信息（mos/dmos等等）json文件路径，如果有多个数据集请在list里依次填写
    mos_type="mos",  # 选择需要训练/预测的分数类型：mos/dmos或是其他
    mos_scales=[0.2],  # 是否对mos分做缩放处理，1代表不缩放
    split_seed=1126,  # 划分训练集和验证集的随机种子
    train_clip_len=1,  # 每个视频训练时输入到模型的视频帧数量（仅对视频训练有效）
    test_clip_len=16,  # 每个视频测试时输入到模型的视频帧数量（仅对视频训练有效）
    test_p=[0.2],  # 验证集比例
    train_p=[0.8],  # 训练集比例
    read_mode='resize_random_crop',  # 读取图片和转换的方式
    batch_size=32,  # batch_size
    sub_img_dim=(720, 1280),  # 图像大小/模型输入大小
    resize_dim=(1080, 1920),  # 图片resize大小
)
