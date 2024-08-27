import configargparse


def config_parser():
    # 用configargparse库解析输入参数
    parser = configargparse.ArgumentParser()
    # 定义一个config文件类型参数，即可以在config中DIY自己的训练参数，而不用繁琐地在命令行输入大量参数
    # 在目录下创建一个configs文件夹，专门存放各类训练配置文件，便于复现
    # 默认为configs/lego.txt，直接运行train_nerf.py默认训练lego
    # 运行时只需要在终端输入：python train_nerf.py --config configs/lego.txt
    parser.add_argument('--config', is_config_file=True, default="configs/lego.txt",
                        help='config file path')

    """训练结果相关参数"""
    # 训练结果文件名
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    # 结果文件夹，即结果存放地址地基础文件名，建议默认为./logs/，即训练结果都存放到该文件夹中
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    """加载数据相关参数"""
    # 数据所在文件夹，默认为lego数据
    parser.add_argument("--datadir", type=str, default='./data/nerf_synthetic/lego',
                        help='input data directory')
    # 定义数据类型，默认为blender，也只支持该类型
    parser.add_argument("--dataset_type", type=str, default='blender',
                        help='options: llff / blender / deepvoxels')
    # 这个参数是给llff类型数据使用，默认为false，但不可删除，网络中需要判断此参数结果
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    # 设置读取数据步幅，默认为8
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    # 是否以白色为背景选项
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    # 是否进行降采样选项
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    """训练模型相关参数"""
    parser.add_argument("--N_iters", type=int, default=10000,
                        help='number of iterations')
    # 网络深度
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    # 网络宽度
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    # 精细网络深度
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    # 精细网络宽度
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')

    # 每次迭代训练中从图像中随机采样的光线数量
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    # 初始学习率
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    # 学习率衰减
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    # 一次参与计算的射线数
    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    # 一次性最多放入网络计算的射线
    parser.add_argument("--netchunk", type=int, default=1024 * 64 * 8,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    # 是否在每次训练迭代中只从一张图片中随机采样一批光线进行训练
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    # 是否不加载权重文件进行训练
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    # 预训练权重文件路径
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    """渲染相关参数"""
    # 粗采样的数量，默认为64
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    # 精细采样的数量，默认为0，即不进行。但是通常会在config文件中设置，一般为128
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    # 用于控制光线采样点的抖动程度。通过加入抖动，模型可以在训练过程中看到更多不同的场景，从而提高模型的稳定性和泛化能力
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    # 是否进行视角编码，一般在config文件中设置，blender数据一般都会进行
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    # 是否进行默认的位置编码，0进行，-1不进行
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    # 位置编码数量，例如10那么XYZ将会变为63，具体编码细节请看文章中的解释
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    # 视角编码数量，例如4，那么3D视角将会变为27，符合文章给出的模型结构
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    # 噪声，增强模型泛化能力，默认为0
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    # 是否仅进行渲染，不进行训练，默认进行训练
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    # 在测试集上进行渲染测试，一般用来评估模型在从未见过的数据上的性能，与上面那个参数配套使用
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    # 控制渲染的分辨率缩放因子，默认为0
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    # 控制训练期间早期阶段的图像裁剪策略的参数，可加快模型的收敛并提高训练的稳定性，由config中定义。
    # 具体来说，在训练的前几个迭代中，模型可能只处理图像的中心区域（crop区域），因为中心区域通常包含更多有用的特征和信息
    # 通过这种方式，模型可以更快地学习到有效的表示。
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    # 在训练初期阶段进行裁剪时，从中心裁剪的比例，同上面那个参数配套使用
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')
    """保存结果相关参数"""
    # 控制台打印输出和度量日志的频率
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    # 生成结果图片的频率
    parser.add_argument("--i_img", type=int, default=500,
                        help='frequency of tensorboard image logging')
    # 生成权重文件的频率
    parser.add_argument("--i_weights", type=int, default=5000,
                        help='frequency of weight ckpt saving')
    # 测试数据进行测试的频率
    parser.add_argument("--i_testset", type=int, default=5000,
                        help='frequency of testset saving')
    # 测试数据生成视频的频率
    parser.add_argument("--i_video", type=int, default=5000,
                        help='frequency of render_poses video saving')

    return parser
