import torch
import torch.nn as nn
import torch.nn.functional as F
import os


'''
    根据论文中的定义，模型的多层结构先是一个8层全连接层，用来训练RGB
    后面再加入视角参数进行训练
    可以看看imgs文件夹下的NeRF网络结构图
'''


class NeRF(nn.Module):
    # 模型初始化需要下列参数，需要在opts中对应加入：
    # D：网络深度
    # W：网络宽度
    # input_ch：输入通道数
    # output_ch：输出通道数
    # skips：半路再次输入位置坐标的地方，一般默认
    # use_viewdirs：是否使用视角参数，即输入为RGB加上俯仰角。一般训练blender数据都会用
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        """这是原版pytorch实现方式，为了方便理解，我将其8层展开"""
        # # 构建多层感知器
        # self.pts_linears = nn.ModuleList(
        #     [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
        #                                 range(D - 1)])
        # 总共8层，input_ch文章中是63，W=256，不理解为什么是63就看一下文章中提到的位置编码
        self.pts_linears = nn.ModuleList(
            [nn.Linear(in_features=input_ch, out_features=W, bias=True),
            nn.Linear(in_features=W, out_features=W, bias=True),
            nn.Linear(in_features=W, out_features=W, bias=True),
            nn.Linear(in_features=W, out_features=W, bias=True),
            nn.Linear(in_features=W, out_features=W, bias=True),
            # 这里就是半路再次输入经过编码的位置坐标的地方，skip[4]的意思就是在这里插入
            nn.Linear(in_features=W + input_ch, out_features=W, bias=True),
            nn.Linear(in_features=W, out_features=W, bias=True),
            nn.Linear(in_features=W, out_features=W, bias=True)]
        )
        # 视角参数感知器，官方代码版，input_ch_views一般是27
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        # # 视角参数感知器，文章版
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        # 使用视角参数时，需要增加全连接层训练视角
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    # 定义前向传播，输入的x是经过位置编码的图片，具体可以DEBUG看看输入的通道数大小
    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            # 每经过一个全连接层后都用relu函数激活
            h = F.relu(h)
            if i in self.skips:
                # 再次加入位置编码信息
                h = torch.cat([input_pts, h], -1)
        # 训练视角
        if self.use_viewdirs:
            # 得到透明度也就是密度结果
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
            # 训练得到的rgb结果
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs


'''
    位置编码是论文中提出获取高频信息从而增强建模细节的方法
    简单理解就是对输入的输入(x,y,z,theta,phi)，其中视角一般用dx,dy,dz表示，进行编码
    对于位置：编码得到60D
    对于视角：编码得到24D
    这里实现一下傅里叶特征编码
'''

# 位置编码，位置编码的原理请看原文章介绍
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        # 输入的维度，一般是XYZ，3维度
        d = self.kwargs['input_dims']
        # 输出维度初始化为0
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            # 利用torch.linspace函数生成一个线性等间隔的序列，然后对这个序列进行2的幂运算
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                # 这里的红色报错不用在意，意思是这两个变量没有定义，实际使用时是有具体值的
                # 实在不想看到红色报警，那就将二者在函数定义开头显示声明一下
                # 这里p_fn其实就是sin和cos，看文章里的公式一下就能对应上
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                # 输出维度加d，d就是3，输入维度，因为是对x做操作，而x就是输入的XYZ坐标
                out_dim += d
        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


'''
    编码有位置编码与视角编码，两种编码有点小区别
    定义一个get_embedder函数
    返回一个lambda函数与编码结果的维度
    lambda函数的作用就是将输入进行编码，方便后面重复使用
'''


def get_embedder(multires, i=0):
    # 判断是否进行位置编码，这段代码可以删除，但是保留它可以测试下不做编码的效果
    if i == -1:
        return nn.Identity(), 3
    # 进行位置编码
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,  # 减1是因为一会是从0开始计数
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


'''
    NeRF训练时有粗网络和精细网络两个模型
    定义一个create_nerf函数，专门用来创建NeRF模型
'''

def create_nerf(args, device):

    # 获取位置编码函数与位置编码结果维度
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0  # 初始化输入相机视角维度为0，这里做初始化是因为不管是否做编码，都需要告诉模型输入的视角维度
    embeddirs_fn = None
    if args.use_viewdirs:
        # 获取视角编码函数
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    # 定义输出维度，根据是否进行精细采样决定输出维度
    # 如果进行则是5，反之4
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    # 获取粗网络模型参数
    grad_vars = list(model.parameters())

    model_fine = None
    # 判断是否需要进行精密采样，需要则再创建一个NeRF
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        # 获取精细网络参数
        grad_vars += list(model_fine.parameters())
    # 定义一个网络运行lambda函数，用来在渲染时直接调用模型进行训练
    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)
    # 使用Adam作为优化器
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999), capturable=True)
    # 设置起始迭代下标
    start = 0
    basedir = args.basedir
    expname = args.expname

    # 判断是否需要加载权重文件运算，即迭代过一定次数的结果，想继续迭代时
    if args.ft_path is not None and args.ft_path != 'None':
        # 指定权重文件进行迭代
        ckpts = [args.ft_path]
    else:
        # 检查输出文件夹下是否有权重文件，如果有则排序好用最新的进行训练
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)  # 提示是否找到了权重文件
    if len(ckpts) > 0 and not args.no_reload:
        # 加载预训练权重文件
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    # 用一个字典来存储训练时网络信息
    render_kwargs_train = {
        'network_query_fn': network_query_fn,  # 网络运行lambda函数
        'perturb': args.perturb,  # 光线抖动参数，能提高模型精度
        'N_importance': args.N_importance,  # 精密采样数量
        'network_fine': model_fine,  # 精密网络
        'N_samples': args.N_samples,  # 粗采样数量
        'network_fn': model,  # 粗网络
        'use_viewdirs': args.use_viewdirs,  # 是否使用视角
        'white_bkgd': args.white_bkgd,  # 使用使用白色作为背景
        'raw_noise_std': args.raw_noise_std,  # 噪声
    }

    # 提高LLFF-style格式的数据建模结果的方法
    # 作者暂时还没研究
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp
    # 再创建一个字典存储上面那个字典的信息，并且将perturb设为False，raw_noise_std设为0
    # 因为验证时不需要加光线抖动和噪声
    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    # 至此模型创建完成，将模型与其相关信息返回
    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


# 进行网络训练
def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    '''
    进行网络训练
    :param inputs: 输入维度
    :param viewdirs: 是否有视角参数
    :param fn: 模型
    :param embed_fn:位置编码
    :param embeddirs_fn:视角编码
    :param netchunk: 网络中一次最多处理的射线数
    :return: 模型训练输出
    '''
    # 将射线整形为N*3
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)  # 进行位置编码
    # 输入了视角
    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)  # 将视角格式整形为射线格式，即一条射线上粒子视角是相同的
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)  # 进行视角编码
        embedded = torch.cat([embedded, embedded_dirs], -1)  # 拼接位置编码与视角编码
    # 执行批处理，根据网络中一次最多处理的射数大小进行批处理，这里返回的是一个函数，用于切割射线
    outputs_flat = batchify(fn, netchunk)(embedded)  # 批处理进行网络训练
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])  # 将网络结果整形为输入射线格式
    return outputs


# 对输入网络的射线进行批处理
def batchify(fn, chunk):
    '''
    模型训练前，对输入模型的射线进行批次划分，使得GPU显存小时仍然可以训练
    :param fn: 模型
    :param chunk: 切片大小
    :return: 模型或者根据输入射线进行批次划分后的函数
    '''
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret