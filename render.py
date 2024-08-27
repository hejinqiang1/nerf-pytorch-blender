import time
import numpy as np
from tqdm import tqdm
import os
import imageio
import torch
import torch.nn.functional as F


# 这个参数用来控制是否输出一些中间信息用来供使用者调试
DEBUG = False

'''
    这几个lambda函数是后面才加的
    是对渲染结果进行进一步处理时用到的
    to8b是在生成视频文件时对输入的RGB进行拉伸和格式转换操作
    img2mse是计算预测图片与真实图片的mse，mse是模型训练的损失，用来更新权重的
    mse2psnr是计算预测图片与真实图片的psnr指标，用来反映模型的精度
'''
# 将像素转换为8byte格式
to8b = lambda x: (255*np.clip(x, 0, 1)).astype(np.uint8)
# 获取图片mse指标
img2mse = lambda x, y: torch.mean((x - y) ** 2)
# 从mse获取图片psnr指标
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    '''
    仅进行渲染操作，一般是验证模型精度时使用
    :param render_poses: 输入图片的相机参数
    :param hwf: 图片高宽焦距
    :param K: 相机内参矩阵
    :param chunk: 一次处理射线数量
    :param render_kwargs: 存储验证时参数的字典
    :param gt_imgs: 图片
    :param savedir: 保存路径
    :param render_factor: 渲染缩放因子
    :return:
    '''
    H, W, focal = hwf
    # 判断渲染是否需要缩放，一般不缩放，因为读取数据时已经对图片做了降采样
    if render_factor != 0:
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor
    # 存储渲染颜色结果
    rgbs = []
    # 存储视察图
    disps = []
    # 记录下渲染时间
    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        # 定义一个render函数实现渲染操作
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True, near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    '''
    体渲染操作
    :param H:高
    :param W:宽
    :param K:焦距
    :param chunk:一次性最多处理的射线
    :param rays:光线
    :param c2w:相机到世界坐标系的转换矩阵
    :param ndc:llff数据采用这种方式效能获得更好的效果，默认为true
    :param near:粒子采样范围，近点距离，一般是2，默认值是0
    :param far:粒子采样范围，远点距离，一般是6，默认值是1
    :param use_viewdirs:是否使用视角参数
    :param c2w_staticcam:控制静态相机的世界坐标到相机坐标的转换矩阵
    :param kwargs:其余所有参数
    :return:rgb_map: [batch_size, 3]光线的RGB预测结果 disp_map: [batch_size]. 视差图，深度的倒数 extras: render_rays()函数返回的字典信息
    '''
    if c2w is not None:  # 已经做好了光线的选取，这里就没有传入c2w矩阵，如果是渲染测试数据和验证数据则会在这里选取光线
        # 定义一个get_rays函数从图片上选取光线
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        rays_o, rays_d = rays
    # 使用视角
    if use_viewdirs:
        # 将视角theta、phi用dx、dy、dz来表示
        viewdirs = rays_d
        # 在有些场景中，相机可能是静止的，而场景是动态的。例如，一个静止的相机拍摄一个移动的物体，这时相机的位姿是固定的，而物体的位置和朝向在变化
        # c2w_staticcam参数可以用于描述这种静止相机的情况
        if c2w_staticcam is not None:
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)  # 归一化操作
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()  # 调整形状

    sh = rays_d.shape  # [..., 3]
    # 这里作者暂时没研究这种操作，不做解释，复制原版pytorch中的ndc_rays函数
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # 这里将射线格式转换为：batch*3
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()
    # 定义射线的近裁剪面和远裁剪面，也就是公式里的距离*方向
    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)  # 拼接得到最终的射线这里是8维度
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)  # 使用视角时，将视角也拼接起来，这里是11维度

    # 进行渲染并整形，chunk是控制一次最多处理的光线数，从而使得能在小显存设备上运行
    # 定义一个batchify_rays来将射线划分批次并进行渲染
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def get_rays(H, W, K, c2w):
    '''
    从像素点获取光线，r(t)=o+td；光线可以用原点+距离*方向来表示，rays_o就是原点，rays_d就是方向
    :param H:高
    :param W:宽
    :param K:焦距
    :param c2w:相机到世界坐标系的转换矩阵
    :return:原点坐标和方向
    '''
    # torch.meshgrid 函数用于生成二维网格坐标，用于将每个像素的坐标映射到图像平面上
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    # torch.meshgrid 的输出顺序是 (Y, X)，所以需要交换顺序
    i = i.t()
    j = j.t()
    # 将图像坐标转换为光线方向dirs就是公式中的d
    # 这里翻译一下K的值
    # k[0][2]=w/2,K[0][0]=f,K[1][2]=H/2,K[1][1]=F
    # i、j对应的是像素平面，转换到成像平面，即原点由左上角转移到中心
    # 除以焦距f，即从物理成像平面转移到相机坐标系，原理就是相机坐标系与物理成像平面的距离就是焦距f
    # 这里可以先看一下imgs里提供的相机针孔成像模型图
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # 将相机坐标系转移到世界坐标系，即用c2w变换矩阵，这里将dirs扩展为H*W*1*3与取出的c2w的3*3矩阵相乘，然后沿着最后一个维度求和
    # rays_d结果为H*W*3
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # 确定光线的起点
    # 起点通常是相机的位置。在世界坐标系中，相机的位置可以通过相机到世界的变换矩阵获得
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    '''
    将选出的射线按一次参与计算的射线数进行批次划分并传入render_rays进行后续操作
    :param rays_flat:从图片上选取的射线
    :param chunk:一次参与计算的射线数
    :param kwargs:其它相关参数
    :return:渲染结果
    '''
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        # 将选出的射线进行渲染，定义一个render_ray函数
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)  # 真正的选取粒子的函数
        for k in ret:  # 将结果转换为列表
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    # 将结果转换为字典
    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render_rays(ray_batch, network_fn, network_query_fn, N_samples, retraw=False, lindisp=False, perturb=0.,
                N_importance=0, network_fine=None, white_bkgd=False, raw_noise_std=0., verbose=False, pytest=False):
    '''
    在射线上选取粒子，然后输入模型进行训练
    :param ray_batch: 传入的射线
    :param network_fn: 构建的粗NeRF网络
    :param network_query_fn: model中的一个函数，用来将采样点的位置、方向以及其他相关信息输入到神经网络中，获取用于体积渲染的颜色和密度值
    :param N_samples: 粗网络粒子采样数量
    :param retraw: 是否保留精细网络的原始预测输出
    :param lindisp: 以逆深度还是深度进行采样
    :param perturb: 扰动
    :param N_importance: 精细网络粒子采样数量
    :param network_fine: 精细网络
    :param white_bkgd: 是否采用白色背景
    :param raw_noise_std:噪声
    :param verbose:打印更多调试信息
    :param pytest:
    :return:[num_rays, 3]. 来自精细网络的射线RGB颜色 [num_rays]. 视差图 1 / depth. [num_rays]. 来自精细网路的沿每条光线累计的不透明度
    raw: [num_rays, num_samples, 4]. 模型的原始预测 rgb0: See rgb_map. 粗网络的输出 rgb0: See rgb_map. 粗网络的输出
    acc0: See acc_map. 粗网络的输出 z_std: [num_rays]. 每个样本沿射线距离的标准偏差
    '''
    N_rays = ray_batch.shape[0]
    # [N_rays, 3] each 这里是把射线原点与方向取出来，可以看看render中是如何堆叠这个变量的
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    # 这里是取出near和far
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]
    # 定义粗采样，在范围内均匀产生64个点
    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        # 采样结果，就是映射到near-far之间，即2-6。可以DEBUG对比下变量的值
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        # 这是逆深度采样，作者暂时没研究
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
    # 将取样结果应用到所有的射线
    z_vals = z_vals.expand([N_rays, N_samples])
    # 添加扰动，增加鲁棒性
    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # 用numpy的固定随机数覆盖u，也就是为什么在tran_nerf开头需要生成一个随机数种子
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand
    # 这里就是公式：r(t)=原点+距离*方向的体现，t是距离，至此射线的粗采样完成，pts：[N_rays, N_samples, 3]
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    # 获取粗网络结果
    raw = network_query_fn(pts, viewdirs, network_fn)
    # 将获取到的结果转换为需要的信息，定义一个raw2outputs
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
    # 进行精密采样
    if N_importance > 0:
        # 保存下粗网络输出结果
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
        # 取每对相邻深度值的平均值，得到这些采样点之间的中间位置
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        # 定义一个sample_pdf函数进行精密采样
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
        z_samples = z_samples.detach()  # 这个操作是计算图中分离张量，使得该张量在后续的计算中不会影响梯度计算
        # 拼接粗采样和精细采样的所有粒子
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        # [N_rays, N_samples + N_importance, 3]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        # 判断有没有精密网络，没有就把用粗网络来训练，但两个网络的输入通道不同，肯定会报错
        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)  # 获取精细网络结果
        # 对精细网络的结果进行体渲染，得到最终的渲染结果
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
    # 在DEBUG模式下这里会输出各种参数结果，使使用者更清楚训练过程
    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    '''
    将模型预测结果转为需要的RGB颜色，也就是所谓的进行体渲染
    :param raw:[num_rays, num_samples along ray, 4]. 模型预测结果
    :param z_vals:[num_rays, num_samples along ray]. 距离
    :param rays_d:[num_rays, 3]. 方向
    :param raw_noise_std:噪声
    :param white_bkgd:是否以白色为背景
    :param pytest:
    :return:rgb_map: [num_rays, 3]. 光线的RGB颜色 disp_map: [num_rays]. 视差图 acc_map: [num_rays]. 每条射线的权重综合
    weights: [num_rays, num_samples]. 分配给每种采样颜色的权重 depth_map: [num_rays]. 估计到对象的距离
    '''

    # 定义一个匿名函数，用于体渲染中的不透明度计算
    # 计算过程如下：
    # 首先对 raw 应用激活函数 act_fn（默认是 ReLU），将其变为非负值
    # 将激活后的值乘以相邻采样点之间的距离 dists。这一步考虑了每个采样点对光线的贡献。
    # 计算负指数函数。对于体渲染来说，这一步有助于计算累积的不透明度。
    # 取 1 减去指数函数的结果，得到最终的不透明度（alpha 值）。这表示光线在通过体积时被吸收的程度。
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # [N_rays, N_samples] 计算相邻采样点间的距离
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)
    # 将距离从归一化空间（如深度空间）转换到实际的三维空间距离
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    # [N_rays, N_samples, 3] 将神经网络的输出激活为颜色值
    rgb = torch.sigmoid(raw[..., :3])
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)
    # 计算不透明度
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True) 计算每条射线的颜色权重，建议看体渲染中颜色求和的公式推到
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
    # 估计到对象的距离
    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)
    # 使用白色作为背景
    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    '''
    精密采样
    :param bins: 粗采样点之间的中间位置
    :param weights: 每条射线的颜色权重
    :param N_samples: 采样数
    :param det: 是否有扰动
    :param pytest:
    :return:
    '''

    # 由于权重有可能值为nans，为了避免计算错误，全部加上一个1e-5
    weights = weights + 1e-5
    # 对权重进行归一化得到一个概率密度函数PDF
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    # 计算一个概率密度函数PDF在每个位置的累积概率，用于生成累积分布函数CDF
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)

    # 是否取均匀样本
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # 测试用，用numpy的固定随机数覆盖U
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # 反转CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples


# 这个操作与get_rays同原理，只是在np数组下进行
def get_rays_np(H, W, K, c2w):
    # 创建网格坐标
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    # 计算光线方向
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # 将光线方向从相机坐标系旋转到世界坐标系
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # 将相机坐标系的原点平移到世界坐标系中，作为所有光线的起点
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d