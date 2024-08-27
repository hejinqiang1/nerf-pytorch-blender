import torch
import numpy as np
from opts import config_parser
from load_blender import load_blender_data
import os
from model import create_nerf
import imageio
from render import render_path, render
from render import to8b, img2mse, mse2psnr
from render import get_rays_np, get_rays
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import time


'''
    作者也是首次学习NeRF，有些地方可能描述有误，欢迎给作者指出问题
    作为给初学者学习使用，注释内容比较多，格式可能不太规范，敬请谅解！
'''

'''
    这里主要是设置GPU相关参数
'''

# 引入torch库
# 设置设备类型，nerf一般是用cuda跑
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 如果电脑有多台GPU，请注意一定要确保所有数据在同一台GPU上
# torch.cuda.set_device(device)
# 设置tensor数据类型，一般是torch.cuda.FloatTensor
torch.set_default_tensor_type('torch.cuda.FloatTensor')
# 使用np生成随机数种子，便于体渲染中添加扰动时重复使用相同数字
np.random.seed(0)
'''
    首先看train函数，也是该py唯一的一个函数
    它将训练的整个流程实现
    参数通过控制台config传入，格式：--config configs/lego.txt
    若对参数传入不理解，可以看该链接：
    https://blog.csdn.net/qq_41813454/article/details/136224020?ops_request_misc=&request_id=&biz_id=102&utm_term=argparse&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-136224020.142^v100^pc_search_result_base4&spm=1018.2226.3001.4187
'''


def train():
    # 设置参数，创建一个opts.py文件，里面定义一个config_parser函数
    # 这里可以跳转到opts.py中看一下，模型训练需要哪些参数
    parser = config_parser()
    # 利用args存储所有参数，这里可以Debug看args参数有哪些
    args = parser.parse_args()

    # 加载数据，只以blender为例，其他类型数据的加载请参考github官方的实现
    if args.dataset_type == 'blender':
        # 加载blender数据，创建要给load_blender.py文件，里面定义一个load_blender_data函数用来加载数据
        # 这里可以跳转到opts.py中看一下，主要是将data文件加下的图片数据按一定格式加载进来
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        # 将加载结果打印到控制台，便于观察数据加载情况
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        # 将i_split存储的训练、验证、测试样本的下标列表用三个列表来存放，便于后续使用
        i_train, i_val, i_test = i_split
        # 定义nerf沿像素射线进行体素采样的距离，一般默认
        near = 2.
        far = 6.
        # 判断参数中white_bkgd是否为true，即背景颜色，默认是true
        if args.white_bkgd:
            # images[..., :3]：这部分表示取images的前3个通道，即RGB通道。其形状为138 * 400 * 400 * 3。
            # images[..., -1:]：这部分表示取images的最后一个通道，即透明度通道。其形状为138 * 400 * 400 * 1。
            # images[..., :3] * images[..., -1:]：这部分表示将RGB通道的每个像素值乘以相应的透明度值。
            # 透明度值的范围通常是[0, 1]，这一步将每个RGB通道的颜色值按透明度进行缩放。
            # (1. - images[..., -1:])：这部分表示透明度的补值，即1 - alpha。其形状仍为138 * 400 * 400 * 1。
            # 整体加和：最终的操作是将经过透明度调整的RGB值加上透明度的补值。
            # 此时images格式为：138*400*400*3，只剩下RGB通道
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return
    # 至此，数据加载成功

    # 做创建模型前的准备工作
    H, W, focal = hwf
    # 获取相机内参矩阵
    # 原理建议查阅相机坐标与世界坐标的转换过程
    K = None
    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    # 判断是否在测试集上进行渲染，一般用来评估模型，即已经训练好模型
    if args.render_test:
        # 取出测试集的相机信息
        render_poses = np.array(poses[i_test])

    # 创建log文件夹，即你定义的存储结果的文件夹，并且将训练的cofig文件保存，便于复现
    basedir = args.basedir
    expname = args.expname
    # 利用os库创建文件夹exist_ok=True是用于存在该文件时不抛出异常
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    # 存储训练参数txt文件与config.txt文件
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())


    # 至此准备工作做完
    # 开始创建模型
    # 创建一个model.py文件，里面实现模型及其相关的函数
    # 模型创建需要两个参数，一个是args，一个是device，传入device主要是为了明确在哪个GPU上运行
    # 避免出现数据加载到不同GPU上的情况
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args, device)

    # 定义迭代起始下标，方便后面记录信息
    global_step = start
    # 用一个字典来存放射线上采样的范围
    bds_dict = {
        'near': near,
        'far': far,
    }
    # 将这两个信息加入到训练和测试参数表中
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    # 因为上面只做渲染时，上面是更新render_poses的值，这里显示确保数据移动到指定的GPU中了
    render_poses = torch.Tensor(render_poses).to(device)

    '''
        现在模型的定义做好了，图片也读取进来了
        模型训练前只需要再写好渲染相关操作就可以了
        简单来说就是两部分：
        1.从图片上获取射线
        2.将模型的预测结果再渲染为图片
        这里创建render.py文件，在里面实现渲染相关的函数
    '''

    # 仅进行渲染，则直接将渲染结果输出，不进行训练过程
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                images = images[i_test]
            else:
                images = None
            # 测试集渲染结果保存路径
            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)
            # 在render.py中定义一个render_path函数，在仅渲染模式下输出渲染结果
            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            # 利用imageio库的mimwrite将视频结果导出，参数含义查阅该函数用法
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=60, quality=8)
            return
    # 一次选取的光线数量
    N_rand = args.N_rand
    # 是否采用batching操作，即是否在每次训练迭代中只从一张图片中随机采样一批光线进行训练
    use_batching = not args.no_batching
    if use_batching:
        # 在所有图片中进行随机光线采样
        print('get rays')
        # 这里定义一个从图片像素获取射线的np版本函数
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:, None]], 1)  # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)  # train images only
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])  # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)
        print('done')
        i_batch = 0
    if use_batching:
        # 将所有图像加载到GPU上
        images = torch.Tensor(images).to(device)
    # 将所有相机信息加载到GPU上
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        # 将光线加载到GPU上
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    # 定义迭代次数，加1是让输出的文件名称更好看
    N_iters = args.N_iters + 1
    """打印信息提示使用者开始训练"""
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # 使用tensorboard记录信息
    writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    # 这里加1是因为N_iters加了个1
    start = start + 1
    # 开始迭代训练
    for i in trange(start, N_iters):
        time0 = time.time()
        # 从所有图片上随机采样
        if use_batching:
            # 随机覆盖所有图像
            batch = rays_rgb[i_batch:i_batch+N_rand]  # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]
            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0
        # 从一张图片上采样
        else:
            # 利用random生成随机数，范围为训练数据的下标
            img_i = np.random.choice(i_train)
            # 取出对应图片
            target = images[img_i]
            # 将图片加载到GPU上
            target = torch.Tensor(target).to(device)
            # 取出对应的相机信息
            pose = poses[img_i, :3, :4]
            # 根据N_rand选取光线
            if N_rand is not None:
                # 获取图片光线
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
                # 根据裁剪策略参数决定是否只对图像中心区域进行采样
                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    # 生成一个中心区域的网格
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    # 如果做了这个操作，第一次迭代时提醒使用者这个操作会一直做到多少次迭代
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
                # 调整coords的形状，可以DEBUG看一下
                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                # 根据一次最大光线数在coords中选取光线，这里的一些变换操作都是为了方便从光线中取出对应位置的光线
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0) # 这里将光线方向和原点堆叠在一起方便后面传入体渲染
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        # 进行体渲染，流程比较复杂，建议先弄明白体渲染原理
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays, verbose=i < 10, retraw=True,
                                        **render_kwargs_train)
        # 优化器梯度置零，这个操作是因为pytorch平台会累积每次迭代的损失，所以每次迭代都需要置零
        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)  # 计算选取的射线预测出的图像RGB与对应位置真实图像的RGB的mse均方误差损失
        # trans = extras['raw'][..., -1]
        loss = img_loss
        psnr = mse2psnr(img_loss)   # 计算选取的射线预测出的图像RGB与对应位置真实图像的RGB的psnr损失
        # 计算粗网络的损失，便于研究
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)
        # 反向传播
        loss.backward()
        optimizer.step()  # 优化器更新权重
        """更新学习率"""
        # 每经过一个完整的衰减周期，学习率将变为原来的 10%
        decay_rate = 0.1
        # 控制学习率衰减的频率
        decay_steps = args.lrate_decay * 1000
        # 计算下一次训练的学习率，global_step是当前的训练步数，每进行一个批次（batch）的训练，global_step 增加一次
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        # 更新优化器中的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        # 计算下本次训练用时
        dt = time.time() - time0

        # 判断当前迭代次数是否需要保存权重文件
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)
        # 判断当前迭代次数是否需要输出视频结果
        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)
        # 判断当前迭代次数是否需要用测试集评估模型精度
        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')
        # 判断当前迭代次数是否需要打印损失信息
        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
            writer.add_scalar('train_Loss', loss, i // args.i_print)
            writer.add_scalar('train_PSNR', loss, i // args.i_print)

        # 这是对验证数据进行生成图片的操作，可以进行，但训练的时间会延长很多
        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        global_step += 1


if __name__=='__main__':
    train()
