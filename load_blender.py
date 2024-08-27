import os
import json
import imageio
import numpy as np
import torch
import cv2


'''这三个lambda函数，主要是做平移和旋转操作'''
# 平移变换矩阵
trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()
# 沿y轴旋转矩阵
rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()
# 沿x轴旋转矩阵
rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    '''
    生成一个表示相机位姿的矩阵，用来将相机坐标系转换到世界坐标系，不明白建议了解下相机坐标转世界坐标原理
    :param theta: 方位角
    :param phi:俯仰角
    :param radius:半径
    :return:相机坐标转世界坐标矩阵
    '''
    # 平移至半径为 radius 的位置
    c2w = trans_t(radius)
    # 绕纵轴（Y轴）旋转 phi 角度
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    # 绕横轴（X轴）旋转 theta 角度
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    # 进行坐标系变换
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    '''
    加载blender数据
    :param basedir:文件地址，一般到场景文件夹，比如./data/nerf_synthetic/lego
    :param half_res:是否降采样
    :param testskip:加载数据步幅
    :return:数据的各类信息，请看return处对返回值的介绍
    '''
    # 用一个列表来存储'train', 'val', 'test'三个字符串，后面用来拼接生成各自的文件路径
    splits = ['train', 'val', 'test']
    # 用一个字典来存储相机参数信息
    metas = {}
    # 遍历读取训练、验证、测试样本的相机信息
    for s in splits:
        # 引入os库，使用os.path.join拼接字符串
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            # 引入json库读取json文件
            metas[s] = json.load(fp)
    # 声明一个图片列表来存储所有图片
    all_imgs = []
    # 声明一个相机姿态列表存储所有相机信息
    all_poses = []
    # 声明一个计数列表来统计三类样本各读取了多少
    counts = [0]
    # 遍历读取三类样本图片
    for s in splits:
        # 取出当前循环样本的相机参数信息
        meta = metas[s]
        # 声明一个临时存储图片的列表
        imgs = []
        # 声明一个临时存储相机信息的列表
        poses = []
        # 判断是否需要跳间隔取样本，训练样本全部取，其他根据传入的testskip参数
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip
        # 这里如果不理解，可以Debug该py看看meta的结构，或者看看流程中给出的json格式，skip主要就是控制间隔
        for frame in meta['frames'][::skip]:
            # 拼接字符串获取图片的路径
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            # 引入imageio库读取图片，如果这里imread被画线不用担心，这是该库版本更新问题
            # 可以使用imgs.append(imageio.v3.imread(fname))替代
            # 读入的图片是RGBA格式，所有最后一个维度大小是4
            imgs.append(imageio.imread(fname))
            # 引入numpy读取相机信息，存储为np数据方便后面做矩阵运算，取别名np是习惯，可以根据自己习惯修改
            poses.append(np.array(frame['transform_matrix']))
        # 至此该类型样本的图片和对应的相机信息已经读入

        # 同样将图片转换为np数组方便后面显卡运算，除以255.主要是做归一化，将像素值归一化到[0，1]。
        # 显卡运算时是float32，显示将np数组转换为该类型：.astype(np.float32)
        imgs = (np.array(imgs) / 255.).astype(np.float32)
        # 相机信息同样进行转换
        poses = np.array(poses).astype(np.float32)
        # 计数数组统计加载了多少张图片，例如100，那么就取列表最后一位数加100，起到分割all_imgs中图片的功能
        # 例如0,100,113,138
        counts.append(counts[-1] + imgs.shape[0])
        # 添加图片和相机信息到总列表中
        all_imgs.append(imgs)
        all_poses.append(poses)
    # 至此，所有图片和相机信息都读取到了np数组中，并且能够区分开

    # 生成一个列表，其中包含三个 NumPy 数组，这些数组由 counts 列表中的值定义的区间生成
    # 例如：[[0...99],[100...112],[113...137]]
    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]
    # 沿着0维度将所有图片和相机信息拼接起来并转换为np数组
    # 即第一个维度，[[100,800,800,4],[13,800,800,4]]->(113,800,800,4)
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    # 获取图片高宽
    H, W = imgs[0].shape[:2]
    # 获取相机视场角，即camera_angle_x，这里不理解可以查阅一下什么是视场角
    camera_angle_x = float(meta['camera_angle_x'])
    # 利用视场角计算相机焦距f，公式可以自行查阅
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    # 这里是生成一组渲染视角，个人理解就是用来从不同角度渲染场景
    # 这里是直接引入torch将其存放到torch中，因为后续渲染时在GPU上做
    # 不理解其用途可以暂时跳过，到后面进行渲染使用到的时候再回来思考是怎么计算的
    # 这里需要定义一个生成一个表示相机位姿的矩阵或向量pose_spherical，其定义在开头位置
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)
    # 判断是否进行下采样到一半
    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.
        # 创建一个降采样后大小的np数组
        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        # 遍历进行降采样
        for i, img in enumerate(imgs):
            # 利用cv2的降采样函数进行降采样，interpolation=cv2.INTER_AREA是区域插值法
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
    # 至此数据加载并预处理成功，返回需要的参数
    # imgs：所有图片
    # poses：图片对应相机参数
    # render_poses：渲染时的相机参数
    # [H, W, focal]：图片高、宽、焦距
    # i_split：分割训练、验证、测试集下标的列表
    return imgs, poses, render_poses, [H, W, focal], i_split


# 可以直接Debug该文件，看一看是如何加载数据的
# 传入的half_res和testskip参数到底有什么用
# 注意文件路径一定要正确
if __name__=='__main__':
    load_blender_data(basedir="./data/nerf_synthetic/chair", half_res=True, testskip=8)
