import torch.nn as nn
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torchvision
import torch.nn.functional as F
import os
import time
from torch import optim
from sklearn import metrics	
import numpy as np
import torch.backends.cudnn as cudnn
import cv2
# from hlindex import HolisticIndexBlock, DepthwiseO2OIndexBlock, DepthwiseM2OIndexBlock
from torch.optim.lr_scheduler import MultiStepLR
import argparse
# from carafe import CARAFEPack
# from skimage.measure import compare_psnr
# from skimage.measure import compare_ssim
# from skimage.measure import compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
# 启用CuDNN（CUDA Deep Neural Network library）加速。
cudnn.enabled = True

# 添加了三个命令行参数的定义：
# 1--upmode 用于指定实现上采样的模式。默认值是'nn'，表示使用最近邻插值法进行上采样
# 2--description 用于指定描述信息的路径 表示描述信息的路径为当前目录下的'0130'文件夹。
# 3--evaluate 用于指定模型的路径 表示模型路径为'experiments/0130/nn/model_best.pth.tar'
# 解析命令行参数，并返回一个包含解析结果的命名空间对象。
def get_arguments():

	parser = argparse.ArgumentParser(description="RecoverNet")

	parser.add_argument("--upmode", type=str, default='nn', help="the mode chosen to implement upsample.") # 'bilinear', 'maxpool', 'nn'. 'indexnet-m2o', ...
	parser.add_argument("--description", type=str, default='./0624-P4/', help="description.")
	parser.add_argument("--evaluate", type=str, default='experiments/0624-P4/nn/model_best.pth.tar', help="path of model.")

	return parser.parse_args()

# 输入参数为x和block_size，其中x是一个张量（tensor），block_size是一个整数，表示块的大小。
# 其中n表示批次大小（batch size），c表示通道数（channel），h表示高度（height），w表示宽度（width）。
# 使用F.unfold()函数对输入张量x进行展开操作，展开的块大小为block_size，步长为block_size。这将把输入张量的每个块展开为一列，并按照展开的顺序排列。
# 使用unfolded_x.view()函数对展开后的张量进行形状变换，将其重新变为原始形状。
# 其中，n保持不变，c乘以block_size的平方，表示每个块的展开后的通道数，h和w分别除以block_size，表示每个块的展开后的高度和宽度。
# 返回变换后的张量。
def space_to_depth(x, block_size):
	n, c, h, w = x.size()
	unfolded_x = F.unfold(x, block_size, stride=block_size)
	return unfolded_x.view(n, c*block_size**2, h//block_size, w//block_size)

class Bottleneck(nn.Module):  # ResNet  Bottleneck
    expansion = 4  # 通道倍增数
 
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_planes, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, stride, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, self.expansion*planes, 1, bias=False),
            nn.BatchNorm2d(self.expansion*planes),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
 
    def forward(self, x):
        identity = x
        out = self.bottleneck(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity  # shortcut
        out = self.relu(out)
        return out
 
 
class FPN(nn.Module):
    def __init__(self, layers):
        super(FPN, self).__init__()
        self.inplanes = 64
        # 处理输入的C1模块（C1代表了RestNet的前几个零散的卷积与池化层）
        self.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        # 搭建自下而上的C2，C3，C4，C5
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], 2)
        self.layer3 = self._make_layer(256, layers[2], 2)
        self.layer4 = self._make_layer(512, layers[3], 2)
        # 对C5 conv，减少通道数，得到P5 # C5 ==> 1x1卷积 ==> P5
        self.toplayer = nn.Conv2d(2048, 256, 1, 1, 0)
        # 3x3卷积 融合特征
        self.smooth1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth3 = nn.Conv2d(256, 1, 3, 1, 1)
        self.smooth4 = nn.Conv2d(256, 1, 3, 1, 1)
        # 横向连接，需要保证通道数相同
        self.latlayer1 = nn.Conv2d(1024, 256, 1, 1, 0)
        self.latlayer2 = nn.Conv2d(512, 256, 1, 1, 0)
        self.latlayer3 = nn.Conv2d(256, 256, 1, 1, 0)
 
    def _make_layer(self, planes, blocks, stride=1):  # C2，C3，C4，C5
        downsample = None
        if stride != 1 or self.inplanes != Bottleneck.expansion*planes:
            downsample = nn.Sequential(  # 统一输出
                nn.Conv2d(self.inplanes, Bottleneck.expansion * \
                          planes, 1, stride, bias=False),
                nn.BatchNorm2d(Bottleneck.expansion*planes)
            )  # 是否上采样
        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)
 
    def _upsample_add(self, x, y):  # 自上而下的采样模块 P5 ==> P4 ==> P3 ==> P2
        _, _, H, W = y.shape  # 获取 H W
        # return F.upsample(x, size=(H, W), mode='bilinear') + y  # 双线性插值采样后 相加
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True) + y # pytorch版本问题 去除报错
 
    def forward(self, x):
        # 自下而上
        c1 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # 自上而下
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4)) # 先对c4进行1x1conv
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # 卷积的融合，平滑处理
        p4 = self.smooth1(p4) # 3x3
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2) 
        p4 = self.smooth4(p4)
	    #p2 = F.interpolate(p2, size=(32, 32), mode='bilinear', align_corners=False)
        #p2 = F.interpolate(p2, size=(32,32),mode='bilinear',align_corners=False)
        p4 = F.interpolate(p4, size=(32,32),mode='bilinear',align_corners=False)
        return p4
         

#train model
def train(model, train_loader, optimizer, criterion):
	running_loss = 0	# 用于记录每个epoch的累计损失
	model.train()		# 将模型设置为训练模式
	cudnn.benchmark = True
	# 迭代训练数据加载器中的批次数据
	for i, (data, target) in enumerate(train_loader):
		data = data.cuda()
		optimizer.zero_grad()	# 将优化器的梯度缓存清零
		output = model(data)	# 通过模型前向传播得到输出
		target = data.clone()	# 克隆数据作为目标
		target = target * 0.3081 + 0.1307	# 对目标数据进行标准化处理
		loss = criterion(output, target)	# 计算损失值
		loss.backward()		# 进行反向传播计算梯度
		optimizer.step()	# 更新模型的参数
		running_loss += loss.item()		# 累计当前批次的损失值
	# 将每个epoch的平均损失值存储在模型的train_loss字典中的epoch_loss列表中
	model.train_loss['epoch_loss'].append(running_loss / (i + 1))	


# calculate loss
def val(epoch, model, val_loader, criterion, save_image_dir):
	model.eval()	# 评估模式
	cudnn.benchmark = False		# 并禁用cudnn.benchmark以节省内存
	val_loss = 0
	psnr = []
	ssim = []
	mse = []
	mae = []
	# 创建一个保存当前epoch图像的目录
	save_image_dir_epoch = save_image_dir + 'epoch{}/'.format(epoch)
	if not os.path.exists(save_image_dir_epoch):
		os.makedirs(save_image_dir_epoch)
	# 使用torch.no_grad()上下文管理器，禁用梯度计算。
	# 在验证数据加载器中迭代批次数据，每个批次的数据移动到GPU上
	with torch.no_grad():
		for i, (data, _) in enumerate(val_loader):
			data = data.cuda()
			output = model(data)	# 通过模型前向传播得到输出
			target = data.clone()	# 克隆数据作为目标
			target = target * 0.3081 + 0.1307	# 进行标准化处理
			val_loss += criterion(output, target).item() # sum up batch loss
			# 计算验证损失值,并将其累加到val_loss中

			# PSNR（峰值信噪比）
			# SSIM（结构相似度指数）
			# MSE（均方误差）
			# MAE（平均绝对误差）
			# psnr.append(psnr_value)用于将计算得到的psnr_value添加到psnr列表的末尾
			# squeeze()去除维度为1的维度,cpu()将张量从GPU内存中移动到CPU内存
			# numpy()将PyTorch张量转换为NumPy数组的方法
			# astype(np.float32)将数组的数据类型转换为指定的数据类型单精度浮点型
			# 将PyTorch张量经过挤压、从GPU移回CPU，并转换为NumPy数组，并且将其数据类型转换为单精度浮点型
			# 为data_range参数传递图像数据的动态范围。通常情况下，取值范围为图像数据类型的最大值减去最小值。
			psnr.append(compare_psnr(target.squeeze().cpu().numpy().astype(np.float32), output.squeeze().cpu().numpy().astype(np.float32)))
			ssim.append(compare_ssim(target.squeeze().cpu().numpy().astype(np.float32), output.squeeze().cpu().numpy().astype(np.float32),data_range=32))
			mse.append(np.sqrt(compare_mse(target.squeeze().cpu().numpy().astype(np.float32), output.squeeze().cpu().numpy().astype(np.float32))))
			mae.append(criterion(output, target).item())

			# 如果需要保存图像，可以取消代码片段中的注释，并将输出和目标图像保存到指定的目录中。

			if i < 200:
				outputs_save = np.clip(output[0].cpu().numpy(), 0, 1) * 255
				target_save = np.clip(target[0].cpu().numpy(), 0, 1) * 255
				cv2.imwrite(save_image_dir_epoch + 'epoch{}_{}.jpg'.format(epoch, i), outputs_save.squeeze().astype(np.uint8))
				cv2.imwrite(save_image_dir_epoch + 'epoch{}_{}_gt.jpg'.format(epoch, i), target_save.squeeze().astype(np.uint8))

	val_loss /= (i + 1)
	# 将计算得到的验证损失和评估指标存储在模型的val_loss字典和measure字典中的相应列表中。
	model.val_loss['epoch_loss'].append(val_loss)
	model.measure['psnr'].append(np.mean(psnr))
	model.measure['ssim'].append(np.mean(ssim))
	model.measure['mse'].append(np.mean(mse))
	model.measure['mae'].append(np.mean(mae))


def save_checkpoint(state, snapshot_dir, filename='.pth.tar'):
	torch.save(state, '{}/{}'.format(snapshot_dir, filename))

def main():
	batchsize = 100
	test_batchsize = 80
	epoches = 100
	args = get_arguments()	# 获取命令行参数
	image_size = 32

	# 标准化操作使用了均值(0.1307,)和标准差(0.3081,)进行归一化处理。
	transform = transforms.Compose([transforms.Resize(image_size),
									transforms.ToTensor(),
									transforms.Normalize((0.1307,), (0.3081,)), ])
	# train=True表示创建训练集
	trainset = torchvision.datasets.FashionMNIST(root='./data/fashionmnist', train=True, download=True,
												 transform=transform)
	# batch_size参数指定每个批次的样本数量，shuffle=True表示在每个周期开始前对数据进行洗牌，num_workers参数指定数据加载的线程数。
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)
	# train=False表示创建测试集，
	testset = torchvision.datasets.FashionMNIST(root='./data/fashionmnist', train=False, download=True,
												transform=transform)
	# batch_size参数指定每个批次的样本数量，shuffle=False表示不对数据进行洗牌，num_workers参数指定数据加载的线程数。
	testloader = torch.utils.data.DataLoader(testset, batch_size=test_batchsize, shuffle=False, num_workers=2)

	# save info of model
	trained_model_dir = 'experiments/' + args.description + '{}/'.format(args.upmode)
	train_info_record = trained_model_dir + args.upmode + '.txt'
	save_image_dir = trained_model_dir + 'results/'

	if not os.path.exists(trained_model_dir):
		os.makedirs(trained_model_dir)

	if not os.path.exists(save_image_dir):
		os.makedirs(save_image_dir)
	# 升采样模式
	net=FPN([2,2,2,2])
	net.cuda()
	# 使用了SGD优化器,指定学习率lr=0.01和动量momentum=0.9。
	optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
	# 创建损失函数对象,使用了L1损失函数
	criterion = nn.L1Loss()

	net.train_loss = {
		'epoch_loss': []
	}
	net.val_loss = {
		'epoch_loss': []
	}
	net.measure = {
		'psnr': [],
		'ssim': [],
		'mse': [],
		'mae': []
	}

	if os.path.isfile(args.evaluate):
		checkpoint = torch.load(args.evaluate)
		net.load_state_dict(checkpoint['state_dict'])	# 将加载的模型参数加载到网络模型net中。
		epoch = checkpoint['epoch']
		val(epoch, net, testloader, criterion, save_image_dir)
		print(' sample: %d psnr: %.2f%%  ssim: %.4f  mse: %0.6f mae: %0.6f' % (
			len(testloader.dataset), net.measure['psnr'][-1], net.measure['ssim'][-1], net.measure['mse'][-1], net.measure['mae'][-1]))
		return

	# train begin
	print('training begin')
	# 调度器来设置学习率的调整策略
	scheduler = MultiStepLR(optimizer, milestones=[50, 70, 85], gamma=0.1)
	for epoch in range(epoches):
		scheduler.step()	# 更新学习率
		start = time.time()
		train(net, trainloader, optimizer, criterion)
		end = time.time()
		print('epoch: %d sample: %d cost %.5f seconds  loss: %.5f' % (
		epoch, len(trainloader.dataset), (end - start), net.train_loss['epoch_loss'][-1]))

		val(epoch, net, testloader, criterion, save_image_dir)
		print(' sample: %d test_loss: %.5f psnr: %.2f%%  ssim: %.4f  mse: %0.6f mae: %0.6f' % (
		    len(testloader.dataset), net.val_loss['epoch_loss'][-1], net.measure['psnr'][-1], net.measure['ssim'][-1], net.measure['mse'][-1], net.measure['mae'][-1]))
		print
		# save checkpoint
		state = {
			'state_dict': net.state_dict(),
			'optimizer': optimizer.state_dict(),
			'epoch': epoch + 1,
			'train_loss': net.train_loss,
			'val_loss': net.val_loss,
			'measure': net.measure
		}
		# save model
		save_checkpoint(state, trained_model_dir, filename='model_ckpt.pth.tar')
		# 如果当前轮数的PSNR指标比之前的轮数都要好,保存最佳模型
		if len(net.measure['psnr']) > 1 and net.measure['psnr'][-1] >= max(net.measure['psnr'][:-1]):
			save_checkpoint(state, trained_model_dir, filename='model_best.pth.tar')

		with open(train_info_record, 'a') as f:
			f.write(
				'lr:{}, epoch:{}, train_loss:{:.4f}, val_loss:{:.6f}, psnr:{:.2f}, ssim:{:.2f}, mse: {:0.6f} mae: {:0.6f}'.format(
					optimizer.param_groups[0]['lr'], epoch, net.train_loss['epoch_loss'][-1],
					net.val_loss['epoch_loss'][-1], net.measure['psnr'][-1], net.measure['ssim'][-1], net.measure['mse'][-1], net.measure['mae'][-1]) + '\n'
			)

if __name__ == "__main__":
	main()