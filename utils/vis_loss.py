import re
import matplotlib.pyplot as plt


# 从文件中读取数据
def extract_train_loss(filename):
    with open(filename, 'r') as file:
        content = file.read()

    # 使用正则表达式匹配所有的train_loss值
    train_losses = re.findall(r'"train_loss": (\d+\.\d+),', content)

    # 将字符串转换为浮点数
    train_losses = [float(loss) for loss in train_losses[:72]]

    return train_losses


# 绘制折线图并保存
def plot_and_save(train_losses1, train_losses2, output_filename):
    # 创建一个范围列表，用于x轴
    epochs1 = range(1, len(train_losses1) + 1)
    epochs2 = range(1, len(train_losses2) + 1)

    # 绘制折线图
    plt.figure(figsize=(5, 5))
    plt.plot(epochs1, train_losses1, 'b-', label='ES-DETR')
    plt.plot(epochs2, train_losses2, 'm-', label='RT-DETR-R50')
    plt.title('PGFD')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 保存图像到文件
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")


# 主函数
def main():
    filename1 = 'other_result/pg/log.txt'  # 第一个文件名
    filename2 = 'other_result/pg_rt/log.txt'  # 第一个文件名
    output_filename = 'train_loss_comparison7.png'  # 输出图像文件名

    # 提取训练损失
    train_losses1 = extract_train_loss(filename1)
    train_losses2 = extract_train_loss(filename2)

    # 检查是否正好有200个数据点
    plot_and_save(train_losses1, train_losses2, output_filename)
    # if len(train_losses1) == 100 and len(train_losses2) == 100:
    #     plot_and_save(train_losses1, train_losses2, output_filename)
    # else:
    #     print(f"Warning: Expected 200 data points for each file, found {len(train_losses1)} and {len(train_losses2)}")


if __name__ == '__main__':
    main()