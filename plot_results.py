import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets

models = ['mit-b0','mit-b1','mit-b2','mit-b3','mit-b4','mit-b5']
datasets = ['landslide-uav-sat','landslide-uav-all', 'landslide-sat-all']

def plot_chart(x,y,epochs,eval,labels,):
    global axs
    # Plot each metric for both datasets in their own subplot
    axs[x, y].plot(epochs, eval[0], label='MIT-B0', color='b')
    axs[x, y].plot(epochs, eval[1], label='MIT-B1', color='g')
    axs[x, y].plot(epochs, eval[2], label='MIT-B2', color='r')
    axs[x, y].plot(epochs, eval[3], label='MIT-B3', color='c')
    axs[x, y].plot(epochs, eval[4], label='MIT-B4', color='k')
    axs[x, y].plot(epochs, eval[5], label='MIT-B5', color='m')
    axs[x, y].set_title(f'Eval {labels}')
    axs[x, y].set_xlabel('Epochs')
    axs[x, y].set_ylabel(labels)
    axs[x, y].legend()
    axs[x, y].grid(True)


for dataset in datasets:
    epochs = []
    eval_loss = []
    eval_mIoU = []
    eval_recall = []
    eval_precision = []
    eval_f1 = []
    eval_iou = []
    eval_accuracy = []


    for model in models:
        df = pd.read_csv(f'results/{dataset}/{model}-{dataset}.csv')
        print(f'{model}-{dataset}.csv')
        epochs.append(df['epoch'])
        eval_loss.append(df['eval_loss'])
        eval_mIoU.append(df['eval_mIoU'])
        eval_recall.append(df['eval_recall'])
        eval_precision.append(df['eval_precision'])
        eval_f1.append(df['eval_f1'])
        eval_iou.append(df['eval_iou'])
        eval_accuracy.append(df['eval_accuracy'])

    # Create subplots for each evaluation metric
    fig, axs = plt.subplots(4, 2, figsize=(15, 20))
    plot_chart(0,0,epochs[0],eval_loss,'Loss')
    plot_chart(0,1,epochs[0],eval_mIoU,'mIoU')
    plot_chart(1,0,epochs[0],eval_recall,'Recall')
    plot_chart(1,1,epochs[0],eval_precision,'Precision')
    plot_chart(2,0,epochs[0],eval_f1,'F1')
    plot_chart(2,1,epochs[0],eval_iou,'IoU')
    plot_chart(3,0,epochs[0],eval_accuracy,'Accuracy')


    # Remove the empty subplot (3, 1)
    fig.delaxes(axs[3, 1])

    # Adjust layout to prevent overlap
    plt.tight_layout(h_pad=5)

    # Show the plot
    plt.savefig(f'{dataset}.png')
