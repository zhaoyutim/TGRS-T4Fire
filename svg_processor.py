import pandas as pd

from matplotlib import pyplot as plt

path = '/Users/zhaoyu/PycharmProjects/T4Fire/svg/afdes2.csv'
if __name__=='__main__':
    df = pd.read_csv(path)
    df['b4_avg']=df['b4'].rolling(1).sum()
    df['b5_avg']=df['b5'].rolling(1).sum()
    df['Date']=pd.to_datetime(df['system:time_start']).dt.strftime('%m-%d')
    print(df.head(5))
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    df.plot(x="Date", y=['b4', 'b5'], figsize=(10,5))
    plt.savefig('afdes2.svg', bbox_inches='tight')
    plt.show()
    # plt.imshow(path)
    # plt.show()