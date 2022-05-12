import pandas as pd

from matplotlib import pyplot as plt

path = '/Users/zhaoyu/PycharmProjects/T4Fire/svg/afdes1.csv'
if __name__=='__main__':
    df = pd.read_csv(path)
    df['b4_avg']=df['b4'].rolling(1).sum()
    df['b5_avg']=df['b5'].rolling(1).sum()
    print(df.head(5))
    df.plot(x="system:time_start", y=['b4', 'b5'], figsize=(10,5))
    plt.savefig('afdes1.svg', bbox_inches='tight')
    plt.show()
    # plt.imshow(path)
    # plt.show()