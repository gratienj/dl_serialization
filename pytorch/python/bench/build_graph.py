import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def data_benchmark(text_file: str) -> None:
    data=pd.DataFrame()
    time=[]
    with open (text_file, "r") as f:
        lines = f.readlines()

        previous_label=10.0
        for i,line in enumerate(lines):
            line=line.split(" ")

            if len(line) !=5:
                continue

            new_label=np.float64(line[0])


            if new_label == previous_label:
                time.append(float(line[-2]))
            else:
                # set time:
                print(len(time))
                data[previous_label]=time
                # reset time
                time=[]
                time.append(float(line[-2]))
                # update previous_label
                previous_label=new_label

        data[new_label]=time

        f.close()

    return data


def build_data(build_path):
    data = data_benchmark(build_path)
    return data


def build_graph(data_torch: pd.DataFrame):
    plt.style.use('default')

    fig, axes= plt.subplots(1,2,figsize=(20,4))

    for i in range(0,data_torch.shape[1]):
        axes[0].plot(data_torch.iloc[:,i].to_list(),marker='^', label=f"data size : {data_torch.columns[i]}")
        axes[0].set_title(f"inference CPU : scalabilité")
        axes[0].set_xlabel("nombre d\'appels du modèle ")
        axes[0].set_yscale('log')
        axes[0].set_ylabel("time (secondes)")

        axes[0].grid(True,which="both", linestyle='--')
        axes[0].legend()


    for i in range(0,10):
        axes[1].plot(data_torch.columns,data_torch.iloc[i,:].to_list(),marker='^', label=f"appel n°: {i}")
        axes[1].set_title(f"inference CPU : dérivées temporelles")
        axes[1].set_xlabel("data size")
        axes[1].set_yscale('log')
        axes[1].set_xscale('log')
        axes[1].set_ylabel("time (secondes)")

        axes[1].grid(True,which="both", linestyle='--')
        axes[1].legend()

    plt.show()
    #fig.savefig("images/inference_cpu_scaling_benchmark.png")
    print("fin sauvegarde")

def build_factor(data_torch: pd.DataFrame):
    row=data_torch.shape[0]
    col=data_torch.shape[1]-1

    scale=[data_torch.columns[j+1]-data_torch.columns[j-1] for j in range(1,col)]

    # dicts:
    data={scale[j]:[] for j in range(0,col-1)}

    for i in range(row):
        for j in range(1,col):
            r=(np.log10(data_torch.iloc[i , j+1])- np.log10(data_torch.iloc[i , j-1]))/2
            data[scale[j-1]].append(r)

    df_data=pd.DataFrame.from_dict(data)
    return df_data




if __name__=="__main__":
    df_LR=build_data('Latance_cluster_LR.txt')
    print(df_LR)
    build_graph(df_LR)
    df_grad=build_factor(df_LR)
    print(df_grad)
    build_graph(df_grad)
