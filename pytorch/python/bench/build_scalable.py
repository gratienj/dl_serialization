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
                data[previous_label]=time
                # reset time
                time=[]
                time.append(float(line[-2]))
                # update previous_label
                previous_label=new_label

        data[new_label]=time

        f.close()

    return data


def build_data(build_path) -> tuple:
    sufix={"LR":"/LR/log_out_LR.txt","MLR":"/MLR/log_out_MLR.txt","nonLR":"/nonLR/log_out_nonLR.txt"}

    # LR :
    LR=build_path+sufix['LR']
    df_LR = data_benchmark(LR)

    # MLR
    MLR=build_path+sufix['MLR']
    df_MLR = data_benchmark(MLR)

    # nonLR
    nonLR=build_path+sufix['nonLR']
    df_nonLR = data_benchmark(nonLR)

    return (df_LR, df_MLR, df_nonLR)



def build_graph(data_torch: tuple, delta_data_torch: tuple):
    # affichage settings: ----------------------
    # plt.style.use('default')

    SMALL_SIZE = 6
    MEDIUM_SIZE = 8


    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)     # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)   # fontsize of the figure title


    message1="temps de réponse par nombre d'appels"
    message2="temps de réponse par data size"
    message3="dérivée temporelle par data size"


    names=['LR','MLR','nonLR']

    fct_call_number_to_plt=10

    #--------------------------------------------


    fig, axes= plt.subplots(3,3,figsize=(20,12))

    for k in range(3):
        for i in range(0,data_torch[k].shape[1]):
            axes[k][0].plot(data_torch[k].iloc[:,i].to_list(),marker='^', label=f"data size : {data_torch[k].columns[i]}")
            axes[k][0].set_title(f"inference CPU : modèle {names[k]} {message1}")
            if k==2: axes[k][0].set_xlabel("nombre d\'appels du modèle ")
            #else: axes[k][0].axes.xaxis.set_visible(False)
            # axes.set_xscale('log')
            axes[k][0].set_yscale('log')
            axes[k][0].set_ylabel("time (secondes)")

            axes[k][0].grid(True,which="both", linestyle='--')
            axes[k][0].legend()


        for i in range(0,fct_call_number_to_plt):
            axes[k][1].plot(data_torch[k].columns,data_torch[k].iloc[i,:].to_list(),marker='^', label=f"appel n°: {i}")
            axes[k][1].set_title(f"inference CPU : modèle {names[k]} {message2}")
            if k==2: axes[k][1].set_xlabel("data size")
            #else: axes[k][1].axes.xaxis.set_visible(False)
            # axes.set_xscale('log')
            axes[k][1].set_yscale('log')
            axes[k][1].set_xscale('log')
            axes[k][1].set_ylabel("time (secondes)")

            axes[k][1].grid(True,which="both", linestyle='--')
            axes[k][1].legend()

        for i in range(0,fct_call_number_to_plt):
            axes[k][2].plot(delta_data_torch[k].columns,delta_data_torch[k].iloc[i,:].to_list(),marker='^', label=f"appel n°: {i}")
            axes[k][2].set_title(f"inference CPU : modèle {names[k]} {message3}")
            if k==2: axes[k][2].set_xlabel("data size")
            #else: axes[k][1].axes.xaxis.set_visible(False)
            # axes.set_xscale('log')
            axes[k][2].set_yscale('log')
            axes[k][2].set_xscale('log')
            axes[k][2].set_ylabel("time (secondes)")

            axes[k][2].grid(True,which="both", linestyle='--')
            axes[k][2].legend()

    plt.show()
    #fig.savefig("images/inference_cpu_scaling_benchmark.png")
    print("fin sauvegarde")


def compute_factor(data_tuple: tuple) -> list:
    delta=[]
    for data_torch in data_tuple:
        row=data_torch.shape[0]
        col=data_torch.shape[1]-1

        scale=[data_torch.columns[j+1]-data_torch.columns[j-1] for j in range(1,col)]

        # initialize dicts:
        data={scale[j]:[] for j in range(0,col-1)}


        for i in range(row):
            for j in range(1,col):
                r=(np.log10(data_torch.iloc[i , j+1])- np.log10(data_torch.iloc[i , j-1]))/2
                data[scale[j-1]].append(r)

        df_data=pd.DataFrame.from_dict(data)

        delta.append(df_data)

    return delta


# build data frame from data files:
data=build_data('build/')

# compute derivatives: ---------
delta_data=compute_factor(data)

# build scalability  graph:-----
build_graph(data, delta_data)
