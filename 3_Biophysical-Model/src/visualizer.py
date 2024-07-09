import numpy as np
import matplotlib.pyplot as plt


def plot_potentials(t_eval, potentials, time_span=None, title=''):
    """複数の膜電位をプロットする関数
    Parameters
    ----------
    t_eval : np.ndarray or list
        評価の対象となる時刻を保存した一次元の配列。
    potentials : np.ndarray or dict
        膜電位の値を保存した行列。
        縦が時刻で横がユニット数。
    time_span : None or tuple of float
        キュー電流を流す時間の開始時刻と終了時刻
    title : str
        フィギュアのタイトル。
    """
    colors = ['tab:blue', 'tab:gray', 'tab:cyan', 'black']
    styles = ['solid', 'dashed', 'dotted', 'dashdot']

    if isinstance(potentials, dict):
        # 辞書型の場合はNumpy配列に変換
        potentials = [val for val in potentials.values()]
        potentials = np.array(potentials).T

    plt.figure(figsize=(12, 2))
    for idx in range(potentials.shape[1]):
        idx_plot = int(idx % len(colors))
        plt.plot(
            t_eval, potentials[:, idx],
            linewidth=1,
            linestyle=styles[idx_plot],
            color=colors[idx_plot],
        )
    if time_span is not None:
        for time_plot in time_span:
            plt.vlines(
                time_plot,
                np.min(potentials),
                np.max(potentials),
                colors='black',
                linestyles='dotted',
            )
    # plt.xlim(t_eval[0], t_eval[-1])
    plt.xlabel('時刻 [msec]')
    plt.ylabel('膜電位')
    plt.title(title)
    plt.show()


def plot_current_and_potential(t_eval, current, potential, title=''):
    """膜電位と電流を同時にプロットする関数

    Parameters
    ----------
    t_eval : np.ndarray or list
        評価の対象となる時刻を保存した一次元の配列
    current : np.ndarray or list
        電流の値を保存した一次元の配列
    potential : np.ndarray or list
        膜電位の値を保存した一次元の配列
    title : str
        フィギュアのタイトル
    """
    fig = plt.figure(figsize=(12, 4))

    plt.subplot(2, 1, 1)
    plt.plot(t_eval, current)
    plt.ylim(-5, 70)
    plt.ylabel('電流')

    plt.subplot(2, 1, 2)
    plt.plot(t_eval, potential)
    plt.ylim(-100, 70)
    plt.ylabel('膜電位')

    # plt.xlim(t_eval[0], t_eval[-1])
    plt.xlabel('時刻')
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_conductance(t_eval, diff_cond, conductances, title=''):
    """コンダクタンスをプロットする関数

    Parameters
    ----------
    t_eval : np.ndarray or list
        評価の対象となる時刻を保存した一次元の配列
    diff_cond : np.ndarray or list
        コンダクタンスの微分(f)の値を保存した一次元の配列
    conductance : np.ndarray or list
        コンダクタンス(g)の値を保存した一次元の配列
    title : str
        フィギュアのタイトル
    """
    plt.figure(figsize=(12, 2))
    plt.plot(t_eval, diff_cond, label='dgdt (f)', color='black', linestyle='dashed')
    plt.plot(t_eval, conductances, label='コンダクタンス (g)', color='tab:blue', linestyle='solid')
    # plt.xlim(t_eval[0], t_eval[-1])
    plt.legend()
    plt.xlabel('時刻 [msec]')
    plt.ylabel('コンダクタンス')
    plt.title(title)
    plt.show()


def plot_raster(t_eval, spikes, time_span=None, title=''):
    """ラスタグラムをプロットする関数

    Parameters
    ----------
    t_eval : np.ndarray or list
        評価の対象となる時刻を保存した一次元の配列
    spikes : np.ndarray
        スパイク発火をゼロイチとして保存した行列。
        縦が時刻で横がユニット数。
    time_span : None or tuple of float
        キュー電流を流す時間の開始時刻と終了時刻
    title : str
        フィギュアのタイトル
    """
    num_unit = spikes.shape[1]

    rslt = []
    for idx_unit in range(num_unit):
        rslt.extend([
            [t, idx_unit] for t, spike in zip(t_eval, spikes[:, idx_unit])
            if spike == 1
        ])
    rslt = np.array(rslt).T

    plt.figure(figsize=(12, 4))
    if np.sum(rslt) != 0:
        plt.scatter(rslt[0], rslt[1], c='black', marker='s', s=0.6)
    else:
        # ひとつも発火がなかった場合の処理
        print('Raster plot is skipped because of no spikes.')
    plt.xlim(t_eval[0], t_eval[-1])
    plt.ylim(-1, num_unit)
    plt.xlabel('時刻 [msec]')
    plt.ylabel('ニューロンの番号')

    # 開始時刻に点線をつける
    if time_span is not None:
        for time_plot in time_span:
            plt.vlines(time_plot, -1, num_unit, colors='black', linestyles='dotted')
    plt.title(title)
    plt.show()
