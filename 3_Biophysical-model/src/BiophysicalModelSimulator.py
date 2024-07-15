import numpy as np  # 行列演算用ライブラリを読み込む
import matplotlib.pyplot as plt  # 可視化用ライブラリを読み込む


DELTA_T = 0.1  # ステップ幅

# 積分発火モデル
E_REST = -65.0  # 静止膜電位
V_ACT = 40.0  # 活動電位
V_RESET = -65.0  # リセット電位
V_INIT = -70.0  # 電位の初期値
V_THERESHOLD = -55.0  # 発火閾値

T_REF = 2.0  # 不応期 [ms]
T_DELAY = 2.0  # 発火の遅延 [ms]

# 時定数
TAU_1_AMPA, TAU_2_AMPA = 1.0, 5.0
TAU_1_NMDA, TAU_2_NMDA = 10.0, 100.0
TAU_1_GABA, TAU_2_GABA = 1.0, 5.0
TAU_CA = 5.0

# 静電容量（キャパシタンス）
C_LIF = 1.0
C_NMDA = 1.0 * 1000
C_EXC = 0.5 * 1000
C_INH = 0.2 * 1000

# ワーキングメモリモデル
E_LEAK = -70.0
E_AMPA = 0.0
E_NMDA = 0.0
E_GABA = -80.0
E_NA = 50.0
E_KCA = -95.0
V_NA = -56.0

# 最大コンダクタンス値
G_MAX_REST = 1.0
G_MAX_LEAK = 25.0  # ワーキングメモリモデルで使用
G_MAX_LEAK_NMDA = 1.0  # NMDA受容体付き神経細胞で使用
G_MAX_LEAK_EXC = 25.0
G_MAX_LEAK_INH = 20.0
G_MAX_NA = 0.2
G_MAX_AMPA = 2.0
G_MAX_NMDA = 10.0
G_MAX_NMDA = 10.0 * 0.7
G_MAX_GABA = 10.0

# カリウムチャネル
ALPHA_CA = 70.0
BETA_K = 1.0

# 重み付け係数（電位依存性イオンチャネルを使用する場合）
# WEIGHT_STRENGTH_E2E, WEIGHT_WIDTH_E2E = 1.0, 8
# WEIGHT_STRENGTH_E2I, WEIGHT_WIDTH_E2I = 0.8, 50
# WEIGHT_STRENGTH_I2E, WEIGHT_WIDTH_I2E = 0.75, 60
# WEIGHT_STRENGTH_I2I, WEIGHT_WIDTH_I2I = 0.1, 80

# 重み付け係数（LIFベースのモデルを使用する場合）
WEIGHT_STRENGTH_E2E, WEIGHT_WIDTH_E2E = 0.8, 10
WEIGHT_STRENGTH_E2I, WEIGHT_WIDTH_E2I = 0.5, 50
WEIGHT_STRENGTH_I2E, WEIGHT_WIDTH_I2E = 0.5, 50
WEIGHT_STRENGTH_I2I, WEIGHT_WIDTH_I2I = 0.2, 40

# 重み付け係数（可視化関数のテスト用）
# WEIGHT_STRENGTH_E2E, WEIGHT_WIDTH_E2E = 1.2, 5
# WEIGHT_STRENGTH_E2I, WEIGHT_WIDTH_E2I = 0.8, 50
# WEIGHT_STRENGTH_I2E, WEIGHT_WIDTH_I2E = 0.4, 100
# WEIGHT_STRENGTH_I2I, WEIGHT_WIDTH_I2I = 0.1, 200

# シードを固定し結果を再現可能に(実行順序にも依存するため要注意)
SEED = 42


# 評価時間の計算

def generate_t_eval(t_max, delta_t=DELTA_T):
    """評価時間を生成する
    Parameters
    ----------
    t_max : float
        最大時刻 (ミリ秒)
    delta_t : float
        ステップ幅 (ミリ秒)

    Returns
    -------
    t_eval : np.ndarray or list
        評価の対象となる時刻を保存した一次元の配列
    """
    # 最大時刻から最大ステップ数を作成
    step_max = int(t_max / delta_t) + 1

    # 0からt_maxまでstep_max個に等分
    t_eval = np.linspace(0, t_max, step_max)
    return t_eval


# 微分方程式ソルバー

def solve_differential_equation(dydt, t, y, delta_t=1, method='euler', **kwargs):
    """微分方程式dydtを指定すると，次の時刻t+delta_tにおける状態yを返す

    Parameters
    ----------
    dydt : function
        引数yの微分を返す微分方程式。
        dydt()は第一引数に時刻t, 第二引数に状態y, 可変長キーワード引数kwargsを持つ。
    t : float
        時刻
    y : numpy.ndarray
        時刻tにおける状態
    delta_t : float
        ステップ幅 (ミリ秒)
    method : str
        'euler'の場合オイラー法，'rk4'の場合ルンゲ・クッタ法を使用
    kwargs : dict
        可変長キーワード引数。この値は，微分方程式dydtにそのまま渡される
        そのため，時刻や状態以外の変数をdydtに与えたい際に利用する

    Returns
    -------
    y_next : numpy.ndarray
        時刻t + delta_tにおける状態yの値
    """
    if method == 'euler':
        y_next = solve_differential_equation_euler(
            dydt=dydt,
            t=t,
            y=y,
            delta_t=delta_t,
            **kwargs
        )
    elif method == 'rk4':
        y_next = solve_differential_equation_rk4(
            dydt=dydt,
            t=t,
            y=y,
            delta_t=delta_t,
            **kwargs
        )
    else:
        raise NotImplementedError()
    return y_next

def solve_differential_equation_euler(dydt, t, y, delta_t=1, **kwargs):
    """オイラー法を用いて次の時刻t+delta_tにおける状態yを返す
    """
    y_next = delta_t * dydt(t, y, **kwargs) + y
    return y_next

def solve_differential_equation_rk4(dydt, t, y, delta_t=1, **kwargs):
    """ルンゲ・クッタ法を用いて次の時刻t+delta_tにおける状態yを返す
    """
    h = delta_t
    k_1 = h * dydt(t, y, **kwargs)
    k_2 = h * dydt(t + 0.5 * h, y + 0.5 * k_1, **kwargs)
    k_3 = h * dydt(t + 0.5 * h, y + 0.5 * k_2, **kwargs)
    k_4 = h * dydt(t + h, y + k_3, **kwargs)
    y_next = y + (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6
    return y_next


# 積分発火モデル

def calc_lif(t, potential, current_ext, last_spike):
    """1時刻における積分発火モデルの計算

    オイラー法を用いて1時刻における積分発火モデルにおける膜電位の計算を行う

    Parameters
    ----------
    t : float
        時刻 (ms)
    potential : float
        膜電位
    current_ext :
        注入電流
    last_spike :
        直近の発火時刻

    Returns
    -------
    potential_next : float
        次の時刻における膜電位
    """
    if last_spike <= t and t <= last_spike + T_REF:
        # 不応期の間は膜電位をリセット電位に固定
        potential_next = V_RESET
    elif potential >= V_THERESHOLD:
        # 不応期ではなく，かつ発火閾値に達した場合は
        # 活動電位に固定
        potential_next = V_ACT
    else:
        # 不応期ではなく，かつ発火閾値に達しない場合は
        # 微分方程式を用いた更新
        potential_delta = DELTA_T * (1.0 / C_LIF) * (
            - G_MAX_REST * (potential - E_REST) + current_ext)
        potential_next = potential + potential_delta
    return potential_next


def simulate_lif(t_eval,
                 current_max=20.0):
    """外部電流に対するLIFモデルの振る舞いをシミュレーション

    Parameters
    ----------
    t_eval : np.ndarray or list
        評価の対象となる時刻を保存した一次元の配列
    current_max : float
        注入電流の最大値

    Returns
    -------
    potential : np.ndarray
        膜電位の時系列全体
    """
    # (B) 初期値の設定
    potential = V_INIT
    last_spike = -100

    # (C) 結果保存用変数の準備
    results = {
        'potential': [],
        'current': [],
    }

    for t in t_eval:
        # (D) 各時刻における計算
        # (D-1) 各時刻における注入電流の作成
        if t < 30.0:
            current = 0
        elif t < 60.0:
            # 30から60ミリ秒にかけて徐々に電流は増加
            current = (current_max / 30.0) * (t - 30.0)
        elif t < 90.0:
            current = current_max
        else:
            current = 0

        # (D-2) 各時刻における膜電位の更新
        potential = calc_lif(
            t=t,
            potential=potential,
            current_ext=current,
            last_spike=last_spike,
        )

        # (D-3) 発火の判定
        spike = 1 if potential >= V_ACT else 0

        # (D-4) 直近の発火時刻の更新
        last_spike = t if spike == 1 else last_spike

        # (E) 計算結果を保存
        results['potential'].append(potential)
        results['current'].append(current)
        results['spike'].append(spike)

    return results


# 積分発火モデルを用いた回路設計


def simulate_network_lif(t_eval, weight):
    """3つの積分発火モデルをつなげたネットワークをシミュレーションする

    Parameters
    ----------
    current : np.ndarray
        電流 () をシーケンス

    Returns
    -------
    potential : np.ndarray
        膜電位 () をシーケンス
    """
    # (A) シミュレーションの設定
    cue_scale = 20

    # シードを固定し実行ごとの乱数を統一する
    np.random.seed(SEED)

    # (B) 初期値の設定
    potential_1, potential_2, potential_3 = V_INIT, V_INIT, V_INIT
    spike_1, spike_2, spike_3 = 0, 0, 0
    last_spike_1, last_spike_2, last_spike_3 = -100, -100, -100

    # (C) 結果保存用変数の準備
    results = {
        'potential_1': [],
        'potential_2': [],
        'potential_3': [],
    }

    for t in t_eval:
        # (D) 各時刻における計算
        # (D-1) 電流を計算
        current_cue_1 = cue_scale * np.random.rand()
        current_syn_1 = weight * spike_2 + weight * spike_3
        current_1 = current_cue_1 + current_syn_1

        current_cue_2 = cue_scale * np.random.rand()
        current_syn_2 = weight * spike_1 + weight * spike_3
        current_2 = current_cue_2 + current_syn_2

        current_cue_3 = cue_scale * np.random.rand()
        current_syn_3 = weight * spike_1 + weight * spike_2
        current_3 = current_cue_3 + current_syn_3

        # (D-2) 膜電位の更新
        potential_1 = calc_lif(
            t=t,
            potential=potential_1,
            current_ext=current_1,
            last_spike=last_spike_1,
        )
        potential_2 = calc_lif(
            t=t,
            potential=potential_2,
            current_ext=current_2,
            last_spike=last_spike_2,
        )
        potential_3 = calc_lif(
            t=t,
            potential=potential_3,
            current_ext=current_3,
            last_spike=last_spike_3,
        )

        # (D-3) スパイクの判定
        spike_1 = 1.0 if potential_1 == V_ACT else 0.0
        spike_2 = 1.0 if potential_2 == V_ACT else 0.0
        spike_3 = 1.0 if potential_3 == V_ACT else 0.0

        # (D-4) 直近の発火時刻の更新
        last_spike_1 = t if spike_1 == 1.0 else last_spike_1
        last_spike_2 = t if spike_2 == 1.0 else last_spike_2
        last_spike_3 = t if spike_3 == 1.0 else last_spike_3

        # (E) 計算結果を保存
        results['potential_1'].append(potential_1)
        results['potential_2'].append(potential_2)
        results['potential_3'].append(potential_3)
    return results


# コンダクタンスのモデル
def calc_synaptic_effect(last_spikes, t, weights, g_max):
    """シナプスの影響度を演算

    行列の計算に注意！

    Parameters
    ----------
    last_spikes : np.ndarray
        直近に発火をした時刻。
        次元は，シナプス前細胞の数 x 1
    t : float
        時刻
    weights : np.ndarray
        重み付け係数
        次元は，シナプス後細胞の数 x シナプス前細胞の数
    g_max : float
        最大コンダクタンス値

    Returns
    -------
    synapse_effects : np.ndarray
        シナプスの影響度。
        次元は，シナプス後細胞の数 x シナプス前細胞の数
    """
    # 直近の時刻から，シナプス前細胞におけるスパイクを判定
    # DIM: num_unit_from x 1
    check_spikes = np.where(
        (t - 2 * DELTA_T <= last_spikes + T_DELAY) & (last_spikes + T_DELAY <= t - DELTA_T), 1.0, 0.0)
    check_spikes = check_spikes / DELTA_T

    # スパイク発火の行列を整形
    # DIM: num_unit x num_unit_from
    check_spikes = np.tile(check_spikes.T, reps=(weights.shape[0], 1))

    # DIM: num_unit x num_unit_from
    synapse_effects = g_max * weights * check_spikes
    return synapse_effects


def calc_dfdt(diff_cond,
              conductances,
              synapse_effects,
              tau_1,
              tau_2):
    """コンダクタンスの二階微分を計算

    Parameters
    ----------
    diff_cond : np.ndarray
        コンダクタンスの微分.
        次元は，シナプス後細胞の数 x シナプス前細胞の数
    conductances : np.ndarray
        コンダクタンス.
        次元は，シナプス後細胞の数 x シナプス前細胞の数
    synapse_effects : np.ndarray
        前シナプスからの影響度.
        次元は，シナプス後細胞の数 x シナプス前細胞の数

    Returns
    -------
    dfdt : np.ndarray
        コンダクタンスの二階微分
        次元は，シナプス後細胞の数 x シナプス前細胞の数
    """
    tau_sumprod = (1 / tau_1) + (1 / tau_2)
    dfdt = - tau_sumprod * diff_cond \
        - (1 / (tau_1 * tau_2)) * conductances \
        + tau_sumprod * synapse_effects
    return dfdt


def differentiate_conductance(t, y, **kwargs):
    """コンダクタンスの微分と二階微分を返す

    Parameters
    ----------
    t : float
        時刻
    y : np.ndarray
        コンダクタンスとコンダクタンスの微分を合体した変数
    kwargs : dict
        可変長キーワード引数

    Returns
    -------
    dydt : np.ndarray
        コンダクタンスの微分と二階微分を合体した変数
    """
    g_max = 1.0

    # 変数の取得
    last_spikes = kwargs['last_spikes']
    weights = kwargs['weights']
    tau_1 = kwargs['tau_1']
    tau_2 = kwargs['tau_2']

    # 状態yをコンダクタンスとコンダクタンスの微分に分解
    diff_cond, conductances = np.split(y, [1], axis=1)

    # (i) シナプスの影響度eの計算
    synapse_effects = calc_synaptic_effect(
        last_spikes=last_spikes,
        t=t,
        weights=weights,
        g_max=g_max,
    )

    # (ii) コンダクタンスの微分fの計算（微分）
    dfdt = calc_dfdt(
        diff_cond=diff_cond,
        conductances=conductances,
        synapse_effects=synapse_effects,
        tau_1=tau_1,
        tau_2=tau_2,
    )

    # (iii) コンダクタンスgの計算（微分）
    dgdt = diff_cond

    # 微分の値を一つの変数にまとめる
    dydt = np.hstack([dfdt, dgdt])
    return dydt


def simulate_conductance(t_eval,
                         tau_1,
                         tau_2):
    """コンダクタンスの挙動をシミュレーションする関数

    Parameters
    ----------
    t_eval : np.ndarray or list
        評価の対象となる時刻を保存した一次元の配列
    tau_1 : float
        コンダクタンスモデルに渡される時定数tau_1
    tau_2 : float
        コンダクタンスモデルに渡される時定数tau_2

    Returns
    -------
    results : dict
        コンダクタンスやその微分の変化をリストとして値に保存した辞書
    """
    # (A) シミュレーションの設定
    # 重み付け係数は，値が1の1 x 1の行列とする.
    # 神経細胞数は1
    weights = np.ones((1, 1))

    # (B) 初期値の設定
    last_spikes = -100 * np.ones((1, 1))
    diff_cond = np.zeros((1, 1))
    conductances = np.zeros((1, 1))
    y = np.hstack([diff_cond, conductances])

    # (C) 結果保存用変数の準備
    results = {
        'diff_cond': [],
        'conductances': [],
    }

    for t in t_eval:
        # (D) 各時刻における計算
        # (D-1) 微分方程式による更新
        y = solve_differential_equation(
            dydt=differentiate_conductance,
            t=t,
            y=y,
            delta_t=DELTA_T,
            method='rk4',
            last_spikes=last_spikes,
            weights=weights,
            tau_1=tau_1,
            tau_2=tau_2,
        )
        diff_cond, conductances = np.split(
            y, [1], axis=1)

        # (D-2) スパイクの判定
        # 10-20msecは常時発火と仮定
        spikes = np.where(
            10 < t and t < 20 , 1, 0
        ).reshape(1, 1)

        # (D-3) 直近の発火時刻の更新
        last_spikes = np.where(
            spikes == 1, t, last_spikes
        )

        # (E) 計算結果を保存
        results['diff_cond'].append(
            diff_cond)
        results['conductances'].append(
            conductances)

    # (F) 返り値の準備
    for key, value in results.items():
        # 結果の値をNumpy型に整形
        results[key] = np.hstack(value).T
    return results

# NMDA受容体付き神経細胞
def calc_block_mg(potential):
    """NMDA受容体のマグネシウムブロックの値を計算
    """
    return 1 / (1 + 0.5 * np.exp(- 0.062 * potential))


def differentiate_nmda_unit(t, y, **kwargs):
    """コンダクタンスの微分と二階微分を返す

    Parameters
    ----------
    t : float
        時刻
    y : np.ndarray
        コンダクタンスとコンダクタンスの微分を合体した変数
    kwargs : dict
        可変長キーワード引数

    Returns
    -------
    dydt : np.ndarray
        コンダクタンスの微分と二階微分を合体した変数
    """
    # 変数の取得
    last_spikes = kwargs['last_spikes']
    weights = kwargs['weights']
    tau_1 = kwargs['tau_1']
    tau_2 = kwargs['tau_2']

    # 状態yをコンダクタンスの微分，コンダクタンス，膜電位の微分に分解
    diff_cond, conductances, potentials = \
        np.split(y, [1, 2], axis=1)

    # (i) シナプスの影響度eの計算
    synapse_effects = calc_synaptic_effect(
        last_spikes=last_spikes,
        t=t,
        weights=weights,
        g_max=G_MAX_NMDA,
    )

    # (ii) コンダクタンスの微分fの計算（微分）
    dfdt = calc_dfdt(
        diff_cond=diff_cond,
        conductances=conductances,
        synapse_effects=synapse_effects,
        tau_1=tau_1,
        tau_2=tau_2,
    )

    # (iii) コンダクタンスのgの計算（微分）
    dgdt = diff_cond

    # (iv) 電流Iの計算
    f_mg = calc_block_mg(potentials)
    current_nmda = - f_mg * conductances * (potentials - E_NMDA)
    current_leak = - G_MAX_LEAK_NMDA * (potentials - E_LEAK)
    current_cue = 1000.0 if t >= 100.0 and t < 150.0 else 0.0
    current = (current_nmda + current_leak + current_cue) / C_NMDA

    # (v) 膜電位vの計算（微分）
    dvdt = current

    # 微分の値を一つの変数にまとめる
    dydt = np.hstack([dfdt, dgdt, dvdt])
    return dydt


def simulate_nmda_unit(t_eval,
                       weight=3.0,
                       tau_1=10.0,
                       tau_2=100.0):
    """コンダクタンスの挙動をシミュレーションする関数

    Parameters
    ----------
    t_eval : np.ndarray or list
        評価の対象となる時刻を保存した一次元の配列
    weight : float
        重みづけ係数
    tau_1 : float
        コンダクタンスモデルに渡される時定数tau_1
    tau_2 : float
        コンダクタンスモデルに渡される時定数tau_2

    Returns
    -------
    results : dict
        コンダクタンスやその微分の変化をリストとして値に保存した辞書
    """
    # (A) シミュレーションの設定
    # 重み付け係数は，値が1の1 x 1の行列とする. 神経細胞数は1
    weights = weight * np.ones((1, 1))

    # (B) 初期値の設定
    last_spikes = -100 * np.ones((1, 1))
    diff_cond = np.zeros((1, 1))
    conductances = np.zeros((1, 1))
    potentials = V_INIT * np.ones((1, 1))

    # (C) 結果保存用変数の準備
    results = {
        'potentials': [],
        'diff_cond': [],
        'conductances': [],
    }

    for t in t_eval:
        # (D) 各時刻における計算
        # 変数をまとめる
        y = np.hstack([diff_cond, conductances, potentials])

        # (D-1) 微分方程式による更新
        y = solve_differential_equation(
            dydt=differentiate_nmda_unit,
            t=t,
            y=y,
            delta_t=DELTA_T,
            method='rk4',
            last_spikes=last_spikes,
            weights=weights,
            tau_1=tau_1,
            tau_2=tau_2,
        )
        diff_cond, conductances, potentials = np.split(y, [1, 2], axis=1)

        # (D-2) 積分発火モデルによる更新
        fuouki = (last_spikes <= t) & (t <= last_spikes + T_REF)
        katsudo = (potentials >= V_THERESHOLD) & (~ fuouki)
        potentials[katsudo] = V_ACT
        potentials[fuouki] = V_RESET

        # (D-3) スパイクの判定
        spikes = np.where(potentials == V_ACT, 1, 0)

        # (D-4) 直近の発火時刻の更新
        last_spikes = np.where(spikes == 1, t, last_spikes)

        # (E) 計算結果を保存
        results['diff_cond'].append(diff_cond)
        results['conductances'].append(conductances)
        results['potentials'].append(potentials)

    # (F) 返り値の準備
    for key, value in results.items():
        # 結果の値をNumpy型に整形
        results[key] = np.hstack(value).T

    return results


# ワーキングメモリモデル
def calc_exp(dist, strength, width):
    """距離から重み付け係数を計算
    Parameters
    ----------
    dist
        細胞間の距離
    strength
        結合強度の分布の最大値を決める
    width
        結合強度の分布の横幅を決める

    Returns
    -------
        重み付け係数
    """
    w = strength * np.exp(- (dist**2) / (2 * width**2))
    return w


def calc_weights(strength, width, degree_pre, degree_post):
    """シナプス前・後細胞の位置から距離を計算し重み付け係数を計算
    """
    # DIM: num_unit_to x num_unit_from
    degree_pre_mat = np.vstack([degree_pre for theta in degree_post])

    # DIM: num_unit_to x num_unit_from
    degree_post_mat = np.vstack([degree_post for theta in degree_pre]).T

    # シナプス前細胞と後細胞の距離を計算
    # DIM: num_unit_to x num_unit_from
    dist = np.minimum(
        (degree_pre_mat - degree_post_mat) % 360,
        (degree_post_mat - degree_pre_mat) % 360,
    )
    weights = calc_exp(
        dist=dist,
        strength=strength,
        width=width,
    )
    return weights


def init_weights_circle(set_exc,
                        set_inh,
                        positions,
                        ):
    """円形に配置されたニューロン間の重みを作成
    """
    num_unit = len(positions)
    num_unit_exc = len(set_exc)
    num_unit_inh = int(num_unit * 0.2)  # 抑制性神経細胞の総数
    num_column = num_unit_inh  # カラムの総数

    strength_exc2exc, width_exc2exc = WEIGHT_STRENGTH_E2E, WEIGHT_WIDTH_E2E
    strength_exc2inh, width_exc2inh = WEIGHT_STRENGTH_E2I, WEIGHT_WIDTH_E2I
    strength_inh2exc, width_inh2exc = WEIGHT_STRENGTH_I2E, WEIGHT_WIDTH_I2E
    strength_inh2inh, width_inh2inh = WEIGHT_STRENGTH_I2I, WEIGHT_WIDTH_I2I

    # weights = np.empty(shape=(num_unit, num_unit))
    weights = 100 * np.ones(shape=(num_unit, num_unit))
    # weights_from_exc = np.empty(shape=(num_unit, num_unit_exc))
    # weights_from_inh = np.empty(shape=(num_unit, num_unit_inh))

    # 興奮性細胞から興奮性細胞の結合
    weights[np.ix_(set_exc, set_exc)] = calc_weights(
        strength=strength_exc2exc,
        width=width_exc2exc,
        degree_pre=np.array(positions)[set_exc],
        degree_post=np.array(positions)[set_exc],
    )
    # weights_from_exc[set_exc, :] = calc_weights(
    #     strength=strength_exc2exc,
    #     width=width_exc2exc,
    #     degree_pre=np.array(positions)[set_exc],
    #     degree_post=np.array(positions)[set_exc],
    # )

    # 興奮性細胞から抑制性細胞の結合
    weights[np.ix_(set_inh, set_exc)] = calc_weights(
        strength=strength_exc2inh,
        width=width_exc2inh,
        degree_pre=np.array(positions)[set_exc],
        degree_post=np.array(positions)[set_inh],
    )
    # weights_from_exc[set_inh, :] = calc_weights(
    #     strength=strength_exc2inh,
    #     width=width_exc2inh,
    #     degree_pre=np.array(positions)[set_exc],
    #     degree_post=np.array(positions)[set_inh],
    # )

    # 抑制性細胞から興奮性細胞の結合
    weights[np.ix_(set_exc, set_inh)] = calc_weights(
        strength=strength_inh2exc,
        width=width_inh2exc,
        degree_pre=np.array(positions)[set_inh],
        degree_post=np.array(positions)[set_exc],
    )
    # weights_from_inh[set_exc, :] = calc_weights(
    #     strength=strength_inh2exc,
    #     width=width_inh2exc,
    #     degree_pre=np.array(positions)[set_inh],
    #     degree_post=np.array(positions)[set_exc],
    # )

    # 抑制性細胞から抑制性細胞の結合
    weights[np.ix_(set_inh, set_inh)] = calc_weights(
        strength=strength_inh2inh,
        width=width_inh2inh,
        degree_pre=np.array(positions)[set_inh],
        degree_post=np.array(positions)[set_inh],
    )
    # weights_from_inh[set_inh, :] = calc_weights(
    #     strength=strength_inh2inh,
    #     width=width_inh2inh,
    #     degree_pre=np.array(positions)[set_inh],
    #     degree_post=np.array(positions)[set_inh],
    # )
    return weights


def generate_network_architecture(num_unit,
                                  ion=False):
    # 神経細胞の個数の設定
    num_unit_exc = int(num_unit * 0.8)  # 興奮性神経細胞の総数
    num_unit_inh = int(num_unit * 0.2)  # 抑制性神経細胞の総数
    num_column = num_unit_inh  # カラムの総数

    # np.split()で使用するインデックスのリストを用意
    split_list = [
        # AMPA受容体のコンダクタンス
        num_unit_exc,
        # AMPA受容体のコンダクタンスの微分
        2 * num_unit_exc,
        # NMDA受容体のコンダクタンス
        3 * num_unit_exc,
        # NMDA受容体のコンダクタンスの微分
        4 * num_unit_exc,
        # GABA受容体のコンダクタンス
        4 * num_unit_exc + num_unit_inh,
        # GABA受容体のコンダクタンスの微分
        4 * num_unit_exc + 2 * num_unit_inh,
    ]
    if ion:
        # カルシウム濃度の数を追加する
        split_list = split_list + [
            # 膜電位
            4 * num_unit_exc + 2 * num_unit_inh + 1,
        ]

    # 興奮性・抑制性神経細胞の設定
    set_exc = [  # 興奮性を示す添字集合
        idx for idx in range(num_unit) if idx < num_unit_exc
    ]
    set_inh = [  # 抑制性を示す添字集合
        idx for idx in range(num_unit) if idx >= num_unit_exc
    ]
    excitation_binary = [  # 興奮性の場合1を持つリストも用意
        1 if idx in set_exc else 0 for idx in range(num_unit)
    ]

    # 神経細胞に対する位置（角度）の割り当て
    positions_exc = [  # 興奮性細胞に位置を割り当て
        np.floor(idx / 4) * np.floor(360 / num_column) \
        for idx in range(num_unit_exc)
    ]
    positions_inh = [  # 抑制性細胞に位置を割り当てる
        idx * np.floor(360 / num_column) \
        for idx in range(num_unit_inh)
    ]
    # 興奮性細胞と抑制性細胞の位置を結合
    positions = positions_exc + positions_inh


    # キュー電流を与える神経細胞の割り当て
    # 角度が180かつ興奮性の細胞にのみキュー電流を与える
    positions_cue = np.where(
        (np.array(positions) == 180) \
            & (np.array(excitation_binary) == 1), 1, 0)

    # 重み
    weights = init_weights_circle(
        set_exc=set_exc,
        set_inh=set_inh,
        positions=positions,
    )

    architecture = {
        'num_unit': num_unit,
        'num_unit_exc': num_unit_exc,
        'num_unit_inh': num_unit_inh,
        'split_list': split_list,
        'set_exc': set_exc,
        'set_inh': set_inh,
        'excitation_binary': excitation_binary,
        'positions': positions,
        'positions_cue': positions_cue,
        'weights': weights,
        # 'weights_from_exc': weights_from_exc,
        # 'weights_from_inh': weights_from_inh,
    }

    return architecture


def test_weight():
    # 重みづけ係数の算出
    weights_from_exc, weights_from_inh = bms.init_weights_circle(
        set_exc=set_exc,
        set_inh=set_inh,
        positions=positions,
    )

    # 結果を確認すると意図通りのサイズになっていることが分かる
    print(
        'Shape of weights_from_exc:', weights_from_exc.shape,
        'Shape of weights_from_inh:', weights_from_inh.shape,
    )

    # 念の為値も確認してみる
    print('weights_from_exc:', weights_from_exc[:10, :10])


def differentiate_working_memory_lif(t, y, **kwargs):
    """ワーキングメモリモデルの微分方程式

    コンダクタンスの微分p, コンダクタンスg, 膜電位v, カルシウム濃度caについて微分する
    状態yは，上記の変数を保存

    Parameters
    ----------
    t: float
        時刻
    y : np.ndarray
        NMDA/AMPA/GABA受容体のコンダクタンスの微分p, コンダクタンスg, 膜電位v, カルシウム濃度ca
    kwargs : dict
        ノイズ電流current_noiseと直近のスパイク時刻last_spikesを含むこと

    Returns
    -------
    dydt : np.ndarray
        コンダクタンスの微分p, コンダクタンスg, 膜電位v, カルシウム濃度caに関する微分
    """
    # 引数の取得
    current_noise = kwargs['current_noise']
    last_spikes = kwargs['last_spikes']
    dysfuncs_ampa = kwargs['dysfuncs_ampa']
    dysfuncs_nmda = kwargs['dysfuncs_nmda']
    dysfuncs_gaba = kwargs['dysfuncs_gaba']

    architecture = kwargs['architecture']
    set_exc = architecture['set_exc']
    set_inh = architecture['set_inh']

    (diff_cond_ampa, conductances_ampa,
     diff_cond_nmda, conductances_nmda,
     diff_cond_gaba, conductances_gaba,
     potentials, ) = np.split(
        y, architecture['split_list'], axis=1)

    # (i) シナプスの影響度eの計算
    synapse_effects_ampa = calc_synaptic_effect(
        last_spikes=last_spikes[architecture['set_exc'], :],
        t=t,
        weights=architecture['weights'][:, architecture['set_exc']],
        g_max=G_MAX_AMPA * dysfuncs_ampa,
    )
    synapse_effects_nmda = calc_synaptic_effect(
        last_spikes=last_spikes[architecture['set_exc'], :],
        t=t,
        weights=architecture['weights'][:, architecture['set_exc']],
        g_max=G_MAX_NMDA * dysfuncs_nmda,
    )
    synapse_effects_gaba = calc_synaptic_effect(
        last_spikes=last_spikes[architecture['set_inh'], :],
        t=t,
        weights=architecture['weights'][:, architecture['set_inh']],
        g_max=G_MAX_GABA * dysfuncs_gaba,
    )

    # (ii) リガンド依存性コンダクタンスの微分fの計算（微分）
    dfdt_ampa = calc_dfdt(
        diff_cond=diff_cond_ampa,
        conductances=conductances_ampa,
        synapse_effects=synapse_effects_ampa,
        tau_1=TAU_1_AMPA,
        tau_2=TAU_2_AMPA,
    )
    dfdt_nmda = calc_dfdt(
        diff_cond=diff_cond_nmda,
        conductances=conductances_nmda,
        synapse_effects=synapse_effects_nmda,
        tau_1=TAU_1_NMDA,
        tau_2=TAU_2_NMDA,
    )
    dfdt_gaba = calc_dfdt(
        diff_cond=diff_cond_gaba,
        conductances=conductances_gaba,
        synapse_effects=synapse_effects_gaba,
        tau_1=TAU_1_GABA,
        tau_2=TAU_2_GABA,
    )

    # (iii) リガンド依存性コンダクタンスgの計算（微分）
    dgdt_ampa = diff_cond_ampa
    dgdt_nmda = diff_cond_nmda
    dgdt_gaba = diff_cond_gaba

    # (v) 電流の計算
    current_ampa = - np.sum(
        conductances_ampa * np.tile(
            potentials - E_AMPA, len(architecture['set_exc'])),
        axis=1,
        keepdims=True,
    )
    f_mg = calc_block_mg(potentials)
    current_nmda = - np.sum(
        conductances_nmda * np.tile(
            f_mg * (potentials - E_NMDA), len(architecture['set_exc'])),
        axis=1,
        keepdims=True,
    )
    current_gaba = - np.sum(
        conductances_gaba * np.tile(
            potentials - E_GABA, len(architecture['set_inh'])),
        axis=1,
        keepdims=True,
    )
    current_leak = np.empty_like(current_nmda)
    current_leak[architecture['set_exc'], :] = \
        - G_MAX_LEAK_EXC * (potentials[architecture['set_exc'], :] - E_LEAK)
    current_leak[architecture['set_inh'], :] = \
        - G_MAX_LEAK_INH * (potentials[architecture['set_inh'], :] - E_LEAK)

    # キュー電流の計算。100-200m秒で電流を流す
    current_cue = \
        (C_EXC if 100.0 <= t and t <= 200.0 else 0.0) \
        * architecture['positions_cue'].reshape(-1, 1)

    # それぞれの電流をまとめる
    current_channel = \
        current_ampa + current_nmda + current_gaba + current_leak
    current = current_channel + current_cue + current_noise

    # (vi) 膜電位vの計算
    # dvdt = current
    dvdt = np.empty_like(potentials)
    dvdt[architecture['set_exc'], :] = current[architecture['set_exc'], :] / C_EXC
    dvdt[architecture['set_inh'], :] = current[architecture['set_inh'], :] / C_INH

    # すべての微分をひとつの配列にまとめる
    dydt = np.hstack([
        dfdt_ampa, dgdt_ampa,
        dfdt_nmda, dgdt_nmda,
        dfdt_gaba, dgdt_gaba,
        dvdt,
    ])
    return dydt


def simulate_working_memory_lif(t_eval,
                                architecture,
                                dysfuncs_ampa=1.0,
                                dysfuncs_nmda=1.0,
                                dysfuncs_gaba=1.0):
    """ワークングメモリの挙動をシミュレーションする関数

    Parameters
    ----------
    t_eval : np.ndarray or list
        評価の対象となる時刻を保存した一次元の配列

    Returns
    -------
    results : dict
        膜電位やスパイクの変化をリストとして値に保存した辞書
    """

    # シードを固定する
    np.random.seed(SEED)

    # 神経細胞の個数の設定
    num_unit = architecture['num_unit']  # 神経細胞の総数
    num_unit_exc = architecture['num_unit_exc']  # 興奮性神経細胞の総数
    num_unit_inh = architecture['num_unit_inh']  # 抑制性神経細胞の総数

    # (B) 初期値の設定
    diff_cond_ampa = np.zeros((
        num_unit, num_unit_exc))
    conductances_ampa = np.zeros((
        num_unit, num_unit_exc))
    diff_cond_nmda = np.zeros((
        num_unit, num_unit_exc))
    conductances_nmda = np.zeros((
        num_unit, num_unit_exc))
    diff_cond_gaba = np.zeros((
        num_unit, num_unit_inh))
    conductances_gaba = np.zeros((
        num_unit, num_unit_inh))
    potentials= V_INIT * np.ones((num_unit, 1))
    y = np.hstack([
        diff_cond_nmda, conductances_nmda,
        diff_cond_ampa, conductances_ampa,
        diff_cond_gaba, conductances_gaba,
        potentials,
    ])
    last_spikes = -100 * np.ones((num_unit, 1))

    # (C) 結果保存用変数の準備
    results = {
        'potentials': [],
        'spikes': [],
    }

    for t in t_eval:
        # (D) 各時刻における計算
        y = np.hstack([
            diff_cond_nmda, conductances_nmda,
            diff_cond_ampa, conductances_ampa,
            diff_cond_gaba, conductances_gaba,
            potentials,
        ])

        # ノイズ電流の作成
        current_noise = np.random.binomial(
            n=1, p=0.1, size=potentials.shape
        ) * 7.0
        current_noise_weight = np.where(
            np.array(architecture['excitation_binary']) == 1, 0.5, 0.36
        ).reshape(current_noise.shape)
        current_noise = 1000 * current_noise_weight * current_noise

        # (D-1) 微分方程式の計算
        # コンダクタンスの微分, コンダクタンス, 膜電位, カルシウム濃度が対象
        y = solve_differential_equation(
            dydt=differentiate_working_memory_lif,
            t=t,
            y=y,
            delta_t=DELTA_T,
            method='rk4',
            last_spikes=last_spikes,
            current_noise=current_noise,
            dysfuncs_ampa=dysfuncs_ampa,
            dysfuncs_nmda=dysfuncs_nmda,
            dysfuncs_gaba=dysfuncs_gaba,
            architecture=architecture,
        )
        (diff_cond_nmda, conductances_nmda,
         diff_cond_ampa, conductances_ampa,
         diff_cond_gaba, conductances_gaba,
         potentials,) = np.split(
            y, architecture['split_list'], axis=1)

        # (D-2) スパイクの判定 (LIFモデルの処理)
        fuouki = (last_spikes <= t) & (t <= last_spikes + T_REF)
        seishi = (potentials < V_THERESHOLD) & (~ fuouki)
        katsudo = (potentials >= V_THERESHOLD) & (~ fuouki)
        potentials[katsudo] = V_ACT
        potentials[fuouki] = V_RESET
        potentials[seishi] = potentials[seishi]

        # (D-3) 直近の発火時刻の更新
        spikes = np.where(potentials == V_ACT, 1, 0)
        last_spikes = np.where(spikes == 1, t, last_spikes)

        # (E) 計算結果を保存
        results['potentials'].append(potentials)
        results['spikes'].append(spikes)

    # (F) 返り値の準備
    for key, value in results.items():
        # 結果の値をNumpy型に整形
        results[key] = np.hstack(value).T
    return results


# [コラム] ホジキン・ハックスリーモデル
def calc_alpha_m(V):
    # alpha_m = (0.1 * (V + 40)) / (- np.exp(- 0.1 * (V + 40)) + 1)
    alpha_m = (2.5 - 0.1 * V) / (np.exp(2.5 - 0.1 * V) - 1)
    return alpha_m

def calc_beta_m(V):
    # beta_m = 4 * np.exp(- (V + 65) / 18)
    beta_m = 4 * np.exp(- V / 18)
    return beta_m

def calc_alpha_h(V):
    # alpha_h = 0.07 * np.exp(- 0.05 * (V + 65))
    alpha_h = 0.07 * np.exp(- V / 20)
    return alpha_h

def calc_beta_h(V):
    # beta_h = 1 / (np.exp(- 0.1 * (V + 35)) + 1)
    beta_h = 1 / (np.exp(3 - 0.1 * V) + 1)
    return beta_h

def calc_alpha_n(V):
    # alpha_n = (0.01 * (V + 55)) / (- np.exp(- 0.1 * (V + 55)) + 1)
    alpha_n = (0.1 - 0.01 * V) / (np.exp(1 - 0.1 * V) - 1)
    return alpha_n

def calc_beta_n(V):
    # beta_n = 0.125 * np.exp(-0.0125 * (V + 65))
    beta_n = 0.125 * np.exp(- V / 80)
    return beta_n

def differentiate_hodgkin_huxley(t, y, square=False, **kwargs):
    E_m, m, h, n = y

    C = 1
    g_bar_leak = 0.3 # ms/cm^2
    E_leak = 10.6 # mV
    g_bar_na = 120 # ms/cm^2
    E_NA = 115 # mV
    g_bar_k = 36 # ms/cm^2
    E_k = -12 # mV

    # 電流の計算
    current_na = - g_bar_na * np.power(m, 3) * h *(E_m - E_NA)
    current_k = - g_bar_k * np.power(n, 4) * (E_m - E_k)
    current_leak = - g_bar_leak * (E_m - E_leak)

    if square:
        # 50ミリ秒ごとに生じる矩形波（オリジナル）
        current_ext = 0.0 if t % 100 < 50 else 20.0
    else:
        # 山﨑・五十嵐（2021）と同じ設定
        current_ext = 9.0

    # 微分方程式の計算
    dE_mdt = (1 / C) * (current_na + current_k + current_leak + current_ext)
    dmdt = calc_alpha_m(E_m) * (1 - m) - calc_beta_m(E_m) * m
    dhdt = calc_alpha_h(E_m) * (1 - h) - calc_beta_h(E_m) * h
    dndt = calc_alpha_n(E_m) * (1 - n) - calc_beta_n(E_m) * n

    dydt = np.hstack([dE_mdt, dmdt, dhdt, dndt])
    return dydt


def simulate_hodgkin_huxley(t_eval,
                            delta_t,
                            method='rk4'):
    """ホジキン・ハックスリーモデルを用いたシミュレーション

    Parameters
    ----------
    t_eval : np.ndarray or list
        評価の対象となる時刻を保存した一次元の配列
    delta_t : float
        ステップ幅 (ミリ秒)
    method : str
        'euler'の場合オイラー法，'rk4'の場合ルンゲ・クッタ法を使用

    Returns
    -------

    Note
    ----
    神経細胞の数は一つのみ対応

    時間幅（delta_t）の値について
    - Eulerの場合: ステップ幅0.1は厳しい。0.01は必要か
    - RK4の場合: ステップ幅0.1は厳しい。0.01は必要か

    定数値について
    - 現在は山﨑・五十嵐（2021）の数字を使っている（静止膜電位を0としている）
    - コメントアウトしている数式は「Juliaで学ぶ計算論的神経科学」を参考にしたもの

    なお，山﨑・五十嵐（2021）では，delta_t = 10μs = 0.01msでルンゲクッタ
    また「HHモデルをオイラー法で解こうとすると1μs程度は必要と」記載あり
    """
    y = np.hstack([-65, 0.05, 0.6, 0.32])
    results = {
        'E': [],
        'm': [],
        'h': [],
        'n': [],
    }
    for step, t in enumerate(t_eval):
        y = solve_differential_equation(
            dydt=differentiate_hodgkin_huxley,
            t=t,
            y=y,
            delta_t=delta_t,
            method=method,
        )
        results['E'].append(y[0])
        results['m'].append(y[1])
        results['h'].append(y[2])
        results['n'].append(y[3])
    return results


# [未掲載] 微分方程式ソルバーのデバッグ関数
def test_solver():
    """微分方程式ソルバーのデバッグ関数
    """
    pass


# [未掲載] イオンチャネルを用いたワーキングメモリモデル (CPSY TOKYOにて使用)
def calc_g_na(potential):
    """ナトリウムチャネルのコンダクタンスを返す
    """
    return G_MAX_NA / (1 + np.exp(- (potential - V_NA) / 7))


def test_calc_g_na():
    # 評価幅を決定
    potential = np.linspace(-100, 100, 10000)

    # ナトリウムのコンダクタンスの挙動
    plt.plot(potential, calc_g_na(potential))
    plt.show()

    # I_NAの挙動
    plt.plot(potential, - calc_g_na(potential) * (potential - E_NA))
    plt.show()


def differentiate_dCadt(t, y, **kwargs):
    """カルシウム濃度（カリウムチャネルのコンダクタンス）の微分方程式

    Parameters
    ----------
    t : float
        時刻
    y : np.ndarray
        カルシウム濃度（カリウムチャネルのコンダクタンス）
    last_spikes : np.ndarray
        直近の発火時刻の更新

    Returns
    -------
    dCadt : np.ndarray
        カルシウム濃度（カリウムチャネルのコンダクタンス）の微分
    """
    # 引数の取得
    ca = y
    last_spikes = kwargs['last_spikes']

    # スパイクの判定
    check_spike = np.where(
        (t - 2 * DELTA_T <= last_spikes) & (last_spikes <= t - DELTA_T),
        1.0,
        0.0
    )
    check_spike = check_spike / DELTA_T

    # 微分の計算
    # dCadt = ALPHA_CA * BETA_K * check_spike - ca / TAU_CA
    dCadt = ALPHA_CA * check_spike - ca / TAU_CA
    return dCadt


def simulate_channel_ca(t_eval):
    """カリウムチャネルの振る舞いをシミュレーション

    Parameters
    ----------
    t_eval : np.ndarray or list
        評価の対象となる時刻を保存した一次元の配列

    Returns
    -------
    results : dict
        カリウムコンダクタンスの変化をリストとして値に保存した辞書
    """
    # (B) 初期値の設定
    ca = 0
    last_spike = -100

    # (C) 結果保存用変数の準備
    results = {
        'ca': [],
    }

    for t in t_eval:
        # (D) 各時刻における計算
        # (D-1) 微分方程式による更新
        ca = solve_differential_equation(
            dydt= differentiate_dCadt,
            t=t,
            y=ca,
            delta_t=DELTA_T,
            method='rk4',
            last_spikes=last_spike,
        )

        # (D-2) スパイクの判定
        # 20-30msecは常時発火と仮定
        spike = 1.0 if 20.0 < t and t < 30.0 else 0.0

        # (D-3) 直近の発火時刻の更新
        last_spike = t if spike == 1 else last_spike

        # (E) 計算結果を保存
        results['ca'].append(ca)
    return results


def test_simulate_channel_ca():
    # 評価時間を定める
    t_eval = generate_t_eval(t_max=100)

    # 実行
    results = simulate_channel_ca(t_eval=t_eval)

    # プロット
    plt.figure(figsize=(12, 2))
    plt.plot(t_eval, results['ca'])
    plt.xlabel('時刻 [msec]')
    plt.ylabel('カルシウム濃度')
    plt.show()


def differentiate_working_memory_ion(t, y, **kwargs):
    """ワーキングメモリモデルの微分方程式

    コンダクタンスの微分p, コンダクタンスg, 膜電位v, カルシウム濃度caについて微分する
    状態yは，上記の変数を保存

    Parameters
    ----------
    t: float
        時刻
    y : np.ndarray
        NMDA/AMPA/GABA受容体のコンダクタンスの微分p, コンダクタンスg, 膜電位v, カルシウム濃度ca
    kwargs : dict
        ノイズ電流current_noiseと直近のスパイク時刻last_spikesを含むこと

    Returns
    -------
    dydt : np.ndarray
        コンダクタンスの微分p, コンダクタンスg, 膜電位v, カルシウム濃度caに関する微分
    """
    # 引数の取得
    current_noise = kwargs['current_noise']
    last_spikes = kwargs['last_spikes']
    dysfuncs_ampa = kwargs['dysfuncs_ampa']
    dysfuncs_nmda = kwargs['dysfuncs_nmda']
    dysfuncs_gaba = kwargs['dysfuncs_gaba']

    architecture = kwargs['architecture']
    set_exc = architecture['set_exc']
    set_inh = architecture['set_inh']

    (diff_cond_ampa, conductances_ampa,
     diff_cond_nmda, conductances_nmda,
     diff_cond_gaba, conductances_gaba,
     potentials, ca) = np.split(
         y, architecture['split_list'], axis=1)

    # (i) シナプスの影響度eの計算
    synapse_effects_ampa = calc_synaptic_effect(
        last_spikes=last_spikes[architecture['set_exc'], :],
        t=t,
        weights=architecture['weights'][:, architecture['set_exc']],
        g_max=G_MAX_AMPA * dysfuncs_ampa,
    )
    synapse_effects_nmda = calc_synaptic_effect(
        last_spikes=last_spikes[architecture['set_exc'], :],
        t=t,
        weights=architecture['weights'][:, architecture['set_exc']],
        g_max=G_MAX_NMDA * dysfuncs_nmda,
    )
    synapse_effects_gaba = calc_synaptic_effect(
        last_spikes=last_spikes[architecture['set_inh'], :],
        t=t,
        weights=architecture['weights'][:, architecture['set_inh']],
        g_max=G_MAX_GABA * dysfuncs_gaba,
    )

    # (ii) リガンド依存性コンダクタンスの微分fの計算（微分）
    dfdt_ampa = calc_dfdt(
        diff_cond=diff_cond_ampa,
        conductances=conductances_ampa,
        synapse_effects=synapse_effects_ampa,
        tau_1=TAU_1_AMPA,
        tau_2=TAU_2_AMPA,
    )
    dfdt_nmda = calc_dfdt(
        diff_cond=diff_cond_nmda,
        conductances=conductances_nmda,
        synapse_effects=synapse_effects_nmda,
        tau_1=TAU_1_NMDA,
        tau_2=TAU_2_NMDA,
    )
    dfdt_gaba = calc_dfdt(
        diff_cond=diff_cond_gaba,
        conductances=conductances_gaba,
        synapse_effects=synapse_effects_gaba,
        tau_1=TAU_1_GABA,
        tau_2=TAU_2_GABA,
    )

    # (iii) リガンド依存性コンダクタンスgの計算（微分）
    dgdt_ampa = diff_cond_ampa
    dgdt_nmda = diff_cond_nmda
    dgdt_gaba = diff_cond_gaba

    # (iv) その他のコンダクタンスの計算
    conductances_na = calc_g_na(potentials)
    conductances_kca = BETA_K * ca
    dCadt = differentiate_dCadt(
        t=t, y=ca, last_spikes=last_spikes
    )

    # (v) 電流の計算
    current_ampa = - np.sum(
        conductances_ampa * np.tile(
            potentials - E_AMPA, len(architecture['set_exc'])),
        axis=1,
        keepdims=True,
    )
    f_mg = calc_block_mg(potentials)
    current_nmda = - np.sum(
        conductances_nmda * np.tile(
            f_mg * (potentials - E_NMDA), len(architecture['set_exc'])),
        axis=1,
        keepdims=True,
    )
    current_gaba = - np.sum(
        conductances_gaba * np.tile(
            potentials - E_GABA, len(architecture['set_inh'])),
        axis=1,
        keepdims=True,
    )
    current_na = - conductances_na * (potentials - E_NA)
    current_kca = - conductances_kca * (potentials - E_KCA)
    current_leak = np.empty_like(current_nmda)
    current_leak[architecture['set_exc'], :] = \
        - G_MAX_LEAK_EXC * (potentials[architecture['set_exc'], :] - E_LEAK)
    current_leak[architecture['set_inh'], :] = \
        - G_MAX_LEAK_INH * (potentials[architecture['set_inh'], :] - E_LEAK)

    # キュー電流の計算。100-200m秒で電流を流す
    current_cue = \
        (C_EXC if 100.0 <= t and t <= 200.0 else 0.0) \
        * architecture['positions_cue'].reshape(-1, 1)

    # それぞれの電流をまとめる
    current_channel = \
        current_ampa + current_nmda + current_gaba \
         + current_na + current_kca + current_leak
    current = current_channel + current_cue + current_noise

    # (vi) 膜電位vの計算
    dvdt = np.empty_like(potentials)
    dvdt[architecture['set_exc'], :] = current[architecture['set_exc'], :] / C_EXC
    dvdt[architecture['set_inh'], :] = current[architecture['set_inh'], :] / C_INH

    # すべての微分をひとつの配列にまとめる
    dydt = np.hstack([
        dfdt_ampa, dgdt_ampa,
        dfdt_nmda, dgdt_nmda,
        dfdt_gaba, dgdt_gaba,
        dvdt, dCadt
    ])
    return dydt


def simulate_working_memory_ion(t_eval,
                                architecture,
                                dysfuncs_ampa=1.0,
                                dysfuncs_nmda=1.0,
                                dysfuncs_gaba=1.0):
    """ワークングメモリの挙動をシミュレーションする関数

    Parameters
    ----------
    t_eval : np.ndarray or list
        評価の対象となる時刻を保存した一次元の配列

    Returns
    -------
    results : dict
        膜電位やスパイクの変化をリストとして値に保存した辞書
    """
    # シードを固定する
    np.random.seed(SEED)

    # 神経細胞の個数の設定
    num_unit = architecture['num_unit']  # 神経細胞の総数
    num_unit_exc = architecture['num_unit_exc']  # 興奮性神経細胞の総数
    num_unit_inh = architecture['num_unit_inh']  # 抑制性神経細胞の総数

    # (B) 初期値の設定
    diff_cond_ampa = np.zeros((
        num_unit, num_unit_exc))
    conductances_ampa = np.zeros((
        num_unit, num_unit_exc))
    diff_cond_nmda = np.zeros((
        num_unit, num_unit_exc))
    conductances_nmda = np.zeros((
        num_unit, num_unit_exc))
    diff_cond_gaba = np.zeros((
        num_unit, num_unit_inh))
    conductances_gaba = np.zeros((
        num_unit, num_unit_inh))
    potentials= V_INIT * np.ones((num_unit, 1))
    ca = np.zeros((num_unit, 1))

    y = np.hstack([
        diff_cond_ampa, conductances_ampa,
        diff_cond_nmda, conductances_nmda,
        diff_cond_gaba, conductances_gaba,
        potentials, ca
    ])
    last_spikes = -100 * np.ones((num_unit, 1))

    # (C) 結果保存用変数の準備
    results = {
        'potentials': [],
        'spikes': [],
    }

    for t in t_eval:
        # (D) 各時刻における計算
        # 前時刻の膜電位を保存
        potentials_pre = potentials

        # ノイズ電流の作成
        current_noise = np.random.binomial(
                n=1, p=0.1, size=potentials.shape
            ) * 7.0
        current_noise_weight = np.where(
            np.array(architecture['excitation_binary']) == 1, 0.5, 0.36
        ).reshape(current_noise.shape)
        current_noise = 1000 * current_noise_weight * current_noise

        # (D-1) 微分方程式による更新
        # コンダクタンスの微分, コンダクタンス, 膜電位, カルシウム濃度が対象
        y = solve_differential_equation(
            dydt=differentiate_working_memory_ion,
            t=t,
            y=y,
            delta_t=DELTA_T,
            method='rk4',
            last_spikes=last_spikes,
            current_noise=current_noise,
            dysfuncs_ampa=dysfuncs_ampa,
            dysfuncs_nmda=dysfuncs_nmda,
            dysfuncs_gaba=dysfuncs_gaba,
            architecture=architecture,
        )
        (diff_cond_ampa, conductances_ampa,
         diff_cond_nmda, conductances_nmda,
         diff_cond_gaba, conductances_gaba,
         potentials, ca) = np.split(
            y, architecture['split_list'], axis=1)

        # (D-2) スパイクの判定
        spikes = np.where(
            (potentials >= V_THERESHOLD) & ~ (potentials_pre >= V_THERESHOLD),
            1,
            0)

        # (D-3) 直近の発火時刻の更新
        last_spikes = np.where(
            spikes == 1, t, last_spikes
        ).reshape(num_unit, 1)

        # (E) 計算結果を保存
        results['potentials'].append(potentials)
        results['spikes'].append(spikes)

    # (F) 返り値の準備
    for key, value in results.items():
        # 結果の値をNumpy型に整形
        results[key] = np.hstack(value).T
    return results
