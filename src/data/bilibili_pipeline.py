"""
B站字幕 → SIRS-M 真实参数拟合管道

Gemini 贡献。将 Video_to_Text 转录的字幕文本按时间戳抽取热梗频次，
利用 scipy L-BFGS-B 非线性优化在 SIRS-M ODE 空间中拟合 β, σ, γ, μ。

对齐「微尘哲学」：
- 时序抽取的滚动窗口平滑 → 对抗语音转文字的局部高频噪声（混沌）
- 参数约束边界 → 保证物理合理性（不产生负值溢出）
- 拟合 = 从混沌的经验数据中提取确定性结构（局部秩序建立）

用法：
    python src/data/bilibili_pipeline.py
    # 读取 data/raw/ 下的字幕 JSON，输出拟合参数 + 图到 data/processed/
"""

import os
import json
import re
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════
# 1. B站字幕时序抽取模块
# ═══════════════════════════════════════════════

def load_bilibili_transcripts(raw_dir="data/raw"):
    """加载 data/raw 目录下所有的B站字幕转录文件。

    期望格式：[{"text": "...", "start": 12.5, "duration": 2.1}, ...]
    """
    transcripts = []
    if not os.path.exists(raw_dir):
        print(f"[警告] 目录 {raw_dir} 不存在，正在创建空目录...")
        os.makedirs(raw_dir, exist_ok=True)
        return transcripts

    for file_name in os.listdir(raw_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(raw_dir, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    transcripts.append({
                        "video_id": os.path.splitext(file_name)[0],
                        "data": data,
                    })
            except Exception as e:
                print(f"[错误] 读取文件 {file_name} 失败: {e}")
    return transcripts


def extract_meme_timeline(transcripts, meme_keyword, bin_width_sec=30):
    """在字幕文本中按时间戳扫描目标热梗的提及率。

    Args:
        transcripts: load_bilibili_transcripts() 的输出
        meme_keyword: 热梗关键词（支持正则，如 "吗喽|马骝"）
        bin_width_sec: 时间分箱宽度（秒）

    Returns:
        t_data: 时间轴 (单位: 时间步)
        I_data: 归一化感染比例序列
    """
    pattern = re.compile(meme_keyword, re.IGNORECASE)
    all_hits = []
    max_time = 0

    for video in transcripts:
        for item in video["data"]:
            text = item.get("text", "")
            start = item.get("start", 0)
            max_time = max(max_time, start)
            matches = len(pattern.findall(text))
            if matches > 0:
                all_hits.append((start, matches))

    if not all_hits:
        print(f"[提示] 未在任何字幕中匹配到关键词: '{meme_keyword}'")
        return np.array([]), np.array([])

    num_bins = int(np.ceil(max_time / bin_width_sec)) + 1
    counts = np.zeros(num_bins)
    for start_time, count in all_hits:
        bin_idx = int(start_time // bin_width_sec)
        if bin_idx < num_bins:
            counts[bin_idx] += count

    # Rolling window smoothing
    counts_smoothed = (
        pd.Series(counts)
        .rolling(window=3, min_periods=1, center=True)
        .mean()
        .values
    )

    max_val = np.max(counts_smoothed)
    I_data = counts_smoothed / max_val if max_val > 0 else counts_smoothed
    t_data = np.arange(len(I_data))

    return t_data, I_data


# ═══════════════════════════════════════════════
# 2. SIRS-M 非线性拟合模块
# ═══════════════════════════════════════════════

def sirs_m_ode(t, y, beta, sigma, gamma, mu, N=1.0):
    """SIRS-M ODE 系统。

    dS/dt = -β·S·I/N + γ·R
    dI/dt =  β·S·I/N - σ·I + μ·R
    dR/dt =  σ·I - γ·R - μ·R
    """
    S, I, R = y
    dS_dt = -beta * S * I / N + gamma * R
    dI_dt = beta * S * I / N - sigma * I + mu * R
    dR_dt = sigma * I - gamma * R - mu * R
    return [dS_dt, dI_dt, dR_dt]


def fit_sirs_m_model(t_data, I_data, init_params=None):
    """用 L-BFGS-B 最小化经验数据与 ODE 轨迹之间的残差平方和。

    Args:
        t_data: 时间轴
        I_data: 经验感染比例序列
        init_params: [beta, sigma, gamma, mu] 初始猜测

    Returns:
        {"beta", "sigma", "gamma", "mu", "R0", "success", "fun"} 或 None
    """
    if len(t_data) == 0 or len(I_data) == 0:
        return None

    if init_params is None:
        init_params = [0.5, 0.2, 0.1, 0.05]

    bounds = [(0.01, 2.0), (0.01, 1.0), (0.0, 0.5), (0.0, 0.5)]

    I0 = max(0.001, I_data[0])
    y0 = [1.0 - I0, I0, 0.0]

    def loss_function(params):
        beta, sigma, gamma, mu = params
        sol = solve_ivp(
            fun=sirs_m_ode,
            t_span=(t_data[0], t_data[-1]),
            y0=y0,
            t_eval=t_data,
            args=(beta, sigma, gamma, mu),
            method="RK45",
        )
        if not sol.success:
            return 1e6
        I_fit = sol.y[1]
        return float(np.sum((I_data - I_fit) ** 2))

    res = minimize(loss_function, init_params, method="L-BFGS-B", bounds=bounds)

    if res.success:
        beta_fit, sigma_fit, gamma_fit, mu_fit = res.x
        r0 = beta_fit / sigma_fit if sigma_fit > 0 else 0.0
        return {
            "beta": float(beta_fit),
            "sigma": float(sigma_fit),
            "gamma": float(gamma_fit),
            "mu": float(mu_fit),
            "R0": float(r0),
            "success": True,
            "fun": float(res.fun),
        }
    return {"success": False, "message": str(res.message)}


# ═══════════════════════════════════════════════
# 3. 可视化 + 运行入口
# ═══════════════════════════════════════════════

def plot_fitting_results(t_data, I_data, fit_res, meme_keyword, save_path=None):
    """绘制经验数据 vs SIRS-M 拟合曲线。"""
    if not fit_res or not fit_res.get("success"):
        print("[错误] 无法绘制拟合结果：拟合不成功或数据为空。")
        return

    beta, sigma, gamma, mu = (
        fit_res["beta"], fit_res["sigma"], fit_res["gamma"], fit_res["mu"]
    )
    I0 = max(0.001, I_data[0])
    y0 = [1.0 - I0, I0, 0.0]

    sol = solve_ivp(
        fun=sirs_m_ode,
        t_span=(t_data[0], t_data[-1]),
        y0=y0,
        t_eval=t_data,
        args=(beta, sigma, gamma, mu),
        method="RK45",
    )

    plt.figure(figsize=(10, 5))
    plt.plot(t_data, I_data, "o", label="Bilibili Empirical Data", alpha=0.6)
    plt.plot(sol.t, sol.y[1], "-", label=f"SIRS-M Fitted ($R_0$={fit_res['R0']:.3f})", linewidth=2)
    plt.title(f"Meme '{meme_keyword}' Real Spreading Curve & SIRS-M Fit")
    plt.xlabel("Time Steps (30s per unit)")
    plt.ylabel("Infected Ratio I(t)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"[成功] 拟合图谱已保存至: {save_path}")
    plt.close()


def generate_demo_data():
    """生成虚拟字幕数据用于管道测试（无真实数据时的冷启动）。"""
    os.makedirs("data/raw", exist_ok=True)
    test_file = "data/raw/demo_bilibili_transcript.json"

    dummy_transcript = []
    for t in range(0, 600, 5):
        base = int(100 * np.exp(-((t - 150) / 80) ** 2))
        resurgence = int(30 * np.exp(-((t - 450) / 100) ** 2))
        noise = int(np.random.randint(0, 5))
        dummy_transcript.append({
            "text": "这个吗喽真是太搞笑了" * (base + resurgence + noise),
            "start": float(t),
            "duration": 5.0,
        })

    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(dummy_transcript, f, ensure_ascii=False)
    print(f"[提示] 已生成演示字幕数据: {test_file}")


if __name__ == "__main__":
    print("=== MemeticChaos B站数据拟合管道 ===\n")

    # 检查是否有真实数据
    raw_dir = "data/raw"
    has_real_data = os.path.isdir(raw_dir) and any(
        f.endswith(".json") for f in os.listdir(raw_dir)
    )

    if not has_real_data:
        print("[提示] 未检测到真实字幕，生成演示数据用于验证管道...\n")
        generate_demo_data()

    # 1. 加载
    transcripts = load_bilibili_transcripts()
    print(f"[加载] 共发现 {len(transcripts)} 个字幕文件")

    # 2. 提取
    target_meme = "吗喽"
    t_data, I_data = extract_meme_timeline(transcripts, target_meme, bin_width_sec=15)

    if len(t_data) == 0:
        print("[错误] 未能提取到时序数据，退出。")
        exit(1)

    print(f"[抽取] 关键词 '{target_meme}': {len(t_data)} 个时间步")

    # 3. 拟合
    print(f"[执行] 正在拟合 SIRS-M 非线性参数...")
    fit_results = fit_sirs_m_model(t_data, I_data)

    if fit_results and fit_results.get("success"):
        print(f"\n{'='*40}")
        print(f"  拟合结果: {target_meme}")
        print(f"{'='*40}")
        print(f"  传播率 β:        {fit_results['beta']:.4f}")
        print(f"  消亡率 σ:        {fit_results['sigma']:.4f}")
        print(f"  重新敏感率 γ:    {fit_results['gamma']:.4f}")
        print(f"  变异复燃率 μ:    {fit_results['mu']:.4f}")
        print(f"  基本再生数 R₀:   {fit_results['R0']:.4f}")
        print(f"  相变状态:        {'[激活] 模因爆发传播' if fit_results['R0'] > 1.0 else '[衰减] 未能建立秩序'}")
        print(f"  残差平方和:      {fit_results['fun']:.6f}")

        # 4. 绘图
        os.makedirs("data/processed", exist_ok=True)
        save_path = f"data/processed/{target_meme}_fit_curve.png"
        plot_fitting_results(t_data, I_data, fit_results, target_meme, save_path)
    else:
        print(f"[错误] 拟合失败: {fit_results.get('message', '未知错误') if fit_results else '无数据'}")
