import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter, find_peaks, peak_prominences
from csv_processor import StreamlitRamanSpectrumProcessor

st.set_page_config(
    page_title="ラマンピークキャリブレーション",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("ラマンピークキャリブレーション")
st.markdown("エタノールを用いてピーク位置と波数をキャリブレーションします")
st.header("データ入力")
uploaded_file = st.file_uploader("CSVファイルをアップロード", type=['csv'], help="ラマンスペクトルのCSVをアップロードしてください")

# ------------------------------------------------------------
# 自動ピーク検出ヘルパー（2次微分＋卓立度）
# ------------------------------------------------------------
def detect_peaks_sd_prom(pixel_index, spectrum, smooth_win=11,
                         deriv_thresh=20.0, prom_thresh=10.0, min_distance=20):
    """
    2次微分(-d2)の極大＋prominenceで自動ピーク検出
    戻り値: (detected_indices, second_derivative, prominences_for_detected)
    """
    x = np.asarray(pixel_index)
    y = np.asarray(spectrum, dtype=float)

    # Savitzky–Golay のウィンドウ安全化（奇数・データ長未満）
    wl = int(smooth_win)
    if wl % 2 == 0:
        wl += 1
    max_wl = max(5, len(y) - 1 - ((len(y) - 1) % 2))
    wl = max(5, min(wl, max_wl))

    # 2次微分（polyorder=2）
    d2 = savgol_filter(y, wl, polyorder=2, deriv=2)

    # まず -d2 のピーク候補（距離でスパース化）
    peaks_all, _ = find_peaks(-d2, distance=int(max(1, min_distance)))

    if deriv_thresh is not None and deriv_thresh > 0:
        # 高さ（= -d2 の値）で閾値
        peaks, _ = find_peaks(-d2, height=deriv_thresh, distance=int(max(1, min_distance)))
    else:
        peaks = peaks_all

    if peaks.size == 0:
        return np.array([], dtype=int), d2, np.array([])

    # 各ピークの卓立度
    prom = peak_prominences(-d2, peaks)[0] if len(peaks) > 0 else np.array([])

    # prominence 閾値でフィルタ
    if prom.size > 0 and prom_thresh is not None and prom_thresh > 0:
        mask = prom >= prom_thresh
        peaks = peaks[mask]
        prom = prom[mask]

    return peaks.astype(int), d2, prom

# ------------------------------------------------------------
# 5次フィット + 摂動（SSE最小化, cm^-1評価）
# ------------------------------------------------------------
def _wl_to_cm1_float(laser_nm, wavelength_nm):
    """float精度で波長→ラマンシフト(cm^-1)へ（csv_processor の int化を回避）"""
    wl = np.asarray(wavelength_nm, dtype=float)
    wl = np.clip(wl, 1e-6, None)  # 0割防止
    return 1e7/laser_nm - 1e7/wl

def fit_poly_with_jitter_cm1(processor, pixels, target_cm1,
                             degree=5, tol_cm1=2.0, max_jitter_px=2,
                             max_iter=25, subpixel=False):
    """
    目的:
        pixel→wavelength を degree 次多項式でフィット。
        その上で各ピークの pixel に ±max_jitter_px（必要ならサブピクセル）の
        微小摂動を許容し、cm^-1 誤差の二乗和(SSE)を最小に。

    評価:
        SSE(cm^-1) = sum((calc_cm1 - target_cm1)^2)
        収束: max|誤差| <= tol_cm1  または 最大反復到達

    戻り:
        coeffs: np.polyfit の係数（wavelength = P(pixel)）
        metrics: dict（SSE, RMSE, MaxAbs, iterations, converged, 最終pixels, residuals(cm^-1), history）
        calc_cm1_final: 各ピークの最終推定cm^-1
    """
    px = np.asarray(pixels, dtype=float)
    wn = np.asarray(target_cm1, dtype=float)

    # 安全化
    if len(px) < 2:
        raise ValueError("ピーク数が不足しています。最低2点が必要です。")
    deg = min(int(degree), len(px) - 1)

    # x昇順
    order = np.argsort(px)
    px = px[order]
    wn = wn[order]

    # 初期フィット（波長領域）
    y_w = processor.wavenumber_to_wavelength(wn)  # ここは元のロジック踏襲
    coeffs = np.polyfit(px, y_w, deg)

    def eval_metrics(px_, coeffs_):
        wave_hat = np.polyval(coeffs_, px_)
        cm1_hat = _wl_to_cm1_float(processor.laser_wavelength, wave_hat)
        resid = cm1_hat - wn
        sse = float(np.sum(resid**2))
        rmse = float(np.sqrt(np.mean(resid**2)))
        max_abs = float(np.max(np.abs(resid)))
        return cm1_hat, resid, sse, rmse, max_abs

    history = []
    for it in range(max_iter):
        cm1_hat, resid, sse, rmse, max_abs = eval_metrics(px, coeffs)
        history.append((sse, rmse, max_abs))
        if max_abs <= tol_cm1:
            return coeffs, {
                "iterations": it,
                "sse": sse, "rmse": rmse, "max_abs": max_abs,
                "pixels": px.copy(), "residuals": resid.copy(),
                "converged": True, "history": history
            }, cm1_hat

        improved_any = False
        for i in range(len(px)):
            base_px = px[i]
            best_dx = 0.0
            best_tuple = (sse, rmse, max_abs, coeffs)

            # 候補：整数 ±max_jitter_px
            candidates = list(range(-max_jitter_px, max_jitter_px + 1))
            # サブピクセル（任意）
            if subpixel:
                candidates += [d/4.0 for d in range(-2, 3)]  # ±0.5 を0.25刻み

            tried = set()
            for dx in candidates:
                if dx in tried:
                    continue
                tried.add(dx)
                new_px = base_px + dx

                # 単調性（隣接と交差しない）維持
                if i > 0 and new_px <= px[i-1]:
                    continue
                if i < len(px)-1 and new_px >= px[i+1]:
                    continue

                px_try = px.copy()
                px_try[i] = new_px

                # 再フィット
                coeffs_try = np.polyfit(px_try, y_w, deg)
                _, _, sse_try, rmse_try, max_abs_try = eval_metrics(px_try, coeffs_try)

                if sse_try + 1e-9 < best_tuple[0]:
                    best_dx = dx
                    best_tuple = (sse_try, rmse_try, max_abs_try, coeffs_try)

            if best_dx != 0.0:
                px[i] = px[i] + best_dx
                coeffs = best_tuple[3]
                improved_any = True

        if not improved_any:
            break

    # 終了（未達ならベストを返す）
    cm1_hat, resid, sse, rmse, max_abs = eval_metrics(px, coeffs)
    return coeffs, {
        "iterations": it+1,
        "sse": sse, "rmse": rmse, "max_abs": max_abs,
        "pixels": px.copy(), "residuals": resid.copy(),
        "converged": (max_abs <= tol_cm1),
        "history": history
    }, cm1_hat


# Initialize processor
processor = StreamlitRamanSpectrumProcessor()

# Sidebar for configuration
with st.sidebar:
    st.header("設定")
    laser_wavelength = st.selectbox("レーザー波長 (nm)", options=[532, 785, 830], index=0)
    processor.laser_wavelength = laser_wavelength
    
    # -------------------------------
    # 自動ピーク検出（サイドバー）
    # -------------------------------
    if uploaded_file is not None:
        st.header("自動ピーク検出")
        smooth_win = st.number_input("平滑化ウィンドウ(奇数)", min_value=1, max_value=501, value=25, step=2, key="auto_smooth_win")
        deriv_thresh = st.number_input("2次微分閾値(高さ)", min_value=0, max_value=1000, value=5, step=1, key="auto_deriv_thresh")
        prom_thresh  = st.number_input("卓立度閾値", min_value=0, max_value=1000, value=5, step=1, key="auto_prom_thresh")
        min_distance = st.number_input("最小ピーク間隔(ピクセル)", min_value=1, max_value=10, value=1, step=1, key="auto_min_distance")

        # 反映する個数（prominence降順の上位）
        default_k = 10  # 既定ピークリストの長さに合わせて10を推奨
        top_k = st.number_input("反映するピーク数(上位prominence順)", min_value=1, max_value=200, value=int(default_k), step=1, key="auto_top_k")


# Main content area
if uploaded_file is not None:
    pixel_index, spectrum_data = processor.read_csv_data(uploaded_file)
    
    if pixel_index is not None and spectrum_data is not None:
        st.success(f"✅ データを読み込みました！ {len(pixel_index)} 点")

        # ▼ ロック集合の初期化
        if "locked_pixels" not in st.session_state:
            st.session_state["locked_pixels"] = set()
        if "locked_wavenumbers" not in st.session_state:
            st.session_state["locked_wavenumbers"] = set()

        # ファイル切替時の初期化
        if st.session_state.get("last_uploaded_name") != uploaded_file.name:
            st.session_state["manual_peaks_idx"] = []        # 手動追加（インデックス）
            st.session_state["excluded_auto_peaks"] = set()  # 自動検出からの除外（インデックス）
            for k in ["auto_peaks_idx", "auto_peaks_prom", "auto_d2",
                      "peaks_applied", "matched_pixels", "matched_wavenumbers", "peak_rois"]:
                st.session_state.pop(k, None)
            st.session_state["locked_pixels"] = set()
            st.session_state["locked_wavenumbers"] = set()
            st.session_state["last_uploaded_name"] = uploaded_file.name

        # 自動検出は毎回実行（設定変更に追従）
        det_idx, d2, det_prom = detect_peaks_sd_prom(
            pixel_index=pixel_index,
            spectrum=spectrum_data,
            smooth_win=st.session_state.get("auto_smooth_win", 11),
            deriv_thresh=st.session_state.get("auto_deriv_thresh", 20.0),
            prom_thresh=st.session_state.get("auto_prom_thresh", 10.0),
            min_distance=st.session_state.get("auto_min_distance", 20),
        )
        st.session_state["auto_peaks_idx"] = det_idx
        st.session_state["auto_peaks_prom"] = det_prom
        st.session_state["auto_d2"] = d2

        # --------------------------------------------------
        # 検出結果の表示＋手動制御（「適用」まではここだけ表示）
        # --------------------------------------------------
        if "auto_peaks_idx" in st.session_state:
            det_idx = st.session_state["auto_peaks_idx"]
            det_prom = st.session_state.get("auto_peaks_prom", np.array([]))
            d2 = st.session_state.get("auto_d2", None)

            if det_idx.size == 0:
                st.warning("ピークが見つかりません。閾値を下げるか、ウィンドウ/間隔を調整してください。")
            else:
                # prominence降順→上位K→x昇順
                if det_prom.size == det_idx.size and det_prom.size > 0:
                    order = np.argsort(det_prom)[::-1]
                    det_idx_sorted = det_idx[order]
                    det_prom_sorted = det_prom[order]
                else:
                    det_idx_sorted = det_idx
                    det_prom_sorted = np.zeros_like(det_idx_sorted, dtype=float)

                top_k = st.session_state.get("auto_top_k", 8)
                sel_idx = det_idx_sorted[:int(top_k)]
                sel_prom = det_prom_sorted[:int(top_k)]
                asc = np.argsort(sel_idx)
                sel_idx = sel_idx[asc]
                sel_prom = sel_prom[asc]

                # 除外セット／手動追加の状態
                if "excluded_auto_peaks" not in st.session_state:
                    st.session_state["excluded_auto_peaks"] = set()
                if "manual_peaks_idx" not in st.session_state:
                    st.session_state["manual_peaks_idx"] = []
                excluded = st.session_state["excluded_auto_peaks"]

                # レイアウト：左（広）／右（狭）
                col_plot, col_ctrl = st.columns([4, 1])

                # ---- 左：グラフ（上：原スペクトル, 下：-2次微分） ----
                with col_plot:
                    valid_auto_idx = np.array([i for i in sel_idx if i not in excluded], dtype=int)
                    manual_idx = np.array(st.session_state.get("manual_peaks_idx", []), dtype=int)
                    excluded_list = np.array([i for i in sel_idx if i in excluded], dtype=int)

                    fig_auto = make_subplots(
                        rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.08
                    )
                    # 上段：原スペクトル
                    fig_auto.add_trace(
                        go.Scatter(x=pixel_index, y=spectrum_data, mode="lines",
                                   name="スペクトル", line=dict(width=2)),
                        row=1, col=1
                    )
                    if len(valid_auto_idx) > 0:
                        fig_auto.add_trace(
                            go.Scatter(x=pixel_index[valid_auto_idx], y=spectrum_data[valid_auto_idx],
                                       mode="markers", name="検出ピーク（有効）",
                                       marker=dict(size=9, symbol="circle")),
                            row=1, col=1
                        )
                    if len(excluded_list) > 0:
                        fig_auto.add_trace(
                            go.Scatter(x=pixel_index[excluded_list], y=spectrum_data[excluded_list],
                                       mode="markers", name="除外ピーク",
                                       marker=dict(size=9, symbol="x")),
                            row=1, col=1
                        )
                    if len(manual_idx) > 0:
                        fig_auto.add_trace(
                            go.Scatter(x=pixel_index[manual_idx], y=spectrum_data[manual_idx],
                                       mode="markers+text", text=["手動"]*len(manual_idx),
                                       textposition="top center",
                                       name="手動ピーク", marker=dict(size=12, symbol="star")),
                            row=1, col=1
                        )

                    # 下段：-2次微分
                    if d2 is not None:
                        fig_auto.add_trace(
                            go.Scatter(x=pixel_index, y=-d2, mode="lines", name="-2次微分", line=dict(width=1)),
                            row=2, col=1
                        )
                        fig_auto.add_hline(y=st.session_state.get("auto_deriv_thresh", 20.0),
                                           line_dash="dash", line_color="gray", row=2, col=1)
                        for x0 in pixel_index[sel_idx]:
                            fig_auto.add_vline(x=x0, line_dash="dot", line_color="red", opacity=0.4, row=2, col=1)

                    fig_auto.update_xaxes(title_text="ピクセル位置", row=2, col=1)
                    fig_auto.update_yaxes(title_text="強度(a.u.)", row=1, col=1)
                    fig_auto.update_yaxes(title_text="-2次微分", row=2, col=1)
                    fig_auto.update_layout(height=620, showlegend=True)
                    st.plotly_chart(fig_auto, use_container_width=True)

                # ---- 右：コンパクトなコントロール ----
                with col_ctrl:
                    # 既定ピーク順チェック（手動調整の上）＋プレビュー
                    st.checkbox(
                        "増加順",
                        value=False,
                        key="chk_increasing",
                        help="オンの場合、既定のエタノール既知ピーク（default_ethanol_peaks）を逆順に適用します。"
                    )
                    _preview = processor.default_ethanol_peaks[:]
                    if st.session_state.get("chk_increasing", False):
                        _preview = list(reversed(_preview))
                    st.caption("既定ピーク（適用順）: " + ", ".join(f"{v:.1f}" for v in _preview))

                    st.markdown("**🔧 手動調整**")

                    tabs = st.tabs(["追加", "除外/復活"])
                    # --- 手動追加 ---
                    with tabs[0]:
                        add_px = st.number_input(
                            "ピクセル", min_value=int(pixel_index.min()),
                            max_value=int(pixel_index.max()),
                            value=int(pixel_index[len(pixel_index)//2]),
                            step=1, key="add_px_input_compact"
                        )
                        snap_win = st.number_input("スナップ±", min_value=0, max_value=100, value=2, step=1, key="snap_win_compact")
                        if st.button("＋ 追加", key="btn_add_manual_peak_compact"):
                            idx0 = int(np.argmin(np.abs(pixel_index - add_px)))
                            if snap_win > 0:
                                w0 = max(0, idx0 - snap_win)
                                w1 = min(len(spectrum_data), idx0 + snap_win + 1)
                                local = spectrum_data[w0:w1]
                                idx0 = w0 + int(np.argmax(local))
                            # 近接チェック（最小ピーク間隔）
                            min_dist = int(st.session_state.get("auto_min_distance", 20))
                            exists_near_auto = any(abs(idx0 - int(i)) < min_dist for i in sel_idx)
                            exists_near_manual = any(abs(idx0 - int(i)) < min_dist for i in st.session_state.get("manual_peaks_idx", []))
                            if exists_near_auto or exists_near_manual:
                                st.warning("近傍に既存のピークがあります。")
                            else:
                                st.session_state["manual_peaks_idx"].append(int(idx0))
                                st.success(f"手動ピーク: pixel={int(pixel_index[idx0])}")
                                st.rerun()

                        # 手動ピークの簡易削除
                        if st.session_state.get("manual_peaks_idx"):
                            del_sel = st.selectbox(
                                "削除対象",
                                options=list(range(len(st.session_state["manual_peaks_idx"]))),
                                format_func=lambda i: f"#{i+1} : px {int(pixel_index[st.session_state['manual_peaks_idx'][i]])}",
                                key="del_manual_sel_compact"
                            )
                            if st.button("🗑️ 削除", key="btn_delete_manual_peak_compact"):
                                removed_idx = st.session_state["manual_peaks_idx"].pop(del_sel)
                                st.success(f"削除: pixel={int(pixel_index[removed_idx])}")
                                st.rerun()

                    # --- 除外/復活（自動検出） ---
                    with tabs[1]:
                        if len(sel_idx) == 0:
                            st.caption("上位Kの候補なし")
                        else:
                            options = []
                            for i, idx in enumerate(sel_idx):
                                status = "除外中" if idx in excluded else "有効"
                                options.append(f"候補{i+1}: px {int(pixel_index[idx])} ({spectrum_data[idx]:.3f}) - {status}")
                            chosen = st.selectbox("対象", options=list(range(len(sel_idx))),
                                                  format_func=lambda k: options[k],
                                                  key="exclude_sel_compact")
                            chosen_idx = sel_idx[chosen]
                            if chosen_idx in excluded:
                                if st.button("↩️ 復活", key="btn_restore_peak_compact"):
                                    excluded.remove(chosen_idx)
                                    st.session_state["excluded_auto_peaks"] = excluded
                                    st.success("復活しました。")
                                    st.rerun()
                            else:
                                if st.button("🚫 除外", key="btn_exclude_peak_compact"):
                                    excluded.add(chosen_idx)
                                    st.session_state["excluded_auto_peaks"] = excluded
                                    st.success("除外しました。")
                                    st.rerun()

                # ---- 更新（適用）ボタン ----
                reset_wn_on_apply = False
                if st.button("更新", use_container_width=True):
                    # ===== 重要な修正点 =====
                    # 既に適用済みなら、手動編集を含む matched_pixels をそのまま尊重する
                    # （以前のように自動候補＋手動候補から毎回再構成しない）
                    if st.session_state.get("peaks_applied", False) and "matched_pixels" in st.session_state:
                        new_pixels = list(st.session_state.matched_pixels)  # 手動変更を保持
                    else:
                        # 初回適用のみ：自動候補＋手動候補から構成
                        valid_auto_idx = np.array([i for i in sel_idx if i not in excluded], dtype=int)
                        manual_idx = np.array(st.session_state.get("manual_peaks_idx", []), dtype=int)
                        combined_idx = np.unique(np.concatenate([valid_auto_idx, manual_idx])).astype(int)
                        combined_idx.sort()
                        if combined_idx.size == 0:
                            st.warning("反映対象が空です。")
                            st.stop()
                        new_pixels = pixel_index[combined_idx].astype(int).tolist()

                    # ▼ ロックを考慮（長さが同じときのみロック位置を上書き）
                    if "matched_pixels" in st.session_state and st.session_state.get("peaks_applied", False):
                        old_pixels = st.session_state.get("matched_pixels", [])
                        if len(old_pixels) == len(new_pixels):
                            locked = st.session_state.get("locked_pixels", set())
                            for i_lock in locked:
                                if 0 <= i_lock < len(new_pixels):
                                    new_pixels[i_lock] = int(old_pixels[i_lock])
                        else:
                            st.session_state["locked_pixels"] = set()

                    st.session_state.matched_pixels = new_pixels

                    # 波数配列の長さ整合（ロック考慮）
                    n = len(st.session_state.matched_pixels)
                    need_reset_wn = (
                        reset_wn_on_apply or
                        ("matched_wavenumbers" not in st.session_state) or
                        (len(st.session_state.matched_wavenumbers) != n)
                    )
                    if need_reset_wn:
                        default_peaks = processor.default_ethanol_peaks[:]
                        if st.session_state.get("chk_increasing", False):
                            default_peaks = list(reversed(default_peaks))

                        if n <= len(default_peaks):
                            new_wn = [float(v) for v in default_peaks[:n]]
                        else:
                            pad = [float(default_peaks[-1])] * (n - len(default_peaks))
                            new_wn = [float(v) for v in default_peaks] + pad

                        if "matched_wavenumbers" in st.session_state and not reset_wn_on_apply:
                            old_wn = st.session_state.get("matched_wavenumbers", [])
                            locked_w = st.session_state.get("locked_wavenumbers", set())
                            for i_lock in locked_w:
                                if 0 <= i_lock < n and i_lock < len(old_wn):
                                    new_wn[i_lock] = float(old_wn[i_lock])
                        st.session_state.matched_wavenumbers = new_wn

                    # ROI は既存を尊重しつつ不足分のみ作成
                    if "peak_rois" not in st.session_state:
                        st.session_state.peak_rois = {}
                    roi_size = 100
                    for i, px in enumerate(st.session_state.matched_pixels):
                        if i not in st.session_state.peak_rois:
                            st.session_state.peak_rois[i] = {
                                "min": max(int(px - roi_size), int(min(pixel_index))),
                                "max": min(int(px + roi_size), int(max(pixel_index))),
                            }

                    st.session_state.peaks_applied = True
                    st.success(f"{len(st.session_state.matched_pixels)} 個のピークを反映しました。")
                    st.rerun()

        # --------------------------------------------------
        # ここから下は「適用」後にのみ表示
        # --------------------------------------------------
        if st.session_state.get("peaks_applied", False) and st.session_state.get("matched_pixels"):
            # 全体スペクトル（適用済みのピークを表示）
            st.header("ラマンスペクトル（Pixel）")
            fig_main = go.Figure()
            fig_main.add_trace(go.Scatter(x=pixel_index, y=spectrum_data, mode='lines',
                                          name='ラマンスペクトル', line=dict(color='lightblue', width=2)))
            colors = ['red','green','orange','purple','brown','pink','gray','cyan','magenta']
            for i, (pixel, wavenumber) in enumerate(zip(st.session_state.matched_pixels, st.session_state.matched_wavenumbers)):
                color = colors[i % len(colors)]
                spectrum_intensity = np.interp(pixel, pixel_index, spectrum_data)
                fig_main.add_trace(go.Scatter(x=[pixel], y=[spectrum_intensity],
                                              mode='markers+text',
                                              name=f'ピーク {i+1}: {wavenumber} cm⁻¹',
                                              marker=dict(color=color, size=12, line=dict(width=2, color='white')),
                                              text=[f'P{i+1}'], textposition="top center",
                                              textfont=dict(size=12, color='white')))
                fig_main.add_vline(x=pixel, line_dash="dash", line_color=color, line_width=2, opacity=0.8)
                if i in st.session_state.peak_rois:
                    roi = st.session_state.peak_rois[i]
                    roi_mask = (pixel_index >= roi['min']) & (pixel_index <= roi['max'])
                    fig_main.add_trace(go.Scatter(x=pixel_index[roi_mask], y=spectrum_data[roi_mask],
                                                  mode='lines', name=f'ROI {i+1}',
                                                  line=dict(color=color, width=4), opacity=0.6, showlegend=False))
            fig_main.update_layout(title="ピーク位置とROIを重ねたラマンスペクトル",
                                   xaxis_title="ピクセル位置", yaxis_title="強度", height=500, showlegend=True)
            st.plotly_chart(fig_main, use_container_width=True)

            # 各ピークのROI付き調整（＋ボタン付き）
            st.subheader("🔍 各ピークのROI付き調整")
            # 行：左にタブ、右に＋ボタン
            col_tabs, col_addbtn = st.columns([6, 1])
            with col_tabs:
                tab_labels = [f"ピーク {i+1}" for i in range(len(st.session_state.matched_pixels))]
                tabs = st.tabs(tab_labels)
            with col_addbtn:
                # 追加ボタン：タブ列の右側
                if st.button("＋", key="btn_add_peak_roi", help="新しいピークを追加します"):
                    # 既存ピークから十分離れた候補のうち、強度最大のピクセルを採用
                    used_idx = np.array([int(np.argmin(np.abs(pixel_index - px))) for px in st.session_state.matched_pixels], dtype=int)
                    min_dist = int(st.session_state.get("auto_min_distance", 20))
                    candidates = np.ones_like(pixel_index, dtype=bool)

                    # 端の安全域（ROI=100のため）
                    edge_pad = 100
                    candidates[:edge_pad] = False
                    candidates[-edge_pad:] = False

                    for ui in used_idx:
                        lo = max(0, ui - min_dist)
                        hi = min(len(pixel_index), ui + min_dist + 1)
                        candidates[lo:hi] = False

                    if np.any(candidates):
                        cand_idx = np.argmax(np.where(candidates, spectrum_data, -np.inf))
                        new_px_val = int(pixel_index[cand_idx])
                    else:
                        # 候補が無い場合は中央近傍
                        center_idx = int(len(pixel_index) // 2)
                        new_px_val = int(pixel_index[center_idx])

                    # 波数は default_ethanol_peaks（増加順チェックに追従）から割当
                    default_peaks = processor.default_ethanol_peaks[:]
                    if st.session_state.get("chk_increasing", False):
                        default_peaks = list(reversed(default_peaks))
                    next_i = len(st.session_state.matched_wavenumbers) if "matched_wavenumbers" in st.session_state else 0
                    if next_i < len(default_peaks):
                        new_wn_val = float(default_peaks[next_i])
                    else:
                        new_wn_val = float(default_peaks[-1])

                    # 追加反映（ソースオブトゥルースへ追記）
                    st.session_state.matched_pixels.append(int(new_px_val))
                    if "matched_wavenumbers" not in st.session_state:
                        st.session_state.matched_wavenumbers = []
                    st.session_state.matched_wavenumbers.append(float(new_wn_val))

                    # ROI 付与
                    roi_size = 100
                    i_new = len(st.session_state.matched_pixels) - 1
                    if "peak_rois" not in st.session_state:
                        st.session_state.peak_rois = {}
                    st.session_state.peak_rois[i_new] = {
                        'min': max(int(new_px_val - roi_size), int(min(pixel_index))),
                        'max': min(int(new_px_val + roi_size), int(max(pixel_index)))
                    }

                    st.success(f"新規ピークを追加: pixel={new_px_val}, wavenumber={new_wn_val:.1f} cm⁻¹")
                    st.rerun()

            # 既存タブ内容
            for i, tab in enumerate(tabs):
                with tab:
                    # ROI 初期化（無ければ適用済みピクセルを基準に自動設定）
                    if i not in st.session_state.peak_rois:
                        peak_pixel = st.session_state.matched_pixels[i]
                        roi_size = 100
                        st.session_state.peak_rois[i] = {
                            'min': max(int(peak_pixel - roi_size), int(min(pixel_index))),
                            'max': min(int(peak_pixel + roi_size), int(max(pixel_index)))
                        }

                    st.markdown(f"**🎯 ピーク {i+1} のコントロール**")
                    col_peak_pos, col_wavenumber = st.columns(2)

                    # ---- ピクセル位置 ----
                    with col_peak_pos:
                        st.markdown("**ピーク位置:**")
                        current_pixel = st.session_state.matched_pixels[i]
                        new_pixel = st.number_input(
                            "ピクセル位置",
                            min_value=int(min(pixel_index)),
                            max_value=int(max(pixel_index)),
                            value=int(current_pixel),
                            step=1,
                            key=f"pixel_input_{i}",
                            help="ピクセル位置を入力してください（ROIは±100ピクセルで自動調整）"
                        )
                        if new_pixel != current_pixel:
                            st.session_state.matched_pixels[i] = int(new_pixel)  # ← 手動編集を即時反映（ソースオブトゥルース）
                            roi_size = 100
                            new_roi_min = max(int(new_pixel - roi_size), int(min(pixel_index)))
                            new_roi_max = min(int(new_pixel + roi_size), int(max(pixel_index)))
                            st.session_state.peak_rois[i]['min'] = new_roi_min
                            st.session_state.peak_rois[i]['max'] = new_roi_max
                            st.rerun()

                    # ---- 波数（cm⁻¹） ----
                    with col_wavenumber:
                        st.markdown("**波数:**")
                        new_wavenumber = st.number_input(
                            "波数 (cm⁻¹)",
                            value=float(st.session_state.matched_wavenumbers[i]),
                            step=0.1,
                            format="%.1f",
                            key=f"wavenumber_input_{i}"
                        )
                        st.session_state.matched_wavenumbers[i] = float(new_wavenumber)
                        current_wavelength = processor.wavenumber_to_wavelength(new_wavenumber)
                        st.caption(f"波長: {current_wavelength:.2f} nm")
                        spectrum_intensity = np.interp(st.session_state.matched_pixels[i], pixel_index, spectrum_data)
                        st.caption(f"ピーク強度: {spectrum_intensity:.1f}")

                    # ROI 設定とビュー
                    col_roi_settings, col_roi_plot = st.columns([1, 3])
                    with col_roi_settings:
                        st.markdown("**ROI設定:**")
                        roi = st.session_state.peak_rois[i]
                        roi_min = st.number_input("ROI最小", min_value=int(min(pixel_index)),
                                                  max_value=int(max(pixel_index)), value=roi['min'], step=1, key=f"roi_min_{i}")
                        roi_max = st.number_input("ROI最大", min_value=int(min(pixel_index)),
                                                  max_value=int(max(pixel_index)), value=roi['max'], step=1, key=f"roi_max_{i}")
                        if roi_min >= roi_max:
                            st.error("⚠️ ROI最小はROI最大より小さくする必要があります")
                        else:
                            st.session_state.peak_rois[i]['min'] = int(roi_min)
                            st.session_state.peak_rois[i]['max'] = int(roi_max)
                        st.caption(f"ROI範囲: {roi_max - roi_min} ピクセル")

                    with col_roi_plot:
                        roi = st.session_state.peak_rois[i]
                        roi_mask = (pixel_index >= roi['min']) & (pixel_index <= roi['max'])
                        fig_roi = go.Figure()
                        fig_roi.add_trace(go.Scatter(x=pixel_index[roi_mask], y=spectrum_data[roi_mask],
                                                     mode='lines', name='ROIスペクトル', line=dict(color='blue', width=2)))
                        color_cycle = ['red','green','orange','purple','brown','pink','gray','cyan','magenta']
                        color = color_cycle[i % len(color_cycle)]
                        peak_pixel = st.session_state.matched_pixels[i]
                        spectrum_intensity = np.interp(peak_pixel, pixel_index, spectrum_data)
                        fig_roi.add_trace(go.Scatter(x=[peak_pixel], y=[spectrum_intensity], mode='markers+text',
                                                     name=f'ピーク {i+1}',
                                                     marker=dict(color=color, size=15, line=dict(width=2, color='white')),
                                                     text=[f'P{i+1}'], textposition="top center",
                                                     textfont=dict(size=14, color='white')))
                        fig_roi.add_vline(x=peak_pixel, line_dash="dash", line_color=color, line_width=3)
                        fig_roi.update_layout(title=f"ピーク {i+1} のROI表示（ピクセル {roi['min']}-{roi['max']}）",
                                              xaxis_title="ピクセル位置", yaxis_title="強度", height=400,
                                              xaxis=dict(range=[roi['min'], roi['max']]))
                        st.plotly_chart(fig_roi, use_container_width=True)

            st.divider()
            # 対応結果テーブル
            st.subheader("ピーク対応結果")
            peak_df = pd.DataFrame({
                'ピーク': [f"ピーク {i+1}" for i in range(len(st.session_state.matched_pixels))],
                'ピクセル位置': [f"{p:.1f}" for p in st.session_state.matched_pixels],
                '波数 (cm⁻¹)': st.session_state.matched_wavenumbers,
            })
            st.dataframe(peak_df, use_container_width=True)

            st.divider()
            # ==========================================
            # 📊 キャリブレーション結果（5次 + 摂動 + L2最小）
            # ==========================================
            st.header("キャリブレーション結果")

            col_cfg1, col_cfg2, col_cfg3, col_cfg4 = st.columns(4)
            with col_cfg1:
                tol_cm1 = st.number_input("許容最大誤差 (cm⁻¹)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
            with col_cfg2:
                max_jitter_px = st.number_input("最大ピクセル摂動 (±px)", min_value=0, max_value=10, value=2, step=1)
            with col_cfg3:
                max_iter = st.number_input("最大反復回数", min_value=1, max_value=200, value=25, step=1)
            with col_cfg4:
                degree = st.number_input("多項式次数", min_value=1, max_value=5, value=5, step=1)

            # 5次（またはデータ数に応じて下げる）+摂動 で SSE(cm^-1) を最小化
            coeffs_poly, metrics, calc_cm1 = fit_poly_with_jitter_cm1(
                processor,
                st.session_state.matched_pixels,
                st.session_state.matched_wavenumbers,
                degree=int(degree),
                tol_cm1=float(tol_cm1),
                max_jitter_px=int(max_jitter_px),
                max_iter=int(max_iter),
                subpixel=False  # 必要なら True
            )
            MAX_DEGREE = 5
            coeffs_poly_padded = coeffs_poly
            if len(coeffs_poly) < (MAX_DEGREE + 1):  # np.polyfitは高次→低次の順
                pad = np.zeros((MAX_DEGREE + 1) - len(coeffs_poly))
                # 高次側（先頭）に0を付与して5次化
                coeffs_poly_padded = np.concatenate([pad, coeffs_poly])
            # メトリクス表示
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1: st.metric("収束", "✅" if metrics["converged"] else "❌")
            with m2: st.metric("反復", f"{metrics['iterations']}")
            with m3: st.metric("RMSE (cm⁻¹)", f"{metrics['rmse']:.3f}")
            with m4: st.metric("最大|誤差| (cm⁻¹)", f"{metrics['max_abs']:.3f}")
            with m5: st.metric("SSE (二乗和)", f"{metrics['sse']:.3f}")

            # 指定 vs 計算（cm^-1）
            target_cm1 = np.array(st.session_state.matched_wavenumbers, dtype=float)
            err = calc_cm1 - target_cm1
            abs_err = np.abs(err)
            sq_err = err**2

            # 誤差プロット（折れ線）
            st.subheader("誤差プロット（cm⁻¹）")
            fig_err = go.Figure()
            fig_err.add_trace(go.Scatter(
                x=list(range(1, len(target_cm1)+1)),
                y=err,
                mode="lines+markers",
                name="差(計算-指定)"
            ))
            fig_err.add_hline(y=float(tol_cm1), line_dash="dash", line_color="red", opacity=0.6)
            fig_err.add_hline(y=-float(tol_cm1), line_dash="dash", line_color="red", opacity=0.6)
            fig_err.update_layout(
                xaxis_title="ピーク番号",
                yaxis_title="誤差 (cm⁻¹)",
                height=360,
                showlegend=False
            )
            st.plotly_chart(fig_err, use_container_width=True)

            st.subheader("多項式近似（pixel→wavenumber）カーブ")
            poly = coeffs_poly_padded if 'coeffs_poly_padded' in locals() else coeffs_poly
            x_fit = np.linspace(min(metrics["pixels"]), max(metrics["pixels"]), 1000)
            y_fit_wl = np.polyval(poly, x_fit)
            y_fit_cm1 = _wl_to_cm1_float(processor.laser_wavelength, y_fit_wl)
            y_pts_wl = np.polyval(poly, metrics["pixels"])
            y_pts_cm1 = _wl_to_cm1_float(processor.laser_wavelength, y_pts_wl)
            fig_fit = go.Figure()
            fig_fit.add_trace(go.Scatter(x=metrics["pixels"], y=y_pts_cm1,
                                        mode="markers", name="採用ピーク(波数)"))
            fig_fit.add_trace(go.Scatter(x=x_fit, y=y_fit_cm1, mode="lines", name=f"{degree}次近似"))
            fig_fit.update_layout(xaxis_title="pixel", yaxis_title="wavenumber (cm⁻¹)", height=380)
            st.plotly_chart(fig_fit, use_container_width=True)

            # 参考：pixel→wavelength カーブ
            with st.expander("多項式近似（pixel→wavelength）カーブを表示", expanded=False):
                x_fit = np.linspace(min(metrics["pixels"]), max(metrics["pixels"]), 1000)
                y_fit = np.polyval(coeffs_poly_padded, x_fit)
                fig_fit = go.Figure()
                fig_fit.add_trace(go.Scatter(x=metrics["pixels"], y=np.polyval(coeffs_poly_padded, metrics["pixels"]),
                                            mode="markers", name="採用ピーク(波長)"))
                fig_fit.add_trace(go.Scatter(x=x_fit, y=y_fit, mode="lines", name=f"{degree}次近似"))
                fig_fit.update_layout(xaxis_title="pixel", yaxis_title="wavelength (nm)", height=380)
                st.plotly_chart(fig_fit, use_container_width=True)

            # エクスポート
            st.subheader("結果のエクスポート")
            formatted_coeffs = [f'"{coeff:.7E}"' for coeff in coeffs_poly_padded[::-1]]
            server_format = f"'b_coeff': [{', '.join(formatted_coeffs)}],\n'laser_wavelength': {processor.laser_wavelength},\n'degree': {int(degree)}"
            st.code(server_format, language='python')

            lines = []
            lines.append(f"Laser wavelength: {processor.laser_wavelength} nm")
            lines.append(f"Degree: {int(degree)}")
            lines.append(f"Converged: {metrics['converged']}")
            lines.append(f"Iterations: {metrics['iterations']}")
            lines.append(f"RMSE(cm^-1): {metrics['rmse']:.6f}")
            lines.append(f"MaxAbs(cm^-1): {metrics['max_abs']:.6f}")
            lines.append(f"SSE(cm^-1^2): {metrics['sse']:.6f}")
            lines.append("Peaks:")
            for i, (px, tgt, est, e, ae, se) in enumerate(zip(metrics["pixels"], target_cm1, calc_cm1, err, abs_err, sq_err), 1):
                lines.append(f"  Peak {i}: pixel={px:.2f}, target={tgt:.3f}, calc={est:.3f}, diff={e:.3f}, |diff|={ae:.3f}, diff^2={se:.3f}")
            lines.append("\nCoeffs (B_0..B_n on wavelength):")
            for i, c in enumerate(coeffs_poly_padded[::-1]):
                lines.append(f"B_{i} = {c:.10e}")
            st.download_button(
                "📥 フィット結果をダウンロード",
                data="\n".join(lines),
                file_name=f"poly{int(degree)}_jitter_fit_{processor.laser_wavelength}nm.txt",
                mime="text/plain"
            )

else:
    st.info("👆 キャリブレーションを開始するにはCSVファイルをアップロードしてください")
