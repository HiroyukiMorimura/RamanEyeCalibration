import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter, find_peaks, peak_prominences
from csv_processor import StreamlitRamanSpectrumProcessor

st.set_page_config(
    page_title="インタラクティブ・ラマン分光キャリブレーション",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 インタラクティブなラマンピーク調整")
st.markdown("ピーク位置と波数を調整して分光器をキャリブレーションします")

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


# Initialize processor
processor = StreamlitRamanSpectrumProcessor()

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ 設定")
    laser_wavelength = st.number_input("レーザー波長 (nm)", value=532, min_value=200, max_value=2000, step=1)
    processor.laser_wavelength = laser_wavelength
    
    st.divider()
    st.header("📁 データ入力")
    uploaded_file = st.file_uploader("CSVファイルをアップロード", type=['csv'], help="ラマンスペクトルのCSVをアップロードしてください")

    # -------------------------------
    # 自動ピーク検出（サイドバー）
    # -------------------------------
    if uploaded_file is not None:
        st.header("自動ピーク検出")
        smooth_win = st.number_input("平滑化ウィンドウ(奇数)", min_value=1, max_value=501, value=25, step=2, key="auto_smooth_win")
        deriv_thresh = st.number_input("2次微分閾値(高さ)", min_value=0, max_value=1000, value=5, step=1, key="auto_deriv_thresh")
        prom_thresh  = st.number_input("卓立度閾値", min_value=0, max_value=1000, value=5, step=1, key="auto_prom_thresh")
        min_distance = st.number_input("最小ピーク間隔(ピクセル)", min_value=1, max_value=1000, value=10, step=1, key="auto_min_distance")

        # 反映する個数（prominence降順の上位）
        default_k = 10  # 既定ピークリストの長さに合わせて10を推奨
        top_k = st.number_input("反映するピーク数(上位prominence順)", min_value=1, max_value=200, value=int(default_k), step=1, key="auto_top_k")

        st.button("🔍 検出を実行", key="btn_run_detect_sidebar")


# Main content area
if uploaded_file is not None:
    pixel_index, spectrum_data = processor.read_csv_data(uploaded_file)
    
    if pixel_index is not None and spectrum_data is not None:
        st.success(f"✅ データを読み込みました！ {len(pixel_index)} 点")

        # ファイルが切り替わったら検出/手動/除外/適用状態を初期化
        if st.session_state.get("last_uploaded_name") != uploaded_file.name:
            st.session_state["manual_peaks_idx"] = []        # 手動追加（インデックス）
            st.session_state["excluded_auto_peaks"] = set()  # 自動検出からの除外（インデックス）
            for k in ["auto_peaks_idx", "auto_peaks_prom", "auto_d2",
                      "peaks_applied", "matched_pixels", "matched_wavenumbers", "peak_rois"]:
                st.session_state.pop(k, None)
            st.session_state["last_uploaded_name"] = uploaded_file.name

        # サイドバーのボタンで検出
        if st.session_state.get("btn_run_detect_sidebar"):
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
                # prominence 降順で並べ替え → 上位K抽出 → x昇順
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

                # ============================
                # レイアウト：左（広）／右（狭）
                # ============================
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

                    # ↓↓↓ グラフ直下：クリックで展開するプレビュー ↓↓↓
                    with st.expander("✅ 現在の反映対象（プレビュー）を表示", expanded=False):
                        rows = []
                        for idx in valid_auto_idx:
                            rows.append({"種別": "自動（有効）", "ピクセル": int(pixel_index[idx]), "強度(a.u.)": float(spectrum_data[idx])})
                        for idx in manual_idx:
                            rows.append({"種別": "手動追加", "ピクセル": int(pixel_index[idx]), "強度(a.u.)": float(spectrum_data[idx])})
                        if rows:
                            st.dataframe(pd.DataFrame(rows).sort_values("ピクセル"), use_container_width=True)
                        else:
                            st.info("反映対象がまだありません。手動追加するか、除外を解除してください。")

                # ---- 右：コンパクトなコントロール ----
                with col_ctrl:
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

                # ---- 適用ボタン（押すまで以降のセクションは表示しない） ----
                st.write("")
                if st.button("✅ 上記の反映対象で matched_pixels を更新", use_container_width=True):
                    valid_auto_idx = np.array([i for i in sel_idx if i not in excluded], dtype=int)
                    manual_idx = np.array(st.session_state.get("manual_peaks_idx", []), dtype=int)
                    combined_idx = np.unique(np.concatenate([valid_auto_idx, manual_idx])).astype(int)
                    combined_idx.sort()
                    if combined_idx.size == 0:
                        st.warning("反映対象が空です。")
                    else:
                        # ピクセル位置を確定
                        st.session_state.matched_pixels = pixel_index[combined_idx].astype(int).tolist()

                        # ★ 波数（cm⁻¹）は既定エタノールピークを「ピーク1から順」にセット
                        default_peaks = processor.default_ethanol_peaks[:]  # 長さ10: [2973,2927,2876,1455,1277,1097,1063,880,434,0]
                        n = len(st.session_state.matched_pixels)
                        if n <= len(default_peaks):
                            st.session_state.matched_wavenumbers = [float(v) for v in default_peaks[:n]]
                        else:
                            # 既定を超える場合は末尾の値(0)でパディング
                            pad = [float(default_peaks[-1])] * (n - len(default_peaks))
                            st.session_state.matched_wavenumbers = [float(v) for v in default_peaks] + pad

                        # ROI を現在の matched_pixels に基づき再構築（±100px）
                        st.session_state.peak_rois = {}
                        roi_size = 100
                        for i, px in enumerate(st.session_state.matched_pixels):
                            st.session_state.peak_rois[i] = {
                                "min": max(int(px - roi_size), int(min(pixel_index))),
                                "max": min(int(px + roi_size), int(max(pixel_index))),
                            }

                        # 適用フラグ
                        st.session_state.peaks_applied = True

                        st.success(f"{len(st.session_state.matched_pixels)} 個のピークを反映しました。下に全体表示とROI調整が現れます。")
                        st.rerun()

        # --------------------------------------------------
        # ここから下は「適用」後にのみ表示
        # --------------------------------------------------
        if st.session_state.get("peaks_applied", False) and st.session_state.get("matched_pixels"):
            # 全体スペクトル（適用済みのピークを表示）
            st.header("📊 ピーク付き全体スペクトル")
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

            # 各ピークのROI付き調整（ピクセル位置は適用結果で初期化済み）
            st.subheader("🔍 各ピークのROI付き調整")
            tab_labels = [f"ピーク {i+1}" for i in range(len(st.session_state.matched_pixels))]
            tabs = st.tabs(tab_labels)
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
                            st.session_state.matched_pixels[i] = new_pixel
                            roi_size = 100
                            new_roi_min = max(int(new_pixel - roi_size), int(min(pixel_index)))
                            new_roi_max = min(int(new_pixel + roi_size), int(max(pixel_index)))
                            st.session_state.peak_rois[i]['min'] = new_roi_min
                            st.session_state.peak_rois[i]['max'] = new_roi_max

                    # ---- 波数（cm⁻¹）：既定エタノールピークが初期値として入る ----
                    with col_wavenumber:
                        st.markdown("**波数:**")
                        # ここで st.session_state.matched_wavenumbers[i] は上の適用時に
                        # processor.default_ethanol_peaks の順番で設定済み
                        new_wavenumber = st.number_input(
                            "波数 (cm⁻¹)",
                            value=float(st.session_state.matched_wavenumbers[i]),
                            step=0.1,
                            format="%.1f",
                            key=f"wavenumber_input_{i}"
                        )
                        st.session_state.matched_wavenumbers[i] = new_wavenumber
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
                            st.session_state.peak_rois[i]['min'] = roi_min
                            st.session_state.peak_rois[i]['max'] = roi_max
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
                '波長 (nm)': [f"{processor.wavenumber_to_wavelength(wn):.2f}" for wn in st.session_state.matched_wavenumbers],
                '波数 (cm⁻¹)': st.session_state.matched_wavenumbers,
            })
            st.dataframe(peak_df, use_container_width=True)

            st.divider()
            # キャリブレーション結果
            if len(st.session_state.matched_pixels) >= 2:
                st.header("📊 キャリブレーション結果")
                matched_wavelengths = [processor.wavenumber_to_wavelength(wn) for wn in st.session_state.matched_wavenumbers]
                coeffs, degree, fitting_results = processor.polynomial_fitting(st.session_state.matched_pixels, matched_wavelengths)
                if coeffs is not None:
                    col1, col2, col3 = st.columns(3)
                    with col1: st.metric("レーザー波長", f"{laser_wavelength} nm")
                    with col2: st.metric("ピーク数", len(st.session_state.matched_pixels))
                    with col3: st.metric("多項式の次数", degree)

                    coeff_col1, coeff_col2 = st.columns(2)
                    with coeff_col1:
                        st.subheader("多項式近似")
                        fig_fit = go.Figure()
                        fig_fit.add_trace(go.Scatter(x=st.session_state.matched_pixels, y=matched_wavelengths,
                                                     mode='markers', name='ピーク位置', marker=dict(color='red', size=10)))
                        x_fit = np.linspace(min(st.session_state.matched_pixels), max(st.session_state.matched_pixels), 1000)
                        y_fit = np.polyval(coeffs, x_fit)
                        fig_fit.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines',
                                                     name=f'{degree} 次近似', line=dict(color='blue', width=2)))
                        fig_fit.update_layout(title=f"多項式近似（次数 {degree}）",
                                              xaxis_title="ピクセル位置", yaxis_title="波長 (nm)", height=400, showlegend=True)
                        st.plotly_chart(fig_fit, use_container_width=True)
                    with coeff_col2:
                        st.subheader("多項式係数")
                        coeffs_df = pd.DataFrame([{'係数': f'B_{i}','値': f'{c:.6e}','説明': f'x^{i}' if i>0 else '定数項'}
                                                  for i, c in enumerate(coeffs[::-1])])
                        st.write(coeffs_df)
                        st.write("**多項式の式:**")
                        equation = " + ".join([f"{c:.3e}" if i==0 else (f"{c:.3e}x" if i==1 else f"{c:.3e}x^{i}")
                                               for i, c in enumerate(coeffs[::-1])])
                        st.code(f"y = {equation}", language='python')

                    st.subheader("多項式近似によるピクセル対波数")
                    x_values = processor.pixel_indexs
                    y_values = np.polyval(coeffs, x_values)
                    fig_pixel_vs_wn = go.Figure()
                    fig_pixel_vs_wn.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines',
                                                         name='ピクセル対波数', line=dict(color='purple', width=2)))
                    fig_pixel_vs_wn.update_layout(title="ピクセル位置 vs 波数",
                                                  xaxis_title="ピクセル位置", yaxis_title="波数 (cm⁻¹)", height=400, showlegend=True)
                    st.plotly_chart(fig_pixel_vs_wn, use_container_width=True)

                    st.subheader("🚀 結果のエクスポート")
                    formatted_coeffs = [f'"{coeff:.7E}"' for coeff in coeffs[::-1]]
                    server_format = f"'b_coeff': [{', '.join(formatted_coeffs)}],\n'laser_wavelength': {laser_wavelength},\n'degree': {degree}"
                    st.code(server_format, language='python')
                    results_text = f"Laser wavelength: {laser_wavelength} nm\nPolynomial degree: {degree}\nNumber of peaks: {len(st.session_state.matched_pixels)}\n\nPeak matching results:\n"
                    for i, (pixel, wavenumber) in enumerate(zip(st.session_state.matched_pixels, st.session_state.matched_wavenumbers)):
                        wavelength = processor.wavenumber_to_wavelength(wavenumber)
                        results_text += f"Peak {i+1}: pixel={pixel:.1f}, wavenumber={wavenumber}, wavelength={wavelength:.2f}\n"
                    results_text += "\nPolynomial coefficients:\n"
                    for i, coeff in enumerate(coeffs[::-1]):
                        results_text += f"B_{i} = {coeff:.10e}\n"
                    st.download_button(label="📥 キャリブレーション結果をダウンロード",
                                       data=results_text,
                                       file_name=f"calibration_results_{laser_wavelength}nm.txt",
                                       mime="text/plain")

else:
    st.info("👆 キャリブレーションを開始するにはCSVファイルをアップロードしてください")
