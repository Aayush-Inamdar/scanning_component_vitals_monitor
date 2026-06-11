import time
import queue
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import signal
from scipy.signal import hilbert

from config.settings import BUFFER_DURATION_SEC, TARGET_FPS
from services.signal_processor import PosSignalProcessor
from services.pos_engine import extract_pos_signal
from services.metrics.hrv import get_rmssd_from_array


def run_pos_tracker(ipc):
    print("[TRACKER] Process initializing. Waiting for MediaPipe ROI lock...")
    ipc.roi_locked_event.wait()
    print("[TRACKER] ROI Lock detected. Starting continuous POS processing...")

    processor = PosSignalProcessor()

    # 1. Rolling Buffers (For Live UI)
    times, r_buf, g_buf, b_buf = [], [], [], []

    # 2. FULL Session Buffers (For Final Graphs)
    f_t, f_r, f_g, f_b, f_bcg = [], [], [], [], []
    last_calc_time = 0

    # LOOP UNTIL SHUTDOWN EVENT IS TRIGGERED
    while not ipc.shutdown_event.is_set():
        try:
            data = ipc.rgb_queue.get(timeout=0.5)
            if data is None:
                break
            t_val, r_val, g_val, b_val, nose_y = data

            # Save to full history for graphs
            f_t.append(t_val); f_r.append(r_val); f_g.append(g_val); f_b.append(b_val)
            f_bcg.append(nose_y if nose_y is not None else (f_bcg[-1] if f_bcg else 0))

            # Save to rolling buffer for live metrics
            times.append(t_val); r_buf.append(r_val); g_buf.append(g_val); b_buf.append(b_val)

            while len(times) > 0 and times[0] < t_val - BUFFER_DURATION_SEC:
                times.pop(0); r_buf.pop(0); g_buf.pop(0); b_buf.pop(0)

            # Inject ML anchor into processor the moment it arrives
            current_anchor = ipc.anchor_hr_value.value
            if current_anchor > 0.0 and processor.anchor_hr != current_anchor:
                processor.anchor_hr = current_anchor
                print(f"[TRACKER] ML Anchor injected: {current_anchor:.1f} BPM")

            if t_val - last_calc_time >= 1.0:
                if len(times) > 30:
                    results = processor.process(
                        np.array(times), np.array(r_buf), np.array(g_buf), np.array(b_buf)
                    )
                    if any(v is not None for v in results.values()):
                        try:
                            ipc.results_queue.put_nowait(results)
                        except queue.Full:
                            pass
                last_calc_time = t_val

        except queue.Empty:
            pass
        except Exception as e:
            print(f"[TRACKER] Math Engine Error: {e}")

    # ==========================================
    # SHUTDOWN TRIGGERED: RENDER FINAL GRAPHS
    # ==========================================
    if len(f_t) < int(TARGET_FPS * 5):
        print("[TRACKER] Not enough data to generate graphs. Shutting down.")
        print("[TRACKER] Process shutting down cleanly.")
        return

    print("\n" + "="*50)
    print("          GENERATING FINAL SESSION GRAPHS          ")
    print("="*50)

    # ------------------------------------------------------------------
    # SYNTHETIC CFR RECONSTRUCTION (Interpolation fixes time-dilation)
    # ------------------------------------------------------------------
    raw_t   = np.array(f_t)
    raw_r   = np.array(f_r)
    raw_g   = np.array(f_g)
    raw_b   = np.array(f_b)
    raw_bcg = np.array(f_bcg)

    total_time        = raw_t[-1] - raw_t[0]
    ideal_frame_count = int(total_time * TARGET_FPS)
    t_arr  = np.linspace(raw_t[0], raw_t[-1], ideal_frame_count)

    r_arr   = np.interp(t_arr, raw_t, raw_r)
    g_arr   = np.interp(t_arr, raw_t, raw_g)
    b_arr   = np.interp(t_arr, raw_t, raw_b)
    bcg_arr = np.interp(t_arr, raw_t, raw_bcg)
    t_axis  = np.linspace(0, total_time, len(t_arr))

    os.makedirs("data", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Pull SQI and anchor from IPC for graph titles
    final_sqi    = ipc.ml_sqi_value.value
    final_anchor = ipc.anchor_hr_value.value
    ml_succeeded = final_anchor > 0.0

    sqi_label = f"ML SQI: {final_sqi:.2f} | Anchor: {final_anchor:.1f} BPM" if ml_succeeded else f"ML SQI: {final_sqi:.2f} | Fallback Mode"

    # ------------------------------------------------------------------
    # GRAPH 1: RAW RGB TIME SERIES
    # ------------------------------------------------------------------
    plt.style.use('default')
    fig_rgb, ax_rgb = plt.subplots(figsize=(10, 4))
    fig_rgb.patch.set_facecolor('white')
    ax_rgb.set_facecolor('white')

    ax_rgb.plot(t_axis, r_arr, color='#ef4444', linewidth=0.8, label='Red Channel',   alpha=0.9)
    ax_rgb.plot(t_axis, g_arr, color='#22c55e', linewidth=0.8, label='Green Channel', alpha=0.9)
    ax_rgb.plot(t_axis, b_arr, color='#3b82f6', linewidth=0.8, label='Blue Channel',  alpha=0.9)

    ax_rgb.set_title(f"Raw RGB Intensity Signals | Duration: {total_time:.1f}s | {sqi_label}",
                     color='black', fontsize=12, fontweight='bold', pad=10)
    ax_rgb.set_xlabel("Time (s)", color='black', fontsize=10)
    ax_rgb.set_ylabel("Mean Pixel Intensity", color='black', fontsize=10)
    ax_rgb.spines['top'].set_visible(False)
    ax_rgb.spines['right'].set_visible(False)
    ax_rgb.tick_params(colors='black')
    legend = ax_rgb.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='black')
    for text in legend.get_texts(): text.set_color("black")

    rgb_save_path = f"data/desktop_raw_rgb_signal_{timestamp}.png"
    plt.savefig(rgb_save_path, bbox_inches='tight', facecolor='white', dpi=300)
    plt.close(fig_rgb)
    print(f"  -> Saved {rgb_save_path}")

    # ------------------------------------------------------------------
    # GRAPH 2: BVP — Full Shen DSP Pipeline
    # ------------------------------------------------------------------
    pos_signal  = extract_pos_signal(t_arr, r_arr, g_arr, b_arr, TARGET_FPS)
    clean_pulse = signal.detrend(pos_signal)

    # Digital Iron: remove high-frequency camera noise
    clean_pulse = signal.savgol_filter(clean_pulse, window_length=7, polyorder=3)

    # 4th-order narrow bandpass for clean time-domain morphology
    b_filt, a_filt = signal.butter(6, [0.8, 1.6], btype='bandpass', fs=TARGET_FPS)
    clean_pulse = signal.filtfilt(b_filt, a_filt, clean_pulse)

    # Hilbert Envelope Normalization
    envelope        = np.abs(hilbert(clean_pulse))
    env_win         = max(5, int(TARGET_FPS * 1.5) | 1)
    envelope_smooth = signal.savgol_filter(envelope, env_win, 3)

    warmup = int(TARGET_FPS * 1.5)
    if len(envelope_smooth) > warmup * 2:
        stable_med = np.median(envelope_smooth[warmup:-warmup])
        envelope_smooth[:warmup]  = stable_med
        envelope_smooth[-warmup:] = stable_med

    clean_pulse = clean_pulse / (envelope_smooth + 1e-8)
    clean_pulse = (clean_pulse - np.mean(clean_pulse)) / (np.std(clean_pulse) + 1e-8)

    bvp_t_axis = np.linspace(0, total_time, len(clean_pulse))

    fig_bvp, ax_bvp = plt.subplots(figsize=(10, 4))
    fig_bvp.patch.set_facecolor('white')
    ax_bvp.set_facecolor('white')
    ax_bvp.plot(bvp_t_axis, clean_pulse, color='#e8633a', linewidth=0.8)
    ax_bvp.set_ylim(-3, 3)
    ax_bvp.set_title(f"Oscillatory rPPG Trace (Optical) | Duration: {total_time:.1f}s | {sqi_label}",
                     color='black', fontsize=12, fontweight='bold', pad=10)
    ax_bvp.set_xlabel("Time (s)", color='black', fontsize=10)
    ax_bvp.set_ylabel("Amplitude (Uniform Z-Score)", color='black', fontsize=10)
    ax_bvp.spines['top'].set_visible(False)
    ax_bvp.spines['right'].set_visible(False)
    ax_bvp.tick_params(colors='black')

    bvp_save_path = f"data/desktop_bvp_signal_{timestamp}.png"
    plt.savefig(bvp_save_path, bbox_inches='tight', facecolor='white', dpi=300)
    plt.close(fig_bvp)
    print(f"  -> Saved {bvp_save_path}")

    # ------------------------------------------------------------------
    # GRAPH 3: BCG — Ballistocardiogram
    # ------------------------------------------------------------------
    clean_bcg = signal.detrend(bcg_arr)

    if len(clean_bcg) > 11:
        clean_bcg = signal.savgol_filter(clean_bcg, window_length=11, polyorder=3)

    b_filt_bcg, a_filt_bcg = signal.butter(6, [0.8, 8.0], btype='bandpass', fs=TARGET_FPS)
    clean_bcg = signal.filtfilt(b_filt_bcg, a_filt_bcg, clean_bcg)

    envelope_bcg        = np.abs(hilbert(clean_bcg))
    envelope_smooth_bcg = signal.savgol_filter(envelope_bcg,
                                               window_length=max(5, int(TARGET_FPS * 2) | 1),
                                               polyorder=3)

    if len(envelope_smooth_bcg) > warmup * 2:
        stable_median_bcg = np.median(envelope_smooth_bcg[warmup:-warmup])
        envelope_smooth_bcg[:warmup]  = stable_median_bcg
        envelope_smooth_bcg[-warmup:] = stable_median_bcg

    clean_bcg = clean_bcg / (envelope_smooth_bcg + 1e-8)

    energy_window = int(TARGET_FPS * 1.5)
    local_energy  = np.convolve(np.abs(clean_bcg), np.ones(energy_window) / energy_window, mode='same')

    median_energy    = np.median(local_energy)
    mad_energy       = np.median(np.abs(local_energy - median_energy))
    threshold_energy = median_energy + (4.0 * mad_energy)
    attenuation      = np.where(local_energy > threshold_energy,
                                threshold_energy / (local_energy + 1e-8), 1.0)

    mask_window  = int(TARGET_FPS * 1.5)
    mask_window  = mask_window if mask_window % 2 != 0 else mask_window + 1
    smooth_mask  = signal.savgol_filter(attenuation, max(5, mask_window), 3)
    clean_bcg    = clean_bcg * smooth_mask

    median_bcg     = np.median(clean_bcg)
    mad_bcg        = np.median(np.abs(clean_bcg - median_bcg))
    robust_std_bcg = 1.4826 * mad_bcg + 1e-8
    clean_bcg      = (clean_bcg - median_bcg) / robust_std_bcg

    fig_bcg, ax_bcg = plt.subplots(figsize=(10, 4))
    fig_bcg.patch.set_facecolor('white')
    ax_bcg.set_facecolor('white')
    ax_bcg.plot(t_axis, clean_bcg, color='#9333ea', linewidth=0.8)

    y_max_bcg = np.max(np.abs(clean_bcg)) * 1.15
    ax_bcg.set_ylim(-y_max_bcg, y_max_bcg)
    ax_bcg.set_title(f"Ballistocardiogram Trace (Mechanical) | Duration: {total_time:.1f}s",
                     color='black', fontsize=12, fontweight='bold', pad=10)
    ax_bcg.set_xlabel("Time (s)", color='black', fontsize=10)
    ax_bcg.set_ylabel("Micro-Displacement (Robust Z-Score)", color='black', fontsize=10)
    ax_bcg.spines['top'].set_visible(False)
    ax_bcg.spines['right'].set_visible(False)
    ax_bcg.tick_params(colors='black')

    bcg_save_path = f"data/desktop_bcg_signal_{timestamp}.png"
    plt.savefig(bcg_save_path, bbox_inches='tight', facecolor='white', dpi=300)
    plt.close(fig_bcg)
    print(f"  -> Saved {bcg_save_path}")

    # ------------------------------------------------------------------
    # GRAPH 4: TACHOGRAM — Vault-Gated Peak Detection
    # ------------------------------------------------------------------
    skip_frames      = int(TARGET_FPS * 5.0)
    peaks            = np.array([])
    raw_rr_intervals = np.array([])
    median_rr        = 0
    final_graph_hr   = None

    if len(clean_pulse) > skip_frames:
        search_pulse = clean_pulse[skip_frames:]

        # Vault HR: prefer ML anchor, fall back to math anchor, fall back to last known
        vault_hr = None
        if final_anchor > 0.0:
            vault_hr = final_anchor
        elif processor.anchor_hr is not None:
            vault_hr = processor.anchor_hr
        elif processor.last_known_hr is not None:
            vault_hr = processor.last_known_hr

        if vault_hr is not None:
            expected_rr_frames = int((60.0 / vault_hr) * TARGET_FPS)
            calc_distance = max(int(expected_rr_frames * 0.70), int(TARGET_FPS * (60.0 / 150.0)))
        else:
            calc_distance = int(TARGET_FPS * (60.0 / 120.0))

        calc_prominence = 0.15
        calc_wlen       = int(TARGET_FPS * 3.0)

        peaks_search, _ = signal.find_peaks(
            search_pulse,
            distance=calc_distance,
            prominence=calc_prominence,
            wlen=calc_wlen
        )
        peaks = peaks_search + skip_frames

        if len(peaks) >= 3:
            duration_sec     = len(search_pulse) / TARGET_FPS
            hr_count         = (len(peaks_search) / duration_sec) * 60.0 if duration_sec > 0 else 0
            raw_rr_intervals = np.diff(peaks) * (1000.0 / TARGET_FPS)
            median_rr        = np.median(raw_rr_intervals)

            if not np.isnan(median_rr) and median_rr > 0:
                hr_median = 60000.0 / median_rr
                if abs(hr_count - hr_median) <= 5.0:
                    final_graph_hr = (hr_count + hr_median) / 2.0
                else:
                    if vault_hr is not None:
                        final_graph_hr = hr_count if abs(hr_count - vault_hr) < abs(hr_median - vault_hr) else hr_median
                    else:
                        final_graph_hr = hr_median

                # Absolute Vault Clamp
                if vault_hr is not None and final_graph_hr is not None:
                    if abs(final_graph_hr - vault_hr) > 8.0:
                        final_graph_hr = vault_hr
                        print(f"[TRACKER] Post-scan HR overridden by Vault Anchor: {final_graph_hr:.1f} BPM")

                if final_graph_hr is not None and not (40.0 <= final_graph_hr <= 150.0):
                    final_graph_hr = None
        else:
            raw_rr_intervals = np.array([])
            median_rr        = 0

    clean_rr_intervals = np.array([])
    if len(peaks) > 2 and len(raw_rr_intervals) > 0 and median_rr > 0:
        valid_mask         = (raw_rr_intervals >= median_rr * 0.75) & (raw_rr_intervals <= median_rr * 1.25)
        clean_rr_intervals = raw_rr_intervals[valid_mask]

        if len(clean_rr_intervals) > 0:
            beat_numbers = np.arange(1, len(clean_rr_intervals) + 1)
            mean_rr      = np.mean(clean_rr_intervals)

            hr_title = f"{final_graph_hr:.1f} BPM" if final_graph_hr else "N/A"

            fig_tacho, ax_tacho = plt.subplots(figsize=(10, 4))
            fig_tacho.patch.set_facecolor('white')
            ax_tacho.set_facecolor('white')
            ax_tacho.plot(beat_numbers, clean_rr_intervals,
                          color='#14b8a6', linewidth=1.0, marker='o', markersize=3)

            ax_tacho.set_title(f"Heartbeat Intervals Tachogram | Mean RR: {mean_rr:.0f} ms | Final HR: {hr_title}",
                               color='black', fontsize=12, fontweight='bold', pad=10)
            ax_tacho.set_xlabel("Beat Number", color='black', fontsize=10)
            ax_tacho.set_ylabel("Interval Duration (ms)", color='black', fontsize=10)
            ax_tacho.set_ylim(max(400, min(clean_rr_intervals) - 50), min(1500, max(clean_rr_intervals) + 50))
            ax_tacho.spines['top'].set_visible(False)
            ax_tacho.spines['right'].set_visible(False)
            ax_tacho.tick_params(colors='black')
            ax_tacho.grid(True, linestyle='--', alpha=0.3)

            tacho_save_path = f"data/desktop_tachogram_{timestamp}.png"
            plt.savefig(tacho_save_path, bbox_inches='tight', facecolor='white', dpi=300)
            plt.close(fig_tacho)
            print(f"  -> Saved {tacho_save_path}")

    # ------------------------------------------------------------------
    # TERMINAL SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "="*50)
    print("          FINAL POST-SCAN DERIVED METRICS          ")
    print("="*50)
    print(f"  ML Signal Quality (SQI) : {final_sqi:.3f}  ({'LOCKED' if ml_succeeded else 'FALLBACK'})")

    if final_graph_hr:
        print(f"  Calculated Heart Rate   : {final_graph_hr:.1f} BPM")
    else:
        print(f"  Calculated Heart Rate   : INSIGNIFICANT DATA")

    if len(peaks) > 2 and len(clean_rr_intervals) > 1:
        hrv_rmssd = get_rmssd_from_array(clean_rr_intervals)
        print(f"  Calculated HRV (RMSSD)  : {hrv_rmssd:.1f} ms")
    else:
        print(f"  Calculated HRV (RMSSD)  : INSIGNIFICANT DATA")

    try:
        br_peaks, _ = signal.find_peaks(envelope_smooth_bcg,
                                        distance=int(TARGET_FPS * (60.0 / 30.0)))
        if len(br_peaks) >= 2:
            br_intervals = np.diff(br_peaks) / TARGET_FPS
            calc_br      = 60.0 / np.median(br_intervals)
            print(f"  Calculated Breathing    : {calc_br:.1f} Breaths/min")
        else:
            print(f"  Calculated Breathing    : INSIGNIFICANT DATA")
    except Exception:
        print(f"  Calculated Breathing    : N/A")

    print("="*50 + "\n")
    print("[TRACKER] Process shutting down cleanly.")