#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later · Copyright (C) 2026 Пантелеев Г.Д.
"""Рисует схемы-графики img/svg/sv_*.svg для тренажёра (вариант 12).
Запуск: python3 tools/draw_src/plots.py  (нужны numpy, scipy, matplotlib)."""
import os
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'img', 'svg')
S1, S2, S3, S4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e3e2de'
plt.rcParams.update({
    'svg.fonttype': 'none', 'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.spines.top': False,
    'axes.spines.right': False, 'lines.linewidth': 2, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'legend.frameon': False, 'axes.titlesize': 12, 'axes.titleweight': 'bold'})

def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name + '.svg'), format='svg', dpi=110, metadata={'Date': None})
    plt.close(fig)

# 1. Наложение частот (ЛР1): fд = 48 кГц
fig, ax = plt.subplots(figsize=(7.6, 3.2))
f = np.linspace(0, 100, 2001); fs = 48
fa = np.abs(f - fs * np.round(f / fs))
ax.plot(f, fa, color=S1, label='кажущаяся частота fₐ = |f − k·fд|')
ax.axhline(24, color=INK2, lw=1, ls='--'); ax.text(99, 25, 'fN = 24 кГц', ha='right', color=INK2, fontsize=10)
pts = [1, 21, 41, 61, 81, 96]
for p in pts:
    a = abs(p - fs * round(p / fs))
    ax.plot(p, a, 'o', ms=8, color=S2, mec='white', mew=2)
    ax.annotate(f'{p}→{a:g}', (p, a), textcoords='offset points', xytext=(0, 9), ha='center', fontsize=10, color=INK)
ax.set_xlabel('частота генератора f, кГц'); ax.set_ylabel('частота в спектре, кГц')
ax.set_xlim(0, 100); ax.set_ylim(-1, 30); ax.legend(loc='upper left')
save(fig, 'sv_alias')

# 2. Окна: АЧХ в дБ (N = 64)
N = 64; fig, ax = plt.subplots(figsize=(7.6, 3.4))
for w, name, col in [(signal.windows.boxcar(N), 'прямоугольное (−13 дБ)', S1), (signal.windows.hann(N), 'Ханна (−31 дБ)', S3),
                     (signal.windows.hamming(N), 'Хэмминга (−43 дБ)', S2), (signal.windows.blackman(N), 'Блэкмана (−58 дБ)', S4)]:
    W = np.fft.fftshift(np.abs(np.fft.fft(w, 4096)))
    W = 20 * np.log10(W / W.max() + 1e-12)
    x = np.linspace(-N / 2, N / 2, 4096, endpoint=False)
    ax.plot(x, W, color=col, lw=1.6, label=name)
ax.set_xlim(0, 12); ax.set_ylim(-100, 3); ax.set_xlabel('частота, в бинах ДПФ (fд/N)'); ax.set_ylabel('дБ')
ax.legend(loc='upper right', fontsize=10)
save(fig, 'sv_windows')

# 3. Однотональная АМ (ЛР5: f0 = 12 Гц, fc = 120 Гц, M = 0,5)
t = np.arange(0, 0.25, 1 / 12000); f0, fc, M = 12, 120, 0.5
u = (1 + M * np.cos(2 * np.pi * f0 * t)) * np.cos(2 * np.pi * fc * t)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.0), gridspec_kw={'width_ratios': [1.5, 1]})
a1.plot(t, u, color=S1, lw=1.2, label='АМ-сигнал')
a1.plot(t, 1 + M * np.cos(2 * np.pi * f0 * t), color=S2, lw=1.6, ls='--', label='огибающая 1 + M·cos')
a1.set_xlabel('t, с'); a1.set_ylim(-1.7, 2.4); a1.set_title('Сигнал, M = 0,5'); a1.legend(loc='upper center', ncol=2, fontsize=9)
for fr, A in [(108, M / 2), (120, 1), (132, M / 2)]:
    a2.vlines(fr, 0, A, color=S1, lw=3); a2.text(fr, A + 0.04, f'{A:g}', ha='center', fontsize=10, color=INK)
a2.set_xlim(90, 150); a2.set_ylim(0, 1.2); a2.set_xticks([108, 120, 132]); a2.set_xlabel('f, Гц'); a2.set_title('Спектр: fc и fc ± f0')
save(fig, 'sv_am')

# 4. Созвездие 8-КАМ (ЛР6)
pts = np.array([complex(i, q) for i in (-3, -1, 1, 3) for q in (-1, 1)])
fig, ax = plt.subplots(figsize=(4.6, 3.4))
ax.plot(pts.real, pts.imag, 'o', ms=11, color=S1, mec='white', mew=2)
ax.axhline(0, color=INK2, lw=1); ax.axvline(0, color=INK2, lw=1)
ax.set_xlim(-4, 4); ax.set_ylim(-2.2, 2.2); ax.set_xticks([-3, -1, 1, 3]); ax.set_yticks([-1, 1])
ax.set_xlabel('I (синфазная)'); ax.set_ylabel('Q (квадратурная)'); ax.set_title('8-КАМ: 8 точек, 3 бита на символ')
save(fig, 'sv_qam8')

# 5. Нули и полюсы фильтра ЛР4
b = [1, 4, 3, 2, 0.5]; a = [1, 0.5]
z = np.roots(b); p = np.roots(a)
fig, ax = plt.subplots(figsize=(5.4, 4.2))
th = np.linspace(0, 2 * np.pi, 400); ax.plot(np.cos(th), np.sin(th), color=INK2, lw=1, ls='--', label='единичная окружность')
ax.plot(z.real, z.imag, 'o', ms=10, mfc='white', mec=S1, mew=2, label='нули (корни b)')
ax.plot(p.real, p.imag, 'x', ms=11, color=S2, mew=2.5, label='полюс z = −0,5 (корень a)')
ax.set_aspect('equal'); ax.set_xlim(-3.6, 1.4); ax.set_ylim(-1.6, 1.6)
ax.set_xlabel('Re z'); ax.set_ylabel('Im z'); ax.set_title('H(z) из ЛР4: полюс внутри круга — устойчив')
ax.legend(loc='lower left', fontsize=9)
save(fig, 'sv_zplane')

# 6. Оконное ПФ сигнала z(n) из ЛР3: окно 128 и 512
fs = 1024; n = np.arange(1024); tt = (n - 512) / fs
zz = np.where(tt < 0, np.sin(2 * np.pi * 10 * tt), np.sin(2 * np.pi * 120 * tt))
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True)
for ax, nw in zip(axs, (128, 512)):
    F, T, Sx = signal.spectrogram(zz, fs, window='hamming', nperseg=nw, noverlap=nw - 16, mode='magnitude')
    ax.pcolormesh(T + tt[0], F, Sx, shading='nearest', cmap='Blues', rasterized=True)
    ax.set_ylim(0, 160); ax.set_xlim(-0.5, 0.5); ax.set_xlabel('время, с (центр окна)'); ax.grid(False)
    ax.set_title(f'окно {nw}: Δf = {fs / nw:g} Гц, Δt = {nw / fs:g} с')
axs[0].set_ylabel('частота, Гц')
save(fig, 'sv_stft')

# 7. Квантование (ЛР1, вариант 12: A = 12 В, q = 6 В)
t = np.arange(0, 0.1, 1e-3); A, q = 12, 6
s = A * np.sin(2 * np.pi * 22 * t); sq = q * np.round(s / q)
fig, ax = plt.subplots(figsize=(7.6, 3.2))
ax.plot(t, s, color=S1, lw=1.6, label='исходный')
ax.step(t, sq, where='post', color=S2, lw=1.6, label='квантованный, q = 6 В')
ax.plot(t, sq - s, color=S3, lw=1.2, label='ошибка, |e| ≤ q/2 = 3 В')
ax.set_xlabel('t, с'); ax.set_ylabel('В'); ax.legend(loc='lower left', fontsize=9, ncol=3)
ax.set_ylim(-16, 14)
save(fig, 'sv_quant')

# 8. Спектр одиночного импульса (ЛР2): τ = 1 с
f = np.linspace(-5, 5, 2001)
fig, ax = plt.subplots(figsize=(7.6, 3.0))
ax.plot(f, np.abs(np.sinc(f)), color=S1, label='|S(f)| = Aτ·|sinc(fτ)|')
ax.set_xticks(range(-5, 6)); ax.set_xlabel('f, Гц'); ax.set_ylabel('|S(f)|')
ax.legend(loc='upper right'); ax.set_title('Нули на f = m/τ = ±1, ±2 … Гц, главный лепесток шириной 2/τ')
save(fig, 'sv_sinc')
print('ok')
