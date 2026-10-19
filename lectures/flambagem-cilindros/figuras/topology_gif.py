# Topology optimization with GIF:
#   - topologia em cima
#   - função objetivo embaixo

from __future__ import division
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

import matplotlib
matplotlib.use("Agg")  # backend não interativo (bom para gerar figuras)
import matplotlib.pyplot as plt
from matplotlib import colors

import imageio.v2 as imageio   # pip install imageio
import io                      # para salvar frames em memória


# =========================================================
# Element stiffness matrix
# =========================================================
def lk():
    E = 1.0
    nu = 0.3
    k = np.array([
        1/2 - nu/6,
        1/8 + nu/8,
        -1/4 - nu/12,
        -1/8 + 3*nu/8,
        -1/4 + nu/12,
        -1/8 - nu/8,
        nu/6,
        1/8 - 3*nu/8
    ])
    KE = E / (1 - nu**2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
    ])
    return KE


# =========================================================
# Optimality criterion update
# =========================================================
def oc(nelx, nely, x, volfrac, dc, dv, g):
    l1 = 0.0
    l2 = 1e9
    move = 0.2
    xnew = np.zeros(nelx * nely)

    while (l2 - l1) / (l1 + l2) > 1e-3:
        lmid = 0.5 * (l2 + l1)
        x_candidate = x * np.sqrt(-dc / dv / lmid)
        xnew[:] = np.maximum(
            0.0,
            np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x_candidate)))
        )
        gt = g + np.sum(dv * (xnew - x))
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid
    return xnew, gt


# =========================================================
# Main driver
# =========================================================
def main(nelx, nely, volfrac, penal, rmin, ft):
    print("Minimum compliance problem with OC")
    print(f"ndes: {nelx} x {nely}")
    print(f"volfrac: {volfrac}, rmin: {rmin}, penal: {penal}")
    print("Filter method: " + ["Sensitivity based", "Density based"][ft])

    Emin = 1e-9
    Emax = 1.0

    ndof = 2 * (nelx + 1) * (nely + 1)

    # Design variables
    x = volfrac * np.ones(nely * nelx, dtype=float)
    xold = x.copy()
    xPhys = x.copy()
    g = 0.0

    # FE helpers
    KE = lk()
    edofMat = np.zeros((nelx * nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edofMat[el, :] = np.array([
                2 * n1 + 2, 2 * n1 + 3,
                2 * n2 + 2, 2 * n2 + 3,
                2 * n2,     2 * n2 + 1,
                2 * n1,     2 * n1 + 1
            ])

    iK = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()

    # Filter
    nfilter = int(nelx * nely * ((2 * (np.ceil(rmin) - 1) + 1)**2))
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    cc = 0
    for i in range(nelx):
        for j in range(nely):
            row = i * nely + j
            kk1 = int(max(i - (np.ceil(rmin) - 1), 0))
            kk2 = int(min(i + np.ceil(rmin), nelx))
            ll1 = int(max(j - (np.ceil(rmin) - 1), 0))
            ll2 = int(min(j + np.ceil(rmin), nely))
            for k in range(kk1, kk2):
                for l in range(ll1, ll2):
                    col = k * nely + l
                    fac = rmin - np.sqrt((i - k)**2 + (j - l)**2)
                    if fac > 0:
                        iH[cc] = row
                        jH[cc] = col
                        sH[cc] = fac
                        cc += 1

    H = coo_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsc()
    Hs = H.sum(1)

    # BCs and loads
    dofs = np.arange(2 * (nelx + 1) * (nely + 1))
    fixed = np.union1d(
        dofs[0:2 * (nely + 1):2],
        np.array([2 * (nelx + 1) * (nely + 1) - 1])
    )
    free = np.setdiff1d(dofs, fixed)

    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    f[1, 0] = -1.0

    # ---------- Figura com 2 subplots ----------
    obj_history = []
    frames = []
    max_frames = 80  # quantos frames vão para o GIF

    fig, (ax_img, ax_obj) = plt.subplots(
        2, 1,
        figsize=(6, 5),
        gridspec_kw={"height_ratios": [2, 2]}
    )

    im = ax_img.imshow(
        -xPhys.reshape((nelx, nely)).T,
        cmap='gray',
        interpolation='none',
        norm=colors.Normalize(vmin=-1, vmax=0)
    )
    ax_img.axis('off')

    line_obj, = ax_obj.plot([], [], color="#1f77b4", linewidth=2)
    ax_obj.set_xlabel("Iterações")          # remove x-label
    ax_obj.set_ylabel("Objetivo")          # remove y-label
    ax_obj.grid(alpha=0.3)

    # remove ticks and tick labels on both axes
    ax_obj.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False
    )
    ax_obj.grid(alpha=0.3)

    # Layout fixo: menos branco, sem vibração
    fig.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.05, hspace=0.05)

    # Loop
    loop = 0
    change = 1.0
    dv = np.ones(nely * nelx)
    dc = np.ones(nely * nelx)
    ce = np.ones(nely * nelx)
    y_fixed = None

    while change > 0.01 and loop < 2000:
        loop += 1

        # Build stiffness matrix
        sK = (KE.flatten()[np.newaxis].T *
              (Emin + (xPhys**penal) * (Emax - Emin))).flatten(order='F')
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K = K[free, :][:, free]

        # Solve
        u[free, 0] = spsolve(K, f[free, 0])

        # Objective and sensitivity
        ue = u[edofMat].reshape(nelx * nely, 8)
        ce[:] = (np.dot(ue, KE) * ue).sum(1)
        obj = ((Emin + xPhys**penal * (Emax - Emin)) * ce).sum()
        dc[:] = (-penal * xPhys**(penal - 1) * (Emax - Emin)) * ce
        dv[:] = np.ones(nely * nelx)

        # Filter sensitivities
        if ft == 0:
            dc[:] = (np.asarray((H * (x * dc))[np.newaxis].T / Hs)[:, 0] /
                     np.maximum(0.001, x))
        elif ft == 1:
            dc[:] = np.asarray(H * (dc[np.newaxis].T / Hs))[:, 0]
            dv[:] = np.asarray(H * (dv[np.newaxis].T / Hs))[:, 0]

        # OC update
        xold[:] = x
        x, g = oc(nelx, nely, x, volfrac, dc, dv, g)

        # Physical density
        if ft == 0:
            xPhys[:] = x
        elif ft == 1:
            xPhys[:] = np.asarray(H * x[np.newaxis].T / Hs)[:, 0]

        # Change
        change = np.linalg.norm(
            x.reshape(nelx * nely, 1) - xold.reshape(nelx * nely, 1),
            np.inf
        )

        # ---------- Atualizar plots ----------
        obj_history.append(obj)

        # Topologia
        im.set_array(-xPhys.reshape((nelx, nely)).T)

        # Histórico da função objetivo
        iters = np.arange(1, len(obj_history) + 1)
        line_obj.set_data(iters, obj_history)

        if y_fixed is None and len(obj_history) >= 1:
            y0 = obj_history[0]
            y_fixed = (0, y0 * 1.05)
            ax_obj.set_ylim(*y_fixed)
        elif y_fixed is not None:
            ax_obj.set_ylim(*y_fixed)

        ax_obj.set_xlim(1, max(len(obj_history), 2))

        # Desenhar figura
        fig.canvas.draw()

        # ---------- Capturar frame (como na 1ª versão, mas via savefig) ----------
        if loop <= max_frames:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)  # “foto” da figura nesta iteração
            buf.seek(0)
            frame = imageio.imread(buf)  # vira array, igual à 1ª versão
            frames.append(frame)
            buf.close()

        print(
            "it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(
                loop,
                obj,
                (g + volfrac * nelx * nely) / (nelx * nely),
                change
            )
        )

    # Salvar GIF
    if frames:
        imageio.mimsave("topopt_evolution_with_obj.gif", frames, fps=10)
        print("GIF saved as 'topopt_evolution_with_obj.gif'")


# =========================================================
# Real main
# =========================================================
if __name__ == "__main__":
    nelx = 180
    nely = 60
    volfrac = 0.4
    rmin = 5.4
    penal = 3.0
    ft = 1  # 0: sens, 1: dens

    main(nelx, nely, volfrac, penal, rmin, ft)