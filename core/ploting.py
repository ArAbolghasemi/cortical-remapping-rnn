
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_trainer_summary(
    trainer,
    axis_shift_step,
    fig_size=(6, 2.5),
    model_name='trainer',
    loss_title="Training loss across decoder switch",
    cursor_title_prefixes=("Before remap", "Early remap", "Late remap"),
    epoch_title="Epoch-wise mean cursor across training",
    remap_label="remap",
    train_label="train",
    eval_label="eval",
    baseline_label="baseline",
    task_label="task",
    late_label="late",
    cursor_label="cursor",
    target_label="target",
    decoder1_phase="decoder_1",
    decoder2_phase="decoder_2_remap",
    cursor_snapshot_which_before="last",
    cursor_snapshot_which_early="first",
    cursor_snapshot_which_late="last",
    loss_line_kwargs=None,
    eval_line_kwargs=None,
    remap_line_kwargs=None,
    cursor_line_kwargs=None,
    target_line_kwargs=None,
    baseline_line_kwargs=None,
    task_line_kwargs=None,
    late_line_kwargs=None,
    title_fontsize=10,
    sharey_cursor_panels=True,
    save_path=None,
    save_dpi=300,
    show=True,
    close_after_save=False,
    return_outputs=False
):
    """
    Plot three summary figures from trainer.history with the same plotting structure
    as the original code.

    Parameters
    ----------
    trainer : object
        Must have:
            - trainer.history.to_dict()
            - trainer.history.snapshots
            - trainer.history.records
    axis_shift_step : int or float
        X location for the vertical remap line.
    fig_size : tuple
        Figure size used for all figures.
    save_path : str or None
        Optional base path for saving figures as PDF.
        Examples:
            "results/trainer_summary"
            "results/myplots/"
        This function will save:
            <base>_loss.pdf
            <base>_cursor_target.pdf
            <base>_epoch_mean_cursor.pdf
    """

    hist = trainer.history.to_dict()
    snaps = trainer.history.snapshots

    loss_line_kwargs = {} if loss_line_kwargs is None else dict(loss_line_kwargs)
    eval_line_kwargs = {} if eval_line_kwargs is None else dict(eval_line_kwargs)
    remap_line_kwargs = {"linestyle": "--"} if remap_line_kwargs is None else dict(remap_line_kwargs)
    cursor_line_kwargs = {} if cursor_line_kwargs is None else dict(cursor_line_kwargs)
    target_line_kwargs = {} if target_line_kwargs is None else dict(target_line_kwargs)
    baseline_line_kwargs = {} if baseline_line_kwargs is None else dict(baseline_line_kwargs)
    task_line_kwargs = {} if task_line_kwargs is None else dict(task_line_kwargs)
    late_line_kwargs = {} if late_line_kwargs is None else dict(late_line_kwargs)

    def _parse_save_paths(save_path):
        if save_path is None:
            return None

        root, ext = os.path.splitext(save_path)
        if ext.lower() == ".pdf":
            base = root
        else:
            if save_path.endswith(os.sep) or (os.path.exists(save_path) and os.path.isdir(save_path)):
                os.makedirs(save_path, exist_ok=True)
                base = os.path.join(save_path, f"{model_name}_summary")
            else:
                directory = os.path.dirname(save_path)
                if directory:
                    os.makedirs(directory, exist_ok=True)
                base = save_path

        return {
            "loss": f"{base}_loss.pdf",
            "cursor_target": f"{base}_cursor_target.pdf",
            "epoch_mean_cursor": f"{base}_epoch_mean_cursor.pdf",
        }

    def _save_fig(fig, path):
        if path is not None:
            fig.savefig(path, format="pdf", bbox_inches="tight", dpi=save_dpi)

    def get_snapshot(phase=None, step=None, which="last"):
        candidates = snaps
        if phase is not None:
            candidates = [s for s in candidates if s["phase"] == phase]
        if step is not None:
            candidates = [s for s in candidates if s["step"] == step]

        if len(candidates) == 0:
            raise ValueError("No matching snapshot found.")

        candidates = sorted(candidates, key=lambda s: s["step"])

        if which == "first":
            return candidates[0]
        elif which == "last":
            return candidates[-1]
        else:
            raise ValueError("which must be 'first' or 'last'")

    save_paths = _parse_save_paths(save_path)
    figures = {}

    # ------------------------------------------------
    # 1) Loss curve across phases
    # ------------------------------------------------
    train_steps = [s for s, m in zip(hist["step"], hist["mode"]) if m == "train"]
    train_loss = [l for l, m in zip(hist["loss"], hist["mode"]) if m == "train"]

    eval_steps = [s for s, m in zip(hist["step"], hist["mode"]) if m == "eval"]
    eval_loss = [l for l, m in zip(hist["loss"], hist["mode"]) if m == "eval"]

    fig1 = plt.figure(figsize=fig_size)
    plt.plot(train_steps, train_loss, label=train_label, **loss_line_kwargs)
    plt.plot(eval_steps, eval_loss, label=eval_label, **eval_line_kwargs)
    plt.axvline(axis_shift_step, label=remap_label, **remap_line_kwargs)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(loss_title)
    plt.legend()
    plt.tight_layout()
    _save_fig(fig1, None if save_paths is None else save_paths["loss"])
    if show:
        plt.show()
    figures["loss"] = fig1
    if close_after_save and not show:
        plt.close(fig1)

    # ------------------------------------------------
    # 2) Cursor vs target:
    #    before remap / early remap / late remap
    # ------------------------------------------------
    snap_before = get_snapshot(phase=decoder1_phase, which=cursor_snapshot_which_before)
    snap_early = get_snapshot(phase=decoder2_phase, which=cursor_snapshot_which_early)
    snap_late = get_snapshot(phase=decoder2_phase, which=cursor_snapshot_which_late)

    fig2, axes = plt.subplots(1, 3, figsize=fig_size, sharey=sharey_cursor_panels)

    for ax, snap, title in zip(
        axes,
        [snap_before, snap_early, snap_late],
        list(cursor_title_prefixes),
    ):
        cursor = snap["cursor"].numpy()
        target = snap["target"].numpy()

        ax.plot(cursor, label=cursor_label, **cursor_line_kwargs)
        ax.plot(target, label=target_label, **target_line_kwargs)
        ax.set_title(f"{title} (step {snap['step']})", fontsize=title_fontsize)
        ax.set_xlabel("Time")

    axes[0].set_ylabel("Value")
    axes[0].legend()
    plt.tight_layout()
    _save_fig(fig2, None if save_paths is None else save_paths["cursor_target"])
    if show:
        plt.show()
    figures["cursor_target"] = fig2
    if close_after_save and not show:
        plt.close(fig2)

    # ------------------------------------------------
    # 3) Epoch-mean cursor over time
    # ------------------------------------------------
    train_records = [r for r in trainer.history.records if r["mode"] == "train"]
    steps_train = np.array([r["step"] for r in train_records])
    base_train = np.array([r["baseline_cursor_mean"] for r in train_records])
    task_train = np.array([r["task_cursor_mean"] for r in train_records])
    late_train = np.array([r["late_cursor_mean"] for r in train_records])

    fig3 = plt.figure(figsize=fig_size)
    plt.plot(steps_train, base_train, label=baseline_label, **baseline_line_kwargs)
    plt.plot(steps_train, task_train, label=task_label, **task_line_kwargs)
    plt.plot(steps_train, late_train, label=late_label, **late_line_kwargs)
    plt.axvline(axis_shift_step, label=remap_label, **remap_line_kwargs)
    plt.xlabel("Step")
    plt.ylabel("Mean cursor")
    plt.title(epoch_title)
    plt.legend()
    plt.tight_layout()
    _save_fig(fig3, None if save_paths is None else save_paths["epoch_mean_cursor"])
    if show:
        plt.show()
    figures["epoch_mean_cursor"] = fig3
    if close_after_save and not show:
        plt.close(fig3)

    if return_outputs:
        return {
            "figures": figures,
            "snapshots": {
                "before": snap_before,
                "early": snap_early,
                "late": snap_late,
            },
            "save_paths": save_paths,
        }




def plot_geometry_metrics(
    trainers,
    trainer_names,
    axis_shift_step,
    use_bci_only=True,
    window_size_trials=5,
    stride_trials=None,
    pca_dim_m=10,
    fig_size=(8, 6),
    global_alpha=1.0,
    marker=".",
    line_kwargs=None,
    remap_line_kwargs=None,
    dim_title="Effective dimensionality (95% variance)",
    var_title="Trajectory variance",
    align_title="Decoder-axis alignment with trajectory",
    angle_title="Consecutive manifold principal angles",
    x_label_windows="Window start trial",
    x_label_next="Next window start trial",
    y_label_dim="Dimensionality",
    y_label_var="Variance",
    y_label_align="Cosine alignment",
    y_label_angle="Angle (deg)",
    effective_dim_label="Effective dim (95%)",
    ambient_var_label="Ambient variance",
    cosine_label="Cosine",
    min_angle_label="Min angle",
    not_enough_windows_text="Not enough windows",
    legend=True,
    title_fontsize=None,
    save_path=None,
    save_dpi=300,
    show=True,
    close_after_save=False,
    return_outputs=False,
):
    """
    Plot geometry metrics for one or more trainers.

    Behavior
    --------
    - If one trainer: all curves are gray.
    - If more than one trainer: use seaborn Set1 palette if available,
      otherwise fall back to matplotlib's Set1 colormap.

    Parameters
    ----------
    trainers : trainer or list of trainers
    trainer_names : str or list of str
    axis_shift_step : int or float
        X location for the vertical remap line.
    save_path : str or None
        Optional path to save the figure as PDF.
        Examples:
            "results/geometry_metrics"
            "results/geometry_metrics.pdf"
    """

    from core import analyze_geometry_windows

    if stride_trials is None:
        stride_trials = window_size_trials

    if not isinstance(trainers, (list, tuple)):
        trainers = [trainers]
    if not isinstance(trainer_names, (list, tuple)):
        trainer_names = [trainer_names]

    if len(trainers) != len(trainer_names):
        raise ValueError("trainers and trainer_names must have the same length.")

    n_trainers = len(trainers)

    line_kwargs = {} if line_kwargs is None else dict(line_kwargs)
    remap_line_kwargs = {"linestyle": "--", "color": "k", "alpha": 0.8} if remap_line_kwargs is None else dict(remap_line_kwargs)

    # color choice
    if n_trainers == 1:
        colors = ["gray"]
    else:
        try:
            import seaborn as sns
            colors = sns.color_palette("Set1", n_colors=n_trainers)
        except Exception:
            cmap = plt.get_cmap("Set1")
            if n_trainers == 1:
                colors = [cmap(0)]
            else:
                colors = [cmap(i / max(n_trainers - 1, 1)) for i in range(n_trainers)]

    # save path handling
    pdf_path = None
    if save_path is not None:
        root, ext = os.path.splitext(save_path)
        if ext.lower() == ".pdf":
            pdf_path = save_path
        else:
            pdf_path = f"{save_path}.pdf"

        save_dir = os.path.dirname(pdf_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    analyses = []
    for trainer, name in zip(trainers, trainer_names):
        analysis = analyze_geometry_windows(
            snapshots=trainer.history.snapshots,
            phase=None,
            use_bci_only=use_bci_only,
            window_size_trials=window_size_trials,
            stride_trials=stride_trials,
            pca_dim_m=pca_dim_m,
        )
        analyses.append((name, analysis))

    fig, axes = plt.subplots(2, 2, figsize=fig_size)

    # -------------------------
    # 1) Effective dimensionality
    # -------------------------
    for (name, analysis), color in zip(analyses, colors):
        wm = analysis["window_metrics"]
        if len(wm) == 0:
            continue

        x_win = np.array([m["start_step"] for m in wm])
        d95 = np.array([m["effective_dim_95"] for m in wm])

        axes[0, 0].plot(
            x_win, d95,
            marker=marker,
            color=color,
            alpha=global_alpha,
            label=name,
            **line_kwargs,
        )

    axes[0, 0].axvline(axis_shift_step, **remap_line_kwargs)
    axes[0, 0].set_title(dim_title, fontsize=title_fontsize)
    axes[0, 0].set_ylabel(y_label_dim)

    # -------------------------
    # 2) Trajectory variance
    # -------------------------
    for (name, analysis), color in zip(analyses, colors):
        wm = analysis["window_metrics"]
        if len(wm) == 0:
            continue

        x_win = np.array([m["start_step"] for m in wm])
        var_amb = np.array([m["trajectory_variance_ambient"] for m in wm])

        axes[0, 1].plot(
            x_win, var_amb,
            marker=marker,
            color=color,
            alpha=global_alpha,
            label=name,
            **line_kwargs,
        )

    axes[0, 1].axvline(axis_shift_step, **remap_line_kwargs)
    axes[0, 1].set_title(var_title, fontsize=title_fontsize)
    axes[0, 1].set_ylabel(y_label_var)

    # -------------------------
    # 3) Decoder alignment cosine
    # -------------------------
    for (name, analysis), color in zip(analyses, colors):
        wm = analysis["window_metrics"]
        if len(wm) == 0:
            continue

        x_win = np.array([m["start_step"] for m in wm])
        align_cos = np.array([m["decoder_alignment_pointwise_cosine"] for m in wm], dtype=float)

        axes[1, 0].plot(
            x_win, align_cos,
            marker=marker,
            color=color,
            alpha=global_alpha,
            label=name,
            **line_kwargs,
        )

    axes[1, 0].axvline(axis_shift_step, **remap_line_kwargs)
    axes[1, 0].set_title(align_title, fontsize=title_fontsize)
    axes[1, 0].set_ylabel(y_label_align)
    axes[1, 0].set_xlabel(x_label_windows)
    if legend:
        axes[1, 0].legend()

    # -------------------------
    # 4) Consecutive manifold principal angles
    # -------------------------
    has_any_ca = False
    for (name, analysis), color in zip(analyses, colors):
        ca = analysis["consecutive_alignment"]
        if len(ca) == 0:
            continue

        has_any_ca = True
        x_ca = np.array([a["start_step_2"] for a in ca])
        min_ang = np.array([a["min_angle_deg"] for a in ca])

        axes[1, 1].plot(
            x_ca, min_ang,
            marker=marker,
            color=color,
            alpha=global_alpha,
            label=name,
            **line_kwargs,
        )

    if has_any_ca:
        axes[1, 1].axvline(axis_shift_step, **remap_line_kwargs)
        axes[1, 1].set_title(angle_title, fontsize=title_fontsize)
        axes[1, 1].set_xlabel(x_label_next)
        axes[1, 1].set_ylabel(y_label_angle)
        if legend:
            axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, not_enough_windows_text, ha="center", va="center")
        axes[1, 1].set_axis_off()

    plt.tight_layout()

    if pdf_path is not None:
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=save_dpi)

    if show:
        plt.show()

    if close_after_save and not show:
        plt.close(fig)

    if return_outputs:
        return {
            "figure": fig,
            "analyses": {name: analysis for name, analysis in analyses},
            "save_path": pdf_path,
        }

    return None

def _plot_color_strip(ax, colors, title=None, title_fontsize=12):
    import matplotlib.colors as mcolors

    strip = np.arange(len(colors))[None, :]
    cmap = mcolors.ListedColormap(colors)
    ax.imshow(strip, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize, pad=6)

def plot_global_pca_phase_trajectories(
    trainer,
    axis_shift_step,
    model_name=None,
    use_bci_only=True,
    global_pca_components=3,
    phase=None,
    sampled_idx=None,
    sample_every=3,
    fig_size=(18, 5),
    axis_palette_name="Set2",
    phase1_cmap_name="Blues_d",
    phase2_cmap_name="Oranges_d",
    axis1_label="Decoder axis 1",
    axis2_label="Decoder axis 2",
    axis_arrow_width=0.01,
    axis_arrow_head_width=0.04,
    axis_arrow_alpha=0.8,
    traj_linewidth=1.0,
    traj_alpha_start=0.15,
    traj_alpha_end=1.0,
    start_marker_size=18,
    start_marker_alpha=0.7,
    end_marker_size=28,
    end_marker_alpha=1.0,
    xlabel_fontsize=17,
    ylabel_fontsize=17,
    title_fontsize=20,
    tick_labelsize=17,
    equal_aspect=True,
    save_path=None,
    save_dpi=300,
    show=True,
    close_after_save=False,
    return_outputs=False,
    show_colormap_strips=False,
    colormap_strip_figsize=(8, 1.6),
    colormap_strip_titles=("Decoder axis 1", "Decoder axis 2"),
    colormap_strip_title_fontsize=12,
    save_colormap_strip_pdf=True,
):
    """
    Plot 2D projections of global PCA trajectories across two decoder phases.

    Parameters
    ----------
    trainer : object
        Trainer with trainer.history.snapshots
    axis_shift_step : int
        Step size for shifting the decoder axes.
    model_name : str or None
        Used in saved filename if save_path is provided.
    use_bci_only : bool
        Passed to fit_global_pca.
    global_pca_components : int
        Number of PCA components to compute.
    phase : str or None
        Passed to fit_global_pca.
    sampled_idx : array-like or None
        Specific trial indices to plot. If None, uses np.arange(0, n_trials, sample_every).
    sample_every : int
        Step size for trial subsampling when sampled_idx is None.
    save_path : str or None
        If provided:
        - if ends with .pdf, saves exactly there
        - otherwise saves into that folder / base path using model_name
    return_outputs : bool
        If True, return figure/global_pca/save_path. Otherwise return None.
    """
    from core import fit_global_pca

    global_pca = fit_global_pca(
        snapshots=trainer.history.snapshots,
        phase=phase,
        use_bci_only=use_bci_only,
        n_components=global_pca_components,
    )

    X_pca = np.asarray(global_pca["X_pca"])          # [n_trials, time, n_pca]
    A_pca = np.asarray(global_pca["A_pca"])          # [n_trials, n_pca]
    evr = np.asarray(global_pca["explained_variance_ratio"])
    trials = np.asarray(global_pca["selected_trials"])

    if X_pca.shape[-1] < 3:
        raise ValueError("Need at least 3 PCA components to make the requested plots.")

    n_trials_total = len(trials)
    if n_trials_total <= axis_shift_step + 1:
        raise ValueError(
            f"Not enough trials for decoder axis indexing: need at least {axis_shift_step + 2}, got {n_trials_total}."
        )

    decoder_axis_1 = A_pca[0, :]
    decoder_axis_2 = A_pca[axis_shift_step + 1, :]

    if sampled_idx is None:
        sampled_idx = np.arange(0, n_trials_total, sample_every)
    else:
        sampled_idx = np.asarray(sampled_idx)

    axis_palette = sns.color_palette(axis_palette_name, n_colors=2)
    phase1_cmap = sns.color_palette(phase1_cmap_name, n_colors=max(axis_shift_step, 1))
    phase2_n = max(n_trials_total - axis_shift_step, 1)
    phase2_cmap = sns.color_palette(phase2_cmap_name, n_colors=phase2_n)

    pairs = [
        (0, 1, "PC1", "PC2", evr[0], evr[1]),
        (0, 2, "PC1", "PC3", evr[0], evr[2]),
        (1, 2, "PC2", "PC3", evr[1], evr[2]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=fig_size)

    for ax, (i1, i2, lab1, lab2, var1, var2) in zip(axes, pairs):
        axis1_coords = decoder_axis_1[[i1, i2]]
        axis2_coords = decoder_axis_2[[i1, i2]]

        ax.arrow(
            0, 0,
            axis1_coords[0], axis1_coords[1],
            color=axis_palette[0],
            width=axis_arrow_width,
            head_width=axis_arrow_head_width,
            length_includes_head=True,
            alpha=axis_arrow_alpha,
            label=axis1_label,
        )
        ax.arrow(
            0, 0,
            axis2_coords[0], axis2_coords[1],
            color=axis_palette[1],
            width=axis_arrow_width,
            head_width=axis_arrow_head_width,
            length_includes_head=True,
            alpha=axis_arrow_alpha,
            label=axis2_label,
        )

        if i1 == 0 and i2 == 1:
            ax.legend()

        for i in sampled_idx:
            if i < 0 or i >= n_trials_total:
                continue

            pts = X_pca[i]
            if i < axis_shift_step:
                c = phase1_cmap[i]
            else:
                c = phase2_cmap[i - axis_shift_step]

            T = pts.shape[0]
            for t in range(T - 1):
                frac = t / max(T - 2, 1)
                alpha = traj_alpha_start + (traj_alpha_end - traj_alpha_start) * frac
                ax.plot(
                    pts[t:t+2, i1],
                    pts[t:t+2, i2],
                    color=(c[0], c[1], c[2], alpha),
                    linewidth=traj_linewidth,
                )

            ax.scatter(
                pts[0, i1], pts[0, i2],
                color=c,
                s=start_marker_size,
                alpha=start_marker_alpha,
            )
            ax.scatter(
                pts[-1, i1], pts[-1, i2],
                color=c,
                s=end_marker_size,
                alpha=end_marker_alpha,
            )

        ax.set_xlabel(f"{lab1} ({100 * var1:.1f}%)", fontsize=xlabel_fontsize)
        ax.set_ylabel(f"{lab2} ({100 * var2:.1f}%)", fontsize=ylabel_fontsize)
        ax.set_title(f"{lab1} vs {lab2}", fontsize=title_fontsize)
        ax.tick_params(axis="both", which="major", labelsize=tick_labelsize)

        if equal_aspect:
            ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()

    pdf_path = None
    if save_path is not None:
        save_name = "model" if model_name is None else model_name

        if save_path.lower().endswith(".pdf"):
            pdf_path = save_path
            save_dir = os.path.dirname(pdf_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
        else:
            os.makedirs(save_path, exist_ok=True)
            pdf_path = os.path.join(save_path, f"{save_name}_global_pca_phase_trajectories.pdf")

        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=save_dpi)

    if show:
        plt.show()

    if close_after_save and not show:
        plt.close(fig)

    colormap_fig = None
    colormap_pdf_path = None

    if show_colormap_strips:
        colormap_fig, axes_cm = plt.subplots(2, 1, figsize=colormap_strip_figsize)

        _plot_color_strip(
            axes_cm[0],
            phase1_cmap,
            title=colormap_strip_titles[0],
            title_fontsize=colormap_strip_title_fontsize,
        )
        _plot_color_strip(
            axes_cm[1],
            phase2_cmap,
            title=colormap_strip_titles[1],
            title_fontsize=colormap_strip_title_fontsize,
        )

        plt.tight_layout()

        if save_path is not None and save_colormap_strip_pdf:
            save_name = "model" if model_name is None else model_name
            if save_path.lower().endswith(".pdf"):
                base_root = os.path.splitext(save_path)[0]
                colormap_pdf_path = f"{base_root}_colormaps.pdf"
            else:
                colormap_pdf_path = os.path.join(
                    save_path, f"{save_name}_global_pca_phase_colormaps.pdf"
                )

            colormap_fig.savefig(
                colormap_pdf_path,
                format="pdf",
                bbox_inches="tight",
                dpi=save_dpi,
            )

        if show:
            plt.show()

        if close_after_save and not show:
            plt.close(colormap_fig)

    if return_outputs:
        return {
        "figure": fig,
        "colormap_figure": colormap_fig,
        "global_pca": global_pca,
        "save_path": pdf_path,
        "colormap_save_path": colormap_pdf_path,
        }
    return None

