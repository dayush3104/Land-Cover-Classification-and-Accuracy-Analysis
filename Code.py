import gradio as gr
import gradio.themes.base
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
import matplotlib.colors as mcolors
from tqdm import tqdm
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tempfile
import os
from datetime import datetime

# --- Theme ---
custom_theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="rose",
    neutral_hue="slate",
    font=["Comic Sans MS", "cursive"],
).set(
    body_background_fill="#f0f8ff",
    block_border_width="2px",
    block_border_color="#a0aec0",
    button_primary_background_fill="#3b82f6",
    button_primary_text_color="white",
)

# --- Class Map and Colors ---
class_map = {
    0: 'Water', 1: 'Trees', 2: 'Grass', 3: 'Flooded Vegetation',
    4: 'Crops', 5: 'Shrub and Scrub', 6: 'Built Area',
    7: 'Bare Ground', 8: 'Snow and Ice'
}
class_colors = {
    0: '#419bdf', 1: '#397d49', 2: '#88b053', 3: '#7a87c6',
    4: '#e49635', 5: '#dfc35a', 6: '#c4281b', 7: '#a59b8f', 8: '#b39fe1'
}
legend_html = "<div style='display:flex;flex-wrap:wrap;gap:10px;'>"
for cid, cname in class_map.items():
    color = class_colors[cid]
    legend_html += (
        f"<div style='display:flex;align-items:center;gap:5px;'>"
        f"<span style='display:inline-block;width:18px;height:18px;background:{color};border-radius:4px;border:1px solid #888;'></span>"
        f"<span style='font-size:14px;'>{cname}</span>"
        f"</div>"
    )
legend_html += "</div>"
# --- Utility Functions ---
def read_raster(path):
    with rasterio.open(path) as src:
        return src.read(), src.profile, src.read(1)

def plot_class_image(class_data, title):
    cmap = mcolors.ListedColormap([class_colors[i] for i in sorted(class_colors)])
    bounds = np.arange(-0.5, 9.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(class_data, cmap=cmap, norm=norm)
    cbar = plt.colorbar(im, ticks=range(9), ax=ax, shrink=0.7)
    cbar.ax.set_yticklabels([class_map[i] for i in range(9)])
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    temp_img_path = os.path.join(tempfile.gettempdir(), f"{title.replace(' ', '_')}_{datetime.now().strftime('%H%M%S')}.png")
    fig.savefig(temp_img_path)
    plt.close(fig)
    return temp_img_path

def generate_eda(class_data, pixel_size):
    flat = class_data.flatten()
    flat = flat[np.isin(flat, list(class_map))]
    counts = Counter(flat)
    total_pixels = sum(counts.values())
    pixel_area = pixel_size ** 2
    df = pd.DataFrame([
        {
            'Class ID': cid,
            'Class Name': class_map.get(cid, 'Unknown'),
            'Pixel Count': count,
            'Area (m²)': count * pixel_area
        } for cid, count in counts.items()
    ])
    df['Percentage (%)'] = 100 * df['Pixel Count'] / total_pixels
    df = df.sort_values('Pixel Count', ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(df['Class Name'], df['Pixel Count'], color='mediumseagreen')
    ax.set_xticklabels(df['Class Name'], rotation=30, ha='right')
    ax.set_ylabel("Pixel Count")
    ax.set_title("Land Cover Distribution by Pixel Count")
    for bar, pct in zip(bars, df['Percentage (%)']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{pct:.2f}%',
                ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    return df, fig

def compute_detailed_accuracy(cm):
    n_classes = cm.shape[0]
    total = np.sum(cm)
    pi_plus = cm.sum(axis=1) / total
    pj_plus = cm.sum(axis=0) / total
    user_acc = np.diag(cm) / cm.sum(axis=1)
    prod_acc = np.diag(cm) / cm.sum(axis=0)
    si_user = np.sqrt(user_acc * (1 - user_acc) / cm.sum(axis=1))
    si_prod = np.sqrt(prod_acc * (1 - prod_acc) / cm.sum(axis=0))
    ci_user = [(round(u - 1.96*s, 4), round(u + 1.96*s, 4)) for u, s in zip(user_acc, si_user)]
    ci_prod = [(round(u - 1.96*s, 4), round(u + 1.96*s, 4)) for u, s in zip(prod_acc, si_prod)]
    oa = np.trace(cm) / total
    po = oa
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / (total ** 2)
    kappa = (po - pe) / (1 - pe) if pe != 1 else None
    sigma_kappa = np.sqrt((po * (1 - po)) / (total * (1 - pe)**2)) if pe != 1 else None
    z = (kappa / sigma_kappa) if sigma_kappa else None
    tau = (po - pe) / (1 - pe) if pe != 1 else None
    overall_df = pd.DataFrame({
        'Metric': ['Overall Accuracy', 'Kappa', 'Sigma Kappa', 'Tau', 'Z-Statistic'],
        'Value': [round(oa, 4), round(kappa, 4), round(sigma_kappa, 4) if sigma_kappa else '',
                  round(tau, 4) if tau else '', round(z, 4) if z else '']
    })
    return overall_df

def save_confusion_matrix_excel(cm, oa, kappa, sigma_kappa, tau, z, class_names):
    total = np.sum(cm)
    pi_plus = cm.sum(axis=1) / total
    pj_plus = cm.sum(axis=0) / total
    user_acc = np.diag(cm) / cm.sum(axis=1)
    prod_acc = np.diag(cm) / cm.sum(axis=0)
    si_user = np.sqrt(user_acc * (1 - user_acc) / cm.sum(axis=1))
    si_prod = np.sqrt(prod_acc * (1 - prod_acc) / cm.sum(axis=0))
    ci_user = [(u - 1.96 * s, u + 1.96 * s) for u, s in zip(user_acc, si_user)]
    ci_prod = [(o - 1.96 * s, o + 1.96 * s) for o, s in zip(prod_acc, si_prod)]
    matrix_data = []
    for i in range(len(class_names)):
        row = [int(cm[i][j]) for j in range(len(class_names))]
        row.append(int(cm[i].sum()))
        row.append(round(pi_plus[i], 4))
        row.append(round(user_acc[i], 4))
        row.append(round(si_user[i], 4))
        row.append(f"{round(ci_user[i][0], 4)} - {round(ci_user[i][1], 4)}")
        matrix_data.append(row)
    columns = class_names + ['Total', 'pi+', 'Ci', 'si', '95% CI of Ci']
    df_matrix = pd.DataFrame(matrix_data, index=class_names, columns=columns)
    df_matrix.loc['Total'] = [int(cm[:, j].sum()) for j in range(len(class_names))] + [int(total), 1.0, '', '', '']
    df_matrix.loc["Producer's reliability (Oj)"] = [round(prod_acc[j], 4) for j in range(len(class_names))] + ['', '', '', '', '']
    df_matrix.loc['si (Prod)'] = [round(si_prod[j], 4) for j in range(len(class_names))] + ['', '', '', '', '']
    df_matrix.loc['95% CI of Oj'] = [f"{round(ci_prod[j][0], 4)} - {round(ci_prod[j][1], 4)}" for j in range(len(class_names))] + ['', '', '', '', '']
    overall_df = pd.DataFrame({
        'Metric': ['Overall Accuracy', 'Kappa', 'Sigma Kappa', 'Tau', 'Z-Statistic'],
        'Value': [round(oa, 4), round(kappa, 4), round(sigma_kappa, 4) if sigma_kappa else '',
                  round(tau, 4) if tau else '', round(z, 4) if z else '']
    })
    temp_dir = tempfile.gettempdir()
    file_name = f"Confusion_Matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    output_path = os.path.join(temp_dir, file_name)
    with pd.ExcelWriter(output_path) as writer:
        df_matrix.to_excel(writer, sheet_name='Confusion Matrix')
        overall_df.to_excel(writer, sheet_name='Accuracy Metrics', index=False)
    return output_path, df_matrix.style.to_html()

def extract_window_samples(img, labels, window_size):
    pad = window_size // 2
    h, w = labels.shape
    img_padded = np.pad(img, ((0, 0), (pad, pad), (pad, pad)), mode='reflect')
    label_padded = np.pad(labels, ((pad, pad), (pad, pad)), mode='reflect')
    samples = []
    targets = []
    
    for i in range(pad, h + pad):
        for j in range(pad, w + pad):
            label = label_padded[i, j]
            if label in class_map:
                window = img_padded[:, i - pad:i + pad + 1, j - pad:j + pad + 1]
                features = window.flatten()
                samples.append(features)
                targets.append(label)
    return np.array(samples), np.array(targets)

def sample_pixels(X, y, ratio, method):
    X_sampled = []
    y_sampled = []
    rng = np.random.default_rng(42)
    if method == "Stratified":
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            n = int(len(idx) * ratio)
            sampled_idx = rng.choice(idx, size=n, replace=False)
            X_sampled.append(X[sampled_idx])
            y_sampled.append(y[sampled_idx])
        return np.vstack(X_sampled), np.concatenate(y_sampled)
    elif method == "Random":
        n = int(len(y) * ratio)
        sampled_idx = rng.choice(len(y), size=n, replace=False)
        return X[sampled_idx], y[sampled_idx]

def classify(s2_img_file, label_img_file, window_size, ratio, method):
    s2_img, s2_profile, _ = read_raster(s2_img_file.name)
    labels_img, label_profile, class_data = read_raster(label_img_file.name)
    height, width = class_data.shape
    pixel_size = abs(label_profile['transform'][0])
    
    X, y = extract_window_samples(s2_img, class_data, window_size)
    if X.shape[0] == 0:
        return "No samples extracted, please check inputs."
    
    X_sampled, y_sampled = sample_pixels(X, y, ratio, method)
    eda_df_sampled, eda_fig_sampled = generate_eda(y_sampled.reshape(-1, 1), pixel_size)
    X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, stratify=y_sampled, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, max_depth=30, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)
    test_report = classification_report(y_test, y_pred_test, output_dict=True)
    report_df = pd.DataFrame(test_report).transpose().round(3)

    cm = confusion_matrix(y_test, y_pred_test)
    cm_fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=list(class_map.values()), yticklabels=list(class_map.values()))
    ax.set_title("Confusion Matrix (Test)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()

    overall_acc_df = compute_detailed_accuracy(cm)
    excel_path, styled_matrix_html = save_confusion_matrix_excel(
        cm=cm,
        oa=overall_acc_df.loc[0, 'Value'],
        kappa=overall_acc_df.loc[1, 'Value'],
        sigma_kappa=overall_acc_df.loc[2, 'Value'],
        tau=overall_acc_df.loc[3, 'Value'],
        z=overall_acc_df.loc[4, 'Value'],
        class_names=list(class_map.values())
    )
    return (
        eda_df_sampled,
        eda_fig_sampled,
        eda_fig_sampled.savefig("eda_plot.png") or "eda_plot.png",  # save plot if not already saved
        report_df,
        cm_fig,
        styled_matrix_html,
        overall_acc_df,
        excel_path,
        "✅ Classification completed successfully!"
    )
# --- Gradio App ---
with gr.Blocks(theme=custom_theme, title="Land Cover Classifier") as demo:
    gr.Markdown(
        "<h1 style='text-align: center; color: #3b82f6;'>🌍 Land Cover Classification & Accuracy Analyzer</h1>"
    )
    gr.HTML(legend_html, label="Class Color Legend")

    with gr.Accordion("📁 Upload Data", open=True):
        with gr.Row():
            s2_input = gr.File(label="Sentinel-2 Image (.tif)", file_types=[".tif"])
            label_input = gr.File(label="Label Image (.tif)", file_types=[".tif"])

    with gr.Accordion("⚙️ Parameters", open=False):
        with gr.Row():
            window_size_input = gr.Slider(minimum=1, maximum=31, step=2, value=5, label="Window Size (odd)")
            sampling_ratio = gr.Slider(minimum=0.05, maximum=1.0, step=0.05, value=0.2, label="Sampling Ratio")
            sampling_method = gr.Dropdown(choices=["Stratified", "Random"], value="Stratified", label="Sampling Method")

    status_text = gr.Textbox(label="Status", value="Waiting for input...", interactive=False)
    run_btn = gr.Button("🚀 Run Classification", variant="primary")

    with gr.Tab("📊 Sampled EDA"):
        eda_table = gr.Dataframe(label="Sampled EDA Table")
        eda_plot = gr.Plot(label="Sampled EDA Plot")
        eda_plot_download = gr.File(label="Download EDA Plot")

    with gr.Tab("📈 Classification Results"):
        report_table = gr.Dataframe(label="Classification Report")
        cm_plot = gr.Plot(label="Confusion Matrix")
        styled_cm = gr.HTML(label="Styled Confusion Matrix")
        accuracy_df = gr.Dataframe(label="Overall Accuracy Metrics")
        download_excel = gr.File(label="⬇️ Download Accuracy Report (.xlsx)")

    def run_classification(s2_img_file, label_img_file, window_size, ratio, method):
        result = classify(s2_img_file, label_img_file, window_size, ratio, method)
        if isinstance(result, str):
            return [None] * 8 + [result]
        (
            eda_df_sampled, eda_fig_sampled, eda_plot_path,
            report_df, cm_fig, styled_matrix_html,
            overall_acc_df, excel_path, message
        ) = result
        return (
            eda_df_sampled, eda_fig_sampled, eda_plot_path,
            report_df, cm_fig, styled_matrix_html,
            overall_acc_df, excel_path, message
        )

    run_btn.click(
        fn=run_classification,
        inputs=[s2_input, label_input, window_size_input, sampling_ratio, sampling_method],
        outputs=[
            eda_table, eda_plot, eda_plot_download,
            report_table, cm_plot, styled_cm,
            accuracy_df, download_excel, status_text
        ]
    )

demo.launch(share=True)