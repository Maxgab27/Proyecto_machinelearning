import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def generar(df, metrics, histories, prediction, terms, folder):
    folder.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style='whitegrid', palette='colorblind')

    def save(name):
        plt.tight_layout()
        plt.savefig(folder / name, dpi=140, bbox_inches='tight')
        plt.close()

    plt.figure(figsize=(12, 5))
    for col, label in [('close', 'Cierre'), ('Media_Movil_7', 'Media de 7 sesiones'),
                       ('Media_Movil_30', 'Media de 30 sesiones')]:
        plt.plot(df.Date, df[col], label=label, linewidth=1)
    plt.title('GGAL: evolución histórica y tendencias')
    plt.ylabel('Precio de cierre (unidad según CSV)')
    plt.xlabel('Fecha')
    plt.legend()
    save('evolucion_precio.png')
    plt.figure(figsize=(9, 5))
    sns.histplot(df.close, bins=40, kde=True)
    plt.xlabel('Precio de cierre'); plt.ylabel('Observaciones')
    plt.title('Distribución del precio de cierre')
    save('histograma_precios.png')
    plt.figure(figsize=(9, 5))
    sns.scatterplot(data=df, x='volume', y='close', alpha=.35, s=12)
    plt.xlabel('Volumen'); plt.ylabel('Precio de cierre')
    plt.title('Volumen y precio de cierre')
    save('dispersion_volumen_precio.png')
    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x='Objetivo', order=[0, 1])
    plt.xticks([0, 1], ['Baja / igual', 'Sube'])
    plt.ylabel('Sesiones'); plt.xlabel('Tendencia de la siguiente sesión')
    save('distribucion_tendencia.png')

    names = list(metrics)
    fig, axes = plt.subplots(1, len(names), figsize=(4*len(names), 4), squeeze=False)
    for ax, name in zip(axes[0], names):
        sns.heatmap(metrics[name]['matriz_confusion'], annot=True, fmt='d', cbar=False, ax=ax,
                    xticklabels=['Baja/igual', 'Sube'], yticklabels=['Baja/igual', 'Sube'])
        ax.set_title(name); ax.set_xlabel('Predicción'); ax.set_ylabel('Real')
    save('matrices_confusion.png')
    fig, ax = plt.subplots(figsize=(10, 5))
    pos = np.arange(len(names))
    for j, metric in enumerate(['exactitud', 'f1', 'roc_auc']):
        ax.bar(pos+(j-1)*.23, [metrics[name][metric] or 0 for name in names], width=.23, label=metric)
    ax.set_xticks(pos, names); ax.set_ylim(0, 1); ax.legend()
    ax.set_title('Evaluación en el mismo período de prueba')
    save('comparacion_modelos.png')
    plt.figure(figsize=(12, 5))
    plt.plot(prediction.Date, prediction.Objetivo.rolling(30).mean(), color='black', label='Subidas observadas (media 30 sesiones)')
    for name in names:
        plt.plot(prediction.Date, prediction[f'{name}_probabilidad'].rolling(30).mean(), label=name, alpha=.8)
    plt.ylim(0, 1); plt.ylabel('Probabilidad / proporción'); plt.xlabel('Fecha de predicción')
    plt.title('Tendencia observada y probabilidades estimadas (suavizadas)')
    plt.legend(fontsize=8)
    save('tendencias_predichas.png')
    for name, history in histories.items():
        plt.figure(figsize=(8, 4))
        plt.plot([r['epoch'] for r in history], [r['loss'] for r in history], label='Entrenamiento')
        plt.plot([r['epoch'] for r in history], [r['val_loss'] for r in history], label='Validación')
        plt.title(name); plt.xlabel('Época'); plt.ylabel('Entropía cruzada'); plt.legend()
        save(f'aprendizaje_{name}.png')
    plt.figure(figsize=(9, 6))
    sns.barplot(data=terms.head(15), x='tfidf_medio', y='termino', color='#176b87')
    plt.xlabel('TF-IDF medio por página'); plt.ylabel('Término')
    plt.title('Términos relevantes del reporte financiero')
    save('nlp_terminos.png')
