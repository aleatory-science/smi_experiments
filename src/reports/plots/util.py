from pathlib import Path

IMG_DIR = Path(__file__).parent.parent.parent.parent / "images"
IMG_DIR.mkdir(exist_ok=True)

if __name__ == "__main__":
    from article.scripts.hdi_plot import sine_wave_hdi
    from article.scripts.sineware_data_plot import sine_wave_data

    sine_wave_hdi(IMG_DIR, plot_format="png")
    sine_wave_data(IMG_DIR, plot_format="png")
