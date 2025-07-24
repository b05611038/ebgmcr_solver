import pandas as pd

files = {
    "carbs_D (21×1401)":       "./carbs_D_spectra.csv",
    "carbs_C (21×3)":          "./carbs_C_concentrations.csv",
    "carbs_S (1401×4)":        "./carbs_S_endmembers.csv",
    "nir_D   (166×235)":       "./nir_D_spectra.csv",
    "nir_C   (166×2)":         "./nir_C_concentrations.csv"
}

for title, path in files.items():
    df = pd.read_csv(path)
    # 1) print title and shape
    print(f"\n=== {title} ===")
    print("Shape:", df.shape)
    # 2) print first 5 column names (and how many more)
    cols = list(df.columns)
    more = f", …(+{len(cols)-5})" if len(cols)>5 else ""
    print("Columns:", ", ".join(cols[:5]) + more)
    # 3) print a small head (first 3 rows × first 5 cols), rounded
    print(df.iloc[:3, :5].round(4).to_markdown(index=False))

