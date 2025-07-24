#!/usr/bin/env Rscript
# prepare_datasets.R
# — Extract and save:
#    • carbs$D, carbs$C, carbs$S from mdatools
#    • NIR$xNIR, NIR$yGlcEtOH from chemometrics

# 1. Install/load required packages
if (!requireNamespace("mdatools", quietly = TRUE)) {
  install.packages("mdatools")
}
if (!requireNamespace("chemometrics", quietly = TRUE)) {
  install.packages("chemometrics")
}

library(mdatools)      # Raman carbs data
library(chemometrics)  # NIR fermentation data

# 2. Process 'carbs' dataset
data("carbs", package = "mdatools")        # carbs is a list with $D, $C, $S
D_carbs <- carbs$D                         # 21 × 1401
C_carbs <- carbs$C                         # 21 ×   3
S_carbs <- carbs$S                         # 1401 ×  3

# Add wavenumber axis (200–1600 cm⁻¹, 1401 points)
wn_carbs <- seq(200, 1600, length.out = nrow(S_carbs))
S_carbs_df <- data.frame(
  wavenumber = wn_carbs,
  S_carbs
)

# 3. Process 'NIR' dataset
data("NIR", package = "chemometrics")       # NIR is a list with xNIR, yGlcEtOH
D_nir <- NIR$xNIR                           # 166 × 235
C_nir <- NIR$yGlcEtOH                       # 166 ×   2
colnames(C_nir) <- c("glucose", "ethanol")

# 4. Write CSV files
outdir <- "."
dir.create(outdir, showWarnings = FALSE)

# Carbs outputs
write.csv(D_carbs,     file.path(outdir, "carbs_D_spectra.csv"),        row.names = FALSE)
write.csv(C_carbs,     file.path(outdir, "carbs_C_concentrations.csv"), row.names = FALSE)
write.csv(S_carbs_df,  file.path(outdir, "carbs_S_endmembers.csv"),     row.names = FALSE)

# NIR outputs
write.csv(D_nir,       file.path(outdir, "nir_D_spectra.csv"),          row.names = FALSE)
write.csv(C_nir,       file.path(outdir, "nir_C_concentrations.csv"),   row.names = FALSE)

message("Wrote CSVs to 'output/':\n",
        "  carbs_D_spectra.csv\n",
        "  carbs_C_concentrations.csv\n",
        "  carbs_S_endmembers.csv\n",
        "  nir_D_spectra.csv\n",
        "  nir_C_concentrations.csv")

