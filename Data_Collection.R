## ============================================================
## R PIPELINE: MULTI-TIME U/V WIND
## ============================================================

## install.packages("RNCEP")   # run once if needed
library(RNCEP)

## ------------------------------------------------------------
## 1. Download U and V wind at 850 hPa over a region, Jan 2000
## ------------------------------------------------------------
uw <- NCEP.gather(
  variable       = "uwnd",
  level          = 850,
  months.minmax  = c(1, 1),      # January
  years.minmax   = c(2000, 2000),
  lat.southnorth = c(30, 70),
  lon.westeast   = c(-20, 40),
  reanalysis2    = FALSE
)

vw <- NCEP.gather(
  variable       = "vwnd",
  level          = 850,
  months.minmax  = c(1, 1),
  years.minmax   = c(2000, 2000),
  lat.southnorth = c(30, 70),
  lon.westeast   = c(-20, 40),
  reanalysis2    = FALSE
)

## ------------------------------------------------------------
## 2. Convert to long data frame
## ------------------------------------------------------------
df_uv <- NCEP.array2df(
  wx.data   = list(uw, vw),
  var.names = c("u", "v")
)
cat("Rows in df_uv:", nrow(df_uv), "\n")
head(df_uv)

## df_uv has columns: datetime, latitude, longitude, u, v (in m/s)

## ------------------------------------------------------------
## 3. Parse datetime and clean
## ------------------------------------------------------------
df_uv$datetime <- as.POSIXct(
  df_uv$datetime,
  format = "%Y_%m_%d_%H",   # e.g. "2000_01_01_00"
  tz     = "UTC"
)

if (any(is.na(df_uv$datetime))) {
  stop("Datetime parse failed, inspect unique(df_uv$datetime).")
}

df_uv$u <- as.numeric(df_uv$u)
df_uv$v <- as.numeric(df_uv$v)

df_uv <- subset(df_uv, is.finite(u) & is.finite(v))
cat("Rows in df_uv after cleanup:", nrow(df_uv), "\n")

## ------------------------------------------------------------
## 4. Choose a 12 x 12 spatial grid and ~20 time slices
## ------------------------------------------------------------
lat_unique <- sort(unique(df_uv$latitude))
lon_unique <- sort(unique(df_uv$longitude))
time_unique <- sort(unique(df_uv$datetime))

# pick 12 roughly evenly spaced latitudes and longitudes
n_lat_keep <- min(12, length(lat_unique))
n_lon_keep <- min(12, length(lon_unique))

lat_idx <- unique(round(seq(1, length(lat_unique), length.out = n_lat_keep)))
lon_idx <- unique(round(seq(1, length(lon_unique), length.out = n_lon_keep)))

lat_sel <- lat_unique[lat_idx]
lon_sel <- lon_unique[lon_idx]

# pick up to 20 time steps (replicates)
n_time_keep <- min(20, length(time_unique))
time_idx <- unique(round(seq(1, length(time_unique), length.out = n_time_keep)))
time_sel <- time_unique[time_idx]

cat("Selected latitudes:", lat_sel, "\n")
cat("Selected longitudes:", lon_sel, "\n")
cat("Selected times:", format(time_sel[1:min(5, length(time_sel))]), "...\n")

df_small <- subset(
  df_uv,
  latitude %in% lat_sel &
    longitude %in% lon_sel &
    datetime %in% time_sel
)

cat("Rows in df_small (raw selection):", nrow(df_small), "\n")

if (nrow(df_small) == 0) {
  stop("No data in selected subgrid and times.")
}

## ------------------------------------------------------------
## 5. Ensure we have a full grid for each time
##    (lat x lon must be complete; drop incomplete times)
## ------------------------------------------------------------
N_space <- length(lat_sel) * length(lon_sel)

complete_times <- sapply(time_sel, function(tt) {
  sum(df_small$datetime == tt) == N_space
})

time_sel_complete <- time_sel[complete_times]
cat("Number of complete time steps:", length(time_sel_complete), "\n")

if (length(time_sel_complete) == 0) {
  stop("No time step has complete 12x12 grid; reduce grid size or inspect data.")
}

df_small <- subset(df_small, datetime %in% time_sel_complete)
cat("Rows in df_small after requiring full grid:", nrow(df_small), "\n")

## ------------------------------------------------------------
## 6. Standardize u and v across ALL selected data (global)
## ------------------------------------------------------------
u_mean <- mean(df_small$u)
u_sd   <- sd(df_small$u)
v_mean <- mean(df_small$v)
v_sd   <- sd(df_small$v)

df_small$u_std <- (df_small$u - u_mean) / u_sd
df_small$v_std <- (df_small$v - v_mean) / v_sd

## ------------------------------------------------------------
## 7. Final columns & export to CSV
## ------------------------------------------------------------
df_c2 <- df_small[, c("datetime", "latitude", "longitude", "u_std", "v_std")]

out_file <- "uv_wind_multi_for_C2.csv"
write.csv(df_c2, out_file, row.names = FALSE)

cat("Wrote multi-time U/V dataset for C2 to:", out_file, "\n")
