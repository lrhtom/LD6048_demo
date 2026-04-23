#!/usr/bin/env Rscript

# Draw a compact neural network architecture diagram for Section 4.
# It reads feature count from Section 3 output and exports a PNG figure.

args <- commandArgs(trailingOnly = TRUE)

get_arg_value <- function(args_vec, key, default_value) {
  prefix <- paste0("--", key, "=")
  matched <- args_vec[startsWith(args_vec, prefix)]
  if (length(matched) == 0) {
    return(default_value)
  }
  sub(prefix, "", matched[[1]])
}

parse_hidden_layers <- function(x) {
  parts <- strsplit(x, ",", fixed = TRUE)[[1]]
  vals <- suppressWarnings(as.integer(trimws(parts)))
  vals <- vals[!is.na(vals) & vals > 0]
  if (length(vals) == 0) {
    return(c(64L, 32L))
  }
  vals
}

safe_feature_count <- function(feature_csv_path) {
  if (!file.exists(feature_csv_path)) {
    return(46L)
  }
  df <- tryCatch(read.csv(feature_csv_path, stringsAsFactors = FALSE), error = function(e) NULL)
  if (is.null(df)) {
    return(46L)
  }
  n <- nrow(df)
  if (is.na(n) || n <= 0) {
    return(46L)
  }
  as.integer(n)
}

build_visible_nodes <- function(layer_size, max_visible = 12L) {
  if (layer_size <= max_visible) {
    return(list(index = seq_len(layer_size), omitted = 0L, ellipsis = FALSE))
  }

  head_n <- 6L
  tail_n <- 5L
  idx <- c(seq_len(head_n), (layer_size - tail_n + 1L):layer_size)
  list(index = idx, omitted = as.integer(layer_size - length(idx)), ellipsis = TRUE)
}

plot_network <- function(layer_sizes, output_path) {
  layer_names <- c("Input", paste0("Hidden ", seq_len(length(layer_sizes) - 2L)), "Output")
  n_layers <- length(layer_sizes)

  x_positions <- seq(0.08, 0.92, length.out = n_layers)
  visible <- lapply(layer_sizes, build_visible_nodes)

  all_y <- list()
  for (i in seq_len(n_layers)) {
    count <- length(visible[[i]]$index)
    all_y[[i]] <- seq(0.1, 0.9, length.out = max(count, 2L))[seq_len(count)]
  }

  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  png(output_path, width = 1800, height = 1100, res = 170)
  par(mar = c(1, 1, 4, 1), xaxs = "i", yaxs = "i")
  plot.new()
  plot.window(xlim = c(0, 1), ylim = c(0, 1))

  bg_col <- "#f7f9fc"
  rect(0, 0, 1, 1, col = bg_col, border = NA)

  # Draw layer-to-layer links.
  for (i in seq_len(n_layers - 1L)) {
    from_y <- all_y[[i]]
    to_y <- all_y[[i + 1L]]
    for (fy in from_y) {
      for (ty in to_y) {
        segments(
          x0 = x_positions[i] + 0.015,
          y0 = fy,
          x1 = x_positions[i + 1L] - 0.015,
          y1 = ty,
          col = adjustcolor("#6c7a89", alpha.f = 0.17),
          lwd = 0.8
        )
      }
    }
  }

  layer_cols <- c("#2e86de", rep("#10ac84", max(0L, n_layers - 2L)), "#ee5253")

  for (i in seq_len(n_layers)) {
    ys <- all_y[[i]]
    points(
      rep(x_positions[i], length(ys)),
      ys,
      pch = 21,
      bg = layer_cols[i],
      col = "white",
      cex = 1.5,
      lwd = 1.2
    )

    title_text <- paste0(layer_names[i], "\n", layer_sizes[i], " units")
    text(x_positions[i], 0.965, labels = title_text, cex = 1.0, font = 2)

    if (visible[[i]]$ellipsis) {
      text(x_positions[i], 0.5, labels = "...", cex = 1.2, col = "#222f3e", font = 2)
      text(
        x_positions[i],
        0.045,
        labels = paste0("+", visible[[i]]$omitted, " omitted"),
        cex = 0.9,
        col = "#576574"
      )
    }
  }

  title("Section 4 Neural Network Architecture", cex.main = 1.5, font.main = 2)
  mtext("Compact view: large layers are partially displayed for readability.", side = 3, line = 0.7, cex = 0.9)

  dev.off()
}

base_dir <- getwd()
feature_path <- file.path(base_dir, "img", "data_feathers", "section3_feature_names.csv")
hidden_arg <- get_arg_value(args, "hidden", "64,32")
out_arg <- get_arg_value(args, "output", file.path("img", "ai_data_img", "section4_nn_structure.png"))

input_dim <- safe_feature_count(feature_path)
hidden_layers <- parse_hidden_layers(hidden_arg)
layer_sizes <- c(input_dim, hidden_layers, 1L)

plot_network(layer_sizes, out_arg)

cat("=== Neural Network Structure Figure Generated ===\n")
cat(sprintf("Input features: %d\n", input_dim))
cat(sprintf("Hidden layers: %s\n", paste(hidden_layers, collapse = " -> ")))
cat("Output units: 1\n")
cat(sprintf("Saved to: %s\n", out_arg))
