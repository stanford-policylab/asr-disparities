#!/usr/bin/env Rscript
rmarkdown::render("src/analysis.Rmd", output_format = "html_document", output_file = "analysis.html")