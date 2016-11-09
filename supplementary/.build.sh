NAME=technical_report

pdflatex ${NAME}.tex
bibtex ${NAME}
pdflatex ${NAME}.tex
pdflatex ${NAME}.tex
