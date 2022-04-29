pandoc --template algo --filter ./pandoc/minted.py --pdf-engine=xelatex --no-highlight --pdf-engine-opt="-shell-escape" -o template.tex --from markdown -V secnumdepth=2 -V --number-sections --toc -V include-before="\renewcommand\labelitemi{$\bullet$}" -V header-includes="\usepackage{minted}" -V geometry="margin=1cm" *-*.md
latexmk -xelatex -shell-escape template.tex -f
latexmk -c -f
