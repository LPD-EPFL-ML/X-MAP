files=( "aux" "bbl" "blg" "fdb_latexmk" "fls" "log" "synctex.gz" "dvi" "out" "toc" )

for file in "${files[@]}"
do
    find . -path ./fig -prune -o -name "*.$file" -exec rm -rf '{}' \;
done
