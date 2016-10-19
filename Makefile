.SUFFIXES: .pdf .tex

.tex.pdf: ../introtosml.cls
	for i in 1 2; do TEXINPUTS=..:$$TEXINPUTS lualatex ${.IMPSRC}; done

default: notes.pdf

clean:
	rm -f *.pdf *.aux *.toc *.log
