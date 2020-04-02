TEX = pdflatex
TEXFLAGS = -interaction nonstopmode -halt-on-error -file-line-error --interaction=batchmode
CONVERT = convert
CONVERTFLAGS = -density 300 -quality 90 -resize 75\%
PDFCROP=pdfcrop

RM = rm -f
TEX_TEMPS = *.log *.out *.aux *.fdb_latexmk *.bbl *.dvi *.fls *.spl
CP = cp -r

FILES = $(wildcard *.tex)
FINALS = $(subst .tex,.gif,$(FILES))

OUTPUT_PREFIX=images/

all: check_deps $(FINALS)

%.gif: %.pdf
	$(CONVERT) $(CONVERTFLAGS) $< $@

%.pdf: %.tex
	$(TEX) $(TEXFLAGS) $<
	$(PDFCROP) $@ $@


.PHONY:
check_deps:
	$(TEX) --version
	$(CONVERT) --version
	$(PDFCROP) --version

.PHONY:
clean:
	$(RM) $(TEX_TEMPS)
	$(RM) *.gif *.pdf

.PHONY:
install:
	$(CP) *.gif $(OUTPUT_PREFIX)

# In case of typo
clena: clean
