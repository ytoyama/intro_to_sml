\ProvidesClass{mathtoolkit}
\LoadClass{article}

\RequirePackage{amsmath}
\RequirePackage{amsfonts}
\RequirePackage{amssymb}
\RequirePackage{enumerate}
\RequirePackage{marvosym}

\author{Yota Toyama}


% commands

\newcommand\nth[1]{\title{Introduction to Statistical Machine Learning \\ Homework #1}}

\let\oldtext\text
\renewcommand\text[1]{\oldtext{ #1 }}
\newcommand\mif{\text{if}}
\newcommand\otherwise{\text{otherwise}}
\newcommand\refeq[1]{Eq. \ref{#1}}
\DeclareMathOperator*\argmin{argmin}
\DeclareMathOperator*\argmax{argmax}

\newcommand\field[1]{\mathbb{#1}}
\renewcommand\vec[1]{\mathbf{#1}}

\newcommand{\norm}[1]{\left\Vert #1\right\Vert}
\renewcommand\ker[1]{\mathrm{ker}(#1)}
\newcommand\im[1]{\mathrm{im}(#1)}
\newcommand\mspan[1]{\mathrm{Span}(#1)}
\newcommand\mdim[1]{\mathrm{dim}(#1)}
\newcommand\rank[1]{\mathrm{rank}(#1)}
\newcommand\mnull[1]{\mathrm{null}(#1)}
\renewcommand\deg[1]{\mathrm{degree}(#1)}
\newcommand\contradiction{\text{\Lightning}}

\newcommand\ip[2]{\langle  #1, #2 \rangle}
\newcommand\st{\text{s.t.}}

\let\oldtherefore\therefore
\renewcommand\therefore{\oldtherefore~}
\let\oldbecause\because
\renewcommand\because{\oldbecause~}

\newenvironment{p}{\begin{enumerate}}{\end{enumerate}}
\newenvironment{subp}{\begin{enumerate}[(a)]}{\end{enumerate}}
